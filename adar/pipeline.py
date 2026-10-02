"""Stage functions. The notebook calls `Pipeline(cfg).run_all()` or the stages one by one.

Every stage is idempotent and resumable (see runstate.RunState); artefacts live under run_dir:

  run_dir/
    state.json  errors.log  run.log
    experts/<task>/           expert fold checkpoints, weights, oof/val/test probabilities
    sklearn/<task>/           TF-IDF + sklearn baselines
    emb/<task>/               cached frozen-encoder embeddings (sharded)
    probe/<task>/             Laya linear probe + stacking baseline
    zeroshot/<task>/          Laya zero-shot probabilities
    head/<task>/              ADAR heads (all seeds / ablations) + predictions
    router/<task>/            cascade + REINFORCE results
    results/<task>/           comparison tables, plots, QC report
    publish/adar-head-<task>/ publishable head (safetensors + config + model card + experts)
    typed_decisions/<model>/  leaderboard predictions + metrics
"""
import copy
import os
import re
import shutil

import numpy as np
import torch

from . import baselines, experts, head, laya_tasks, report, router
from .data import load_task
from .encoder import FrozenEncoder, cached_embed
from .runstate import RunState
from .utils import LOG, count_params, device_list, load_json, save_json, save_npz, setup_logging

DEFAULTS = {
    "run_dir": "/kaggle/working/adar_run",
    "tasks": ["ag_news", "emotion"],
    "task_sizes": {
        "ag_news": {"n_train": None, "n_val": 6000, "n_test": None, "max_len": 128, "expert_max_len": 64},
        "emotion": {"n_train": None, "n_val": None, "n_test": None, "max_len": 64, "expert_max_len": 48},
    },
    "data_source": "auto",          # auto | hf | csv (csv works for ag_news only)
    "local_data_dir": None,
    "n_folds": 3,
    "seed": 0,
    "encoder": {"backend": "laya", "model_id": "convaiinnovations/laya", "subfolder": None,
                "batch_size": 128, "shard_size": 4096},
    "experts": ["textcnn", "rnn", "lstm", "gru"],
    "expert_epochs": 10,
    "sklearn_models": ["sk_naive_bayes", "sk_logreg", "sk_linear_svm", "sk_random_forest", "sk_mlp"],
    "head_seeds": [0, 1, 2],
    "head_epochs": 30,
    "ablations": {"laya_only": {"use_experts": False}, "experts_only": {"use_state": False},
                  "no_attention": {"use_transformer": False}},
    "laya_zero_shot": True,
    "typed_decisions": {"enabled": True, "models": ["convaiinnovations/laya-typed-decisions", "convaiinnovations/laya"],
                        "train_limit": None, "test_limit": None},
    "publish": {"hf_repo_prefix": None, "private": True},
    "measure_cpu_latency": True,
    "continue_on_error": True,
    "time_budget_h": 11.5,
}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Pipeline:
    def __init__(self, cfg=None):
        self.cfg = merge(DEFAULTS, cfg)
        self.run_dir = self.cfg["run_dir"]
        os.makedirs(self.run_dir, exist_ok=True)
        setup_logging(os.path.join(self.run_dir, "run.log"))
        RunState.restore_from_inputs(self.run_dir)
        self.rs = RunState(self.run_dir, self.cfg["continue_on_error"], self.cfg["time_budget_h"])
        save_json(os.path.join(self.run_dir, "config.json"), self.cfg)
        self.devices = device_list()
        LOG.info("devices: %s", self.devices)
        self._tasks, self._encoders = {}, None

    # ------------------------------------------------------------------ helpers -------------
    def p(self, *parts):
        path = os.path.join(self.run_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def d(self, *parts):
        path = os.path.join(self.run_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def task(self, name):
        if name not in self._tasks:
            sz = self.cfg["task_sizes"][name]
            t = load_task(name, sz["n_train"], sz["n_val"], sz["n_test"], self.cfg["n_folds"], self.cfg["seed"],
                          local_dir=self.cfg["local_data_dir"], source=self.cfg["data_source"])
            fp_path = self.p("data", f"{name}_fingerprint.json")
            old = load_json(fp_path)
            if old and old["fingerprint"] != t.fingerprint():
                raise RuntimeError(f"{name}: data differs from the run being resumed ({old['fingerprint']} vs "
                                   f"{t.fingerprint()}); use the same data source or a new run_dir")
            save_json(fp_path, {"fingerprint": t.fingerprint(), "summary": t.summary()})
            self._tasks[name] = t
        return self._tasks[name]

    def encoders(self, max_len):
        e = self.cfg["encoder"]
        if self._encoders is None:
            devs = [d for d in self.devices if d.type == "cuda"] or [torch.device("cpu")]
            LOG.info("loading %s encoder (%s) on %s", e["backend"], e["model_id"], devs)
            self._encoders = [FrozenEncoder(e["backend"], e["model_id"], d, max_len, e.get("subfolder")) for d in devs]
        for enc in self._encoders:
            enc.max_len = max_len
        return self._encoders

    def free_encoders(self):
        self._encoders = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ stages --------------
    def stage_experts(self, name):
        t = self.task(name)
        return experts.run_experts(t, self.d("experts", name), self.devices[0], self.cfg["experts"],
                                   self.cfg["expert_epochs"], self.cfg["task_sizes"][name]["expert_max_len"],
                                   seed=self.cfg["seed"])

    def stage_sklearn(self, name):
        t = self.task(name)
        return baselines.run_tfidf_baselines(t, self.d("sklearn", name),
                                             self.cfg["sklearn_models"], self.cfg["seed"])

    def stage_embed(self, name):
        t = self.task(name)
        e = self.cfg["encoder"]
        ml = self.cfg["task_sizes"][name]["max_len"]
        encs = self.encoders(ml)
        meta = {"model": e["model_id"], "backend": e["backend"], "max_len": ml, "data": t.fingerprint()}
        cache = self.d("emb", name)
        dims = {}
        for split in ("train", "val", "test"):
            x, _ = t.split(split)
            z = cached_embed(encs, x, cache, split, e["shard_size"], e["batch_size"], meta)
            dims[split] = list(z.shape)
        opts = encs[0].embed(t.option_texts())
        np.save(os.path.join(cache, "options.npy"), opts)
        info = {"dims": dims, "encoder_dim": encs[0].dim, "encoder_params": count_params(encs[0].model)}
        if self.cfg["measure_cpu_latency"]:
            q = {"label": {"type": "choice", "instructions": t.instruction,
                           "criteria": {l: t.descriptions[l] for l in t.labels}}}
            lat = encs[0].cpu_latency_ms(t.x_test, question=q if e["backend"] == "laya" else None)
            info["encoder_cpu_latency_ms"] = lat["encoder"]
            info["zero_shot_cpu_latency_ms"] = lat["zero_shot"]
        save_json(os.path.join(cache, "info.json"), info)
        return info

    def load_emb(self, name):
        cache = os.path.join(self.run_dir, "emb", name)
        out = {}
        for split in ("train", "val", "test"):
            files = sorted(f for f in os.listdir(os.path.join(cache, split)) if re.fullmatch(r"shard_\d{5}\.npy", f))
            out[split] = np.concatenate([np.load(os.path.join(cache, split, f)).astype(np.float32) for f in files])
        out["options"] = np.load(os.path.join(cache, "options.npy")).astype(np.float32)
        return out

    def stage_probe(self, name):
        t = self.task(name)
        emb = self.load_emb(name)
        d = self.d("probe", name)
        info = {"probe": baselines.run_embedding_probe(t, emb, d)}
        # stacking baseline: one logistic regression over [embedding ; log expert probs]
        npz = os.path.join(d, "stacking_logreg.npz")
        if not os.path.exists(npz):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            oof, val, test = experts.load_expert_probs(self.d("experts", name), self.cfg["experts"])

            def feats(z, probs):
                return np.hstack([z] + [np.log(np.clip(probs[k], 1e-6, 1)) for k in self.cfg["experts"]])
            ftr = feats(emb["train"], oof)
            sc = StandardScaler().fit(ftr)
            clf = LogisticRegression(C=0.5, max_iter=3000).fit(sc.transform(ftr), t.y_train)
            pv = clf.predict_proba(sc.transform(feats(emb["val"], val)))
            pt = clf.predict_proba(sc.transform(feats(emb["test"], test)))
            save_npz(npz, val=pv.astype(np.float32), test=pt.astype(np.float32))
            info["stacking_val_acc"] = float((pv.argmax(1) == t.y_val).mean())
        return info

    def stage_zeroshot(self, name):
        t = self.task(name)
        if self.cfg["encoder"]["backend"] != "laya":
            LOG.warning("zero-shot needs the laya backend; skipped")
            return {"skipped": True}
        enc = self.encoders(self.cfg["task_sizes"][name]["max_len"])[0]
        d = self.d("zeroshot", name)
        pv = laya_tasks.laya_zero_shot(enc.agent, t, t.x_val, d, "val")
        pt = laya_tasks.laya_zero_shot(enc.agent, t, t.x_test, d, "test")
        save_npz(os.path.join(d, "laya_zero_shot.npz"), val=pv, test=pt)
        return {"val_acc": float((pv.argmax(1) == t.y_val).mean()), "test_acc": float((pt.argmax(1) == t.y_test).mean())}

    def stage_head(self, name):
        t = self.task(name)
        emb = self.load_emb(name)
        oof, val, test = experts.load_expert_probs(self.d("experts", name), self.cfg["experts"])
        kinds = self.cfg["experts"]
        P = {s: np.stack([d[k] for k in kinds], 1).astype(np.float32) for s, d in
             (("train", oof), ("val", val), ("test", test))}
        dev = self.devices[0]
        out_dir = self.d("head", name)
        variants = {"adar": {}, **self.cfg["ablations"]}
        summary = {}
        for vname, var in variants.items():
            no_exp = var.get("use_experts", True) is False
            seeds = self.cfg["head_seeds"] if vname == "adar" else self.cfg["head_seeds"][:1]
            for s in seeds:
                npz = os.path.join(out_dir, f"{vname}_s{s}.npz")
                if os.path.exists(npz):
                    continue
                model, hist = head.train_head(emb["train"], None if no_exp else P["train"], t.y_train,
                                              emb["val"], None if no_exp else P["val"], t.y_val, emb["options"],
                                              os.path.join(out_dir, "ckpt", f"{vname}_s{s}.pt"), dev, variant=var,
                                              seed=s, epochs=self.cfg["head_epochs"])
                pv = head.predict(model, emb["val"], emb["options"], None if no_exp else P["val"], dev, calibrated=False)
                pt = head.predict(model, emb["test"], emb["options"], None if no_exp else P["test"], dev, calibrated=False)
                arrays = {"val": pv["probs"], "test": pt["probs"], "act_val": pv["act_p"], "act_test": pt["act_p"]}
                if pt["expert_p"] is not None:
                    arrays.update(expert_p_test=pt["expert_p"], alpha_test=pt["alpha"])
                save_npz(npz, **arrays)
                meta = {"variant": vname, "seed": s, "history": hist, "temperature": float(model.temperature),
                        "params": sum(p.numel() for p in model.parameters()),
                        "cpu_latency_ms": head.head_cpu_latency_ms(model, emb["train"].shape[1], emb["options"],
                                                                   0 if no_exp else len(kinds))}
                save_json(npz.replace(".npz", ".json"), meta)
                if vname == "adar" and s == self.cfg["head_seeds"][0]:
                    self._publish(name, t, model, emb, kinds)
                summary[f"{vname}_s{s}"] = float((pt["probs"].argmax(1) == t.y_test).mean())
        return summary

    def _publish(self, name, t, model, emb, kinds):
        pub = os.path.join(self.run_dir, "publish", f"adar-head-{name}")
        e = self.cfg["encoder"]
        head.save_head(model, pub, {
            "task": name, "labels": t.labels, "descriptions": t.descriptions, "instruction": t.instruction,
            "encoder": {"backend": e["backend"], "model_id": e["model_id"], "subfolder": e.get("subfolder"),
                        "max_len": self.cfg["task_sizes"][name]["max_len"], "pooling": "mean+first_token"},
            "experts": {"kinds": kinds, "max_len": self.cfg["task_sizes"][name]["expert_max_len"],
                        "folds": self.cfg["n_folds"]},
        }, emb["options"])
        src = self.d("experts", name)
        dst = os.path.join(pub, "experts")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(src, "vocab.json"), dst)
        for f in os.listdir(os.path.join(src, "weights")):
            shutil.copy2(os.path.join(src, "weights", f), dst)
        LOG.info("publishable head written to %s", pub)

    def stage_router(self, name):
        t = self.task(name)
        kinds = self.cfg["experts"]
        oof, val, test = experts.load_expert_probs(self.d("experts", name), kinds)
        s0 = self.cfg["head_seeds"][0]
        h = np.load(os.path.join(self.run_dir, "head", name, f"adar_s{s0}.npz"))
        from .metrics import apply_temperature, fit_temperature, to_logits
        th = fit_temperature(to_logits(h["val"]), t.y_val)
        p1v, p1t = apply_temperature(h["val"], th), apply_temperature(h["test"], th)
        lat = self.latencies(name)
        cost0 = sum(lat[k] for k in kinds)
        cost1 = lat.get("encoder", 0.0) + lat.get("adar_head", 0.0)
        res = router.run_router(oof, val, test, kinds, t.y_train, t.y_val, t.y_test, p1v, p1t, cost0, cost1,
                                seed=self.cfg["seed"])
        d = self.d("router", name)
        save_npz(os.path.join(d, "cascade.npz"), test=res.pop("p_cascade_test"), tier0_test=res.pop("p_tier0_test"),
                 tier0_val=res.pop("p_tier0_val"))
        res.update({"cost0_ms": cost0, "cost1_ms": cost1})
        save_json(os.path.join(d, "router.json"), res)
        return res["chosen"]

    def latencies(self, name):
        lat = {}
        for k in self.cfg["experts"]:
            info = load_json(os.path.join(self.run_dir, "experts", name, f"{k}.json"), {})
            lat[k] = info.get("cpu_latency_ms", float("nan"))
        info = load_json(os.path.join(self.run_dir, "emb", name, "info.json"), {})
        lat["encoder"] = info.get("encoder_cpu_latency_ms", float("nan"))
        hj = load_json(os.path.join(self.run_dir, "head", name, f"adar_s{self.cfg['head_seeds'][0]}.json"), {})
        lat["adar_head"] = hj.get("cpu_latency_ms", float("nan"))
        return lat

    def stage_report(self, name):
        return report.build_task_report(self, name)

    def stage_typed_decisions(self, model_id):
        if self.cfg["encoder"]["backend"] != "laya":
            return {"skipped": "needs laya"}
        import laya
        dev = str(self.devices[0]) if self.devices[0].type == "cuda" else "cpu"
        agent = laya.load(model_id, device=dev)
        td = self.cfg["typed_decisions"]
        out = self.d("typed_decisions", model_id.split("/")[-1])
        try:
            return laya_tasks.typed_decisions_stage(agent, model_id, out, td["train_limit"], td["test_limit"])
        finally:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stage_bundle(self):
        return report.bundle(self)

    # ------------------------------------------------------------------ driver --------------
    def run_task(self, name):
        rs = self.rs
        rs.run(f"experts:{name}", self.stage_experts, name=name)
        rs.run(f"sklearn:{name}", self.stage_sklearn, name=name)
        rs.run(f"embed:{name}", self.stage_embed, name=name)
        rs.run(f"probe:{name}", self.stage_probe, deps=[f"embed:{name}", f"experts:{name}"], name=name)
        if self.cfg["laya_zero_shot"]:
            rs.run(f"zeroshot:{name}", self.stage_zeroshot, name=name)
        rs.run(f"head:{name}", self.stage_head, deps=[f"embed:{name}", f"experts:{name}"], name=name)
        rs.run(f"router:{name}", self.stage_router, deps=[f"head:{name}"], name=name)
        rs.run(f"report:{name}", self.stage_report, deps=[f"head:{name}"], force=True, name=name)

    def run_all(self):
        for name in self.cfg["tasks"]:
            self.run_task(name)
        self.free_encoders()
        td = self.cfg["typed_decisions"]
        if td["enabled"]:
            for m in td["models"]:
                self.rs.run(f"typed_decisions:{m}", self.stage_typed_decisions, model_id=m)
        self.rs.run("bundle", self.stage_bundle, force=True)
        LOG.info("stage summary:\n%s", self.rs.summary())
