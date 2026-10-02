"""Laya-native evaluations.

1. laya_zero_shot  - Laya answering the task as a typed `choice` question (no training at all).
2. typed-decisions - the System One leaderboard (LocalLLaMA/typed-decisions): evaluate a Laya
                     checkpoint, then "ADAR-Cal": per (question type, option count) temperatures
                     fitted on the *train* split only and applied to test.
"""
import json
import os
import time

import numpy as np
from scipy.optimize import minimize_scalar

from .utils import LOG, save_json


# ---------------------------------------------------------------- zero-shot classification ----
def laya_zero_shot(agent, task, texts, cache_dir, split, batch_size=64, shard_size=2048):
    os.makedirs(cache_dir, exist_ok=True)
    q = {"label": {"type": "choice", "instructions": task.instruction,
                   "criteria": {l: task.descriptions[l] for l in task.labels}}}
    n_shards = (len(texts) + shard_size - 1) // shard_size
    parts = []
    t0 = time.time()
    for i in range(n_shards):
        path = os.path.join(cache_dir, f"{split}_zs_{i:04d}.npy")
        if not os.path.exists(path):
            chunk = texts[i * shard_size:(i + 1) * shard_size]
            res = agent.predict_batch(chunk, q, batch_size=batch_size, sort_by_length=len(chunk) > batch_size)
            p = np.array([[r["answers"]["label"]["probabilities"][l] for l in task.labels] for r in res],
                         dtype=np.float32)
            np.save(path + ".tmp.npy", p)
            os.replace(path + ".tmp.npy", path)
            LOG.info("  laya zero-shot %s shard %d/%d (%.0fs)", split, i + 1, n_shards, time.time() - t0)
        parts.append(np.load(path))
    p = np.concatenate(parts)
    p = np.clip(p, 1e-6, None)
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- typed-decisions ------------
TD_DATASET = "LocalLLaMA/typed-decisions"
QT = {"choice": 0, "score": 1, "noul": 2}


def _option_keys(question):
    t, crit = question["type"], question.get("criteria")
    if t == "choice":
        return list(crit.keys())
    if t == "noul":
        return ["false", "true"]
    n = len(crit) if isinstance(crit, list) else 4
    return [str(i) for i in range(n)]


def _pred_vector(answer, question):
    t = question["type"]
    if t == "noul":
        p1 = float(answer["noul"])
        return np.array([1 - p1, p1])
    keys = _option_keys(question)
    return np.array([float(answer["probabilities"].get(k, 0.0)) for k in keys])


def _gold_vector(gold_q, question):
    keys = _option_keys(question)
    g = np.array([float(gold_q["probabilities"].get(k, 0.0)) for k in keys])
    return g / g.sum() if g.sum() > 0 else np.full(len(keys), 1.0 / len(keys))


def load_typed_decisions(split):
    from datasets import load_dataset
    return load_dataset(TD_DATASET, "all", split=split)


def run_typed_decisions(agent, split, out_path, limit=None):
    """Predict every decision of a split. Resumable (JSONL, one line per case)."""
    ds = load_typed_decisions(split)
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                done.add(json.loads(line)["case"])
    n = len(ds) if limit is None else min(limit, len(ds))
    t0 = time.time()
    with open(out_path, "a") as f:
        for i in range(n):
            if i in done:
                continue
            row = ds[i]
            state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
            questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
            gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
            ts = time.perf_counter()
            res = agent.system_one(state, questions)
            ms = (time.perf_counter() - ts) * 1000
            recs = []
            for qid, q in questions.items():
                if qid not in gold or qid not in res["answers"]:
                    continue
                recs.append({"qid": qid, "type": q["type"], "k": len(_option_keys(q)),
                             "pred": _pred_vector(res["answers"][qid], q).tolist(),
                             "gold": _gold_vector(gold[qid], q).tolist()})
            f.write(json.dumps({"case": i, "latency_ms": ms, "decisions": recs}) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                LOG.info("  typed-decisions %s %d/%d (%.0fs)", split, i + 1, n, time.time() - t0)
    return out_path


def read_decisions(path):
    cases = [json.loads(l) for l in open(path)]
    decs = [d for c in cases for d in c["decisions"]]
    lat = [c["latency_ms"] for c in cases]
    return decs, lat


def td_metrics(decs, temps=None, bins=10):
    """Leaderboard-style metrics over decisions.
    accuracy: argmax(pred) == argmax(gold); brier: sum_k (p_k - g_k)^2 (soft gold) and vs one-hot;
    ece: top-label confidence vs argmax-correctness, 10 equal-width bins; kl: KL(gold || pred)."""
    acc, bs, bh, kl, conf, corr = [], [], [], [], [], []
    for d in decs:
        p = np.clip(np.asarray(d["pred"], dtype=float), 1e-6, 1.0)
        if temps is not None:
            t = temps.get(_bucket(d), temps.get(d["type"], 1.0))
            z = np.log(p) / t
            p = np.exp(z - z.max())
        p = p / p.sum()
        g = np.asarray(d["gold"], dtype=float)
        a = int(p.argmax() == g.argmax())
        acc.append(a)
        bs.append(float(((p - g) ** 2).sum()))
        bh.append(float(((p - np.eye(len(g))[g.argmax()]) ** 2).sum()))
        kl.append(float((g * (np.log(np.clip(g, 1e-12, 1)) - np.log(p))).sum()))
        conf.append(p.max())
        corr.append(a)
    conf, corr = np.array(conf), np.array(corr, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf <= edges[1])
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
    return {"decisions": len(decs), "accuracy": float(np.mean(acc)), "brier_soft": float(np.mean(bs)),
            "brier_onehot": float(np.mean(bh)), "ece": float(e), "kl": float(np.mean(kl))}


def _bucket(d):
    return f"{d['type']}:{d['k']}"


def fit_td_temperatures(decs, objective="soft", min_count=30):
    """Per (type, option-count) temperature, falling back to per-type, fitted on train decisions."""
    def loss(ds, t):
        tot = 0.0
        for d in ds:
            z = np.log(np.clip(np.asarray(d["pred"]), 1e-6, 1)) / t
            lp = z - z.max() - np.log(np.exp(z - z.max()).sum())
            g = np.asarray(d["gold"])
            tot -= (g * lp).sum() if objective == "soft" else lp[g.argmax()]
        return tot / len(ds)

    def fit(ds):
        r = minimize_scalar(lambda lt: loss(ds, np.exp(lt)), bounds=(np.log(0.2), np.log(5.0)), method="bounded")
        return float(np.exp(r.x))

    temps = {}
    for t in QT:
        ds = [d for d in decs if d["type"] == t]
        if len(ds) >= min_count:
            temps[t] = fit(ds)
    for b in sorted({_bucket(d) for d in decs}):
        ds = [d for d in decs if _bucket(d) == b]
        if len(ds) >= min_count:
            temps[b] = fit(ds)
    return temps


def typed_decisions_stage(agent, model_name, out_dir, train_limit=None, test_limit=None):
    os.makedirs(out_dir, exist_ok=True)
    tr_path = run_typed_decisions(agent, "train", os.path.join(out_dir, "train_predictions.jsonl"), train_limit)
    te_path = run_typed_decisions(agent, "test", os.path.join(out_dir, "test_predictions.jsonl"), test_limit)
    tr, _ = read_decisions(tr_path)
    te, lat = read_decisions(te_path)
    res = {"model": model_name, "shipped": {"test": td_metrics(te), "train": td_metrics(tr)},
           "latency_ms_p50": float(np.median(lat)), "latency_ms_p90": float(np.percentile(lat, 90))}
    variants = {}
    for obj in ("soft", "hard"):
        temps = fit_td_temperatures(tr, obj)
        variants[obj] = {"temperatures": temps, "train": td_metrics(tr, temps), "test": td_metrics(te, temps)}
    # choose the calibration variant on TRAIN only (test is never used for selection)
    pick = min(variants, key=lambda k: variants[k]["train"]["ece"])
    res["adar_cal"] = {"selected_on_train": pick, **variants}
    save_json(os.path.join(out_dir, "metrics.json"), res)
    LOG.info("typed-decisions %s shipped test: %s", model_name, res["shipped"]["test"])
    LOG.info("typed-decisions %s ADAR-Cal(%s) test: %s", model_name, pick, variants[pick]["test"])
    return {"shipped_test": res["shipped"]["test"], "adar_cal_test": variants[pick]["test"]}
