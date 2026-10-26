"""Comparison tables, plots, QC gates, model cards and the output bundle.

Fairness rules applied to *every* model in the table:
  * one temperature per model fitted on the validation split, applied to test (raw ECE also shown)
  * identical test rows, identical metric code
  * paired bootstrap 95% CIs for every "ADAR beats X" claim
"""
import glob
import os
import shutil
import zipfile

import numpy as np

from . import metrics as M
from .utils import LOG, load_json, save_json

SK_LABELS = {"sk_naive_bayes": "Naive Bayes (TF-IDF)", "sk_logreg": "Logistic Regression (TF-IDF)",
             "sk_linear_svm": "Linear SVM (TF-IDF)", "sk_random_forest": "Random Forest (TF-IDF)",
             "sk_mlp": "sklearn MLPClassifier (TF-IDF)"}
EXPERT_LABELS = {"textcnn": "TextCNN", "rnn": "Bi-RNN", "lstm": "Bi-LSTM", "gru": "Bi-GRU"}
EXPERT_UNITS = {"textcnn": "I-II", "rnn": "III", "lstm": "III", "gru": "III"}


def encoder_label(pipe):
    e = pipe.cfg["encoder"]
    if e["backend"] == "laya":
        return "Laya"
    return f"{os.path.basename(str(e['model_id']).rstrip('/'))} (stand-in)"


def _row(key, label, family, unit, val, test, y_val, y_test, latency=None, params=None, calibrate=True):
    if calibrate and val is not None:
        t = M.fit_temperature(M.to_logits(val), y_val)
        pt = M.apply_temperature(test, t)
    else:
        t, pt = 1.0, test
    m = M.all_metrics(pt, y_test)
    m["ece_raw"] = M.ece(test, y_test)
    return {"key": key, "model": label, "family": family, "unit": unit, "temperature": t, "latency_ms": latency,
            "params": params, **m, "_probs": pt}


def collect(pipe, name):
    t = pipe.task(name)
    rd = pipe.run_dir
    yv, yt = t.y_val, t.y_test
    lat = pipe.latencies(name)
    emb_info = load_json(os.path.join(rd, "emb", name, "info.json"), {})
    enc_ms, enc_params = lat.get("encoder"), emb_info.get("encoder_params")
    kinds = pipe.cfg["experts"]
    enc = encoder_label(pipe)
    rows = []
    for k in pipe.cfg["sklearn_models"]:
        f = os.path.join(rd, "sklearn", name, f"{k}.npz")
        if os.path.exists(f):
            d, j = np.load(f), load_json(f.replace(".npz", ".json"), {})
            rows.append(_row(k, SK_LABELS.get(k, k), "scikit-learn", "-", d["val"], d["test"], yv, yt,
                             j.get("cpu_latency_ms")))
    exp_val, exp_test, exp_params = [], [], 0
    for k in kinds:
        f = os.path.join(rd, "experts", name, f"{k}.npz")
        if os.path.exists(f):
            d, j = np.load(f), load_json(f.replace(".npz", ".json"), {})
            rows.append(_row(k, EXPERT_LABELS[k], "Deep learning (syllabus)", EXPERT_UNITS[k], d["val"], d["test"],
                             yv, yt, j.get("cpu_latency_ms"), j.get("params")))
            exp_val.append(d["val"])
            exp_test.append(d["test"])
            exp_params += j.get("params") or 0
    tier0_ms = sum(lat[k] for k in kinds if k in lat)
    if len(exp_val) == len(kinds):
        rows.append(_row("expert_ensemble", "Average of syllabus experts", "Ensemble", "I-III",
                         np.mean(exp_val, 0), np.mean(exp_test, 0), yv, yt, tier0_ms, exp_params))
    f = os.path.join(rd, "zeroshot", name, "laya_zero_shot.npz")
    if os.path.exists(f):
        d = np.load(f)
        rows.append(_row("laya_zero_shot", "Laya zero-shot (typed choice)", "System One", "III",
                         d["val"], d["test"], yv, yt, emb_info.get("zero_shot_cpu_latency_ms"), enc_params))
        rows[-1]["ece_shipped"] = rows[-1]["ece_raw"]
    f = os.path.join(rd, "probe", name, "laya_linear_probe.npz")
    if os.path.exists(f):
        d = np.load(f)
        rows.append(_row("laya_linear_probe", f"{enc} encoder + Logistic Regression", "System One + sklearn", "III",
                         d["val"], d["test"], yv, yt, enc_ms, enc_params))
    f = os.path.join(rd, "probe", name, "stacking_logreg.npz")
    if os.path.exists(f):
        d = np.load(f)
        rows.append(_row("stacking_logreg", f"Stacking LogReg ({enc} emb + experts)", "Ensemble", "I-III",
                         d["val"], d["test"], yv, yt, (enc_ms or 0) + tier0_ms))
    # ADAR heads
    seeds = pipe.cfg["head_seeds"]
    adar_rows = []
    for s in seeds:
        f = os.path.join(rd, "head", name, f"adar_s{s}.npz")
        if os.path.exists(f):
            d, j = np.load(f), load_json(f.replace(".npz", ".json"), {})
            adar_rows.append(_row(f"adar_s{s}", f"ADAR (seed {s})", "ADAR (ours)", "I-IV", d["val"], d["test"], yv, yt,
                                  (enc_ms or 0) + tier0_ms + j.get("cpu_latency_ms", 0), j.get("params")))
    if adar_rows:
        agg = dict(adar_rows[0])
        agg.update({"key": "adar", "model": f"ADAR (ours, mean of {len(adar_rows)} seeds)"})
        for m in ("accuracy", "macro_f1", "nll", "brier", "ece", "ece_raw", "aurc"):
            vals = [r[m] for r in adar_rows]
            agg[m], agg[m + "_std"] = float(np.mean(vals)), float(np.std(vals))
        rows.append(agg)
    ab_lat = {"laya_only": lambda h: (enc_ms or 0) + h, "experts_only": lambda h: tier0_ms + h,
              "no_attention": lambda h: (enc_ms or 0) + tier0_ms + h}
    for v in pipe.cfg["ablations"]:
        f = os.path.join(rd, "head", name, f"{v}_s{seeds[0]}.npz")
        if os.path.exists(f):
            d, j = np.load(f), load_json(f.replace(".npz", ".json"), {})
            fn = ab_lat.get(v, lambda h: None)
            rows.append(_row(f"ablation_{v}", f"ADAR ablation: {v.replace('_', ' ')}", "Ablation", "-", d["val"],
                             d["test"], yv, yt, fn(j.get("cpu_latency_ms", 0)), j.get("params")))
    rj = load_json(os.path.join(rd, "router", name, "router.json"))
    f = os.path.join(rd, "router", name, "cascade.npz")
    if rj and os.path.exists(f):
        r = _row("adar_cascade", f"ADAR cascade (escalates {rj['chosen']['escalation_rate']:.0%})", "ADAR (ours)",
                 "I-IV", None, np.load(f)["test"], yv, yt, rj["chosen"]["mean_cost_ms"], calibrate=False)
        rows.append(r)
    return rows, adar_rows, rj


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def table_md(rows):
    cols = ["model", "family", "unit", "accuracy", "macro_f1", "nll", "brier", "ece", "ece_raw", "aurc",
            "latency_ms", "params"]
    head = ["Model", "Family", "Syllabus unit", "Accuracy", "Macro-F1", "NLL", "Brier", "ECE (TS)", "ECE (raw)",
            "AURC", "CPU ms/sample", "Params"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if c in ("accuracy", "macro_f1", "nll", "brier", "ece") and r.get(c + "_std") is not None:
                cells.append(f"{v:.4f} ± {r[c + '_std']:.4f}")
            elif c == "latency_ms":
                cells.append(_fmt(v, 2))
            elif c == "params":
                cells.append(f"{v / 1e6:.2f}M" if isinstance(v, (int, float)) and v else "-")
            else:
                cells.append(_fmt(v))
        if r["family"].startswith("ADAR"):
            cells[0] = f"**{cells[0]}**"
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def qc_checks(rows, adar_rows, rj, y_test):
    """Honest pass/fail gates. A claim passes only if the paired-bootstrap 95% CI excludes 0."""
    by = {r["key"]: r for r in rows}
    if not adar_rows:
        return [{"check": "ADAR trained", "pass": False, "detail": "no ADAR head results"}]
    a = adar_rows[0]["_probs"]
    checks = []

    def beats(key_list, label):
        cands = [by[k] for k in key_list if k in by]
        if not cands:
            return
        best = max(cands, key=lambda r: r["accuracy"])
        d, lo, hi = M.paired_bootstrap_delta(a, best["_probs"], y_test)
        checks.append({"check": f"ADAR accuracy > best {label} ({best['model']})", "pass": bool(lo > 0),
                       "detail": f"delta={d * 100:+.2f}pp, 95% CI [{lo * 100:+.2f}, {hi * 100:+.2f}]"})

    beats(list(SK_LABELS), "scikit-learn model")
    beats(list(EXPERT_LABELS), "syllabus deep model")
    beats(["expert_ensemble"], "expert ensemble")
    beats(["laya_zero_shot"], "Laya zero-shot")
    beats(["laya_linear_probe"], "Laya linear probe")
    beats(["stacking_logreg"], "stacking baseline")
    non_adar = [r for r in rows if not r["family"].startswith(("ADAR", "Ablation"))]
    if non_adar:
        for m in ("nll", "brier"):
            best = min(non_adar, key=lambda r: r[m])
            checks.append({"check": f"ADAR {m.upper()} lower than every baseline (best: {best['model']})",
                           "pass": bool(adar_rows[0][m] < best[m]),
                           "detail": f"ADAR {adar_rows[0][m]:.4f} vs {best[m]:.4f}"})
        best = min(non_adar, key=lambda r: r["ece"])
        checks.append({"check": f"ADAR ECE (after TS) <= best baseline ECE + 0.005 (best: {best['model']})",
                       "pass": bool(adar_rows[0]["ece"] <= best["ece"] + 0.005),
                       "detail": f"ADAR {adar_rows[0]['ece']:.4f} vs {best['ece']:.4f}"})
    for r in rows:
        if r["key"].startswith("ablation_"):
            d, lo, hi = M.paired_bootstrap_delta(a, r["_probs"], y_test)
            checks.append({"check": f"full ADAR >= {r['model']} (accuracy; NLL lower)",
                           "pass": bool(hi > 0 and adar_rows[0]["nll"] <= r["nll"] + 1e-4),
                           "detail": f"delta={d * 100:+.2f}pp CI [{lo * 100:+.2f}, {hi * 100:+.2f}], "
                                     f"NLL {adar_rows[0]['nll']:.4f} vs {r['nll']:.4f}"})
    if len(adar_rows) > 1:
        sd = float(np.std([r["accuracy"] for r in adar_rows]))
        checks.append({"check": "ADAR stable across seeds (accuracy std < 0.5pp)", "pass": bool(sd < 0.005),
                       "detail": f"std={sd * 100:.2f}pp over {len(adar_rows)} seeds"})
    if rj:
        c = rj["chosen"]
        ok = c["accuracy"] >= c["always_tier1_accuracy"] - 0.005 and c["escalation_rate"] < 0.95
        checks.append({"check": "Cascade keeps accuracy within 0.5pp of always-Laya while skipping Laya calls",
                       "pass": bool(ok),
                       "detail": f"acc {c['accuracy']:.4f} vs {c['always_tier1_accuracy']:.4f}; Laya called on "
                                 f"{c['escalation_rate']:.0%} of inputs; mean cost {c['mean_cost_ms']:.1f} vs "
                                 f"{c['always_tier1_cost_ms']:.1f} ms"})
    return checks


def plots(rows, adar_rows, rj, y_test, out_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {r["key"]: r for r in rows}
    os.makedirs(out_dir, exist_ok=True)
    if not adar_rows:
        return
    a = adar_rows[0]
    base = [r for r in rows if not r["family"].startswith(("ADAR", "Ablation"))]
    best_dl = max([by[k] for k in EXPERT_LABELS if k in by], key=lambda r: r["accuracy"], default=None)
    best_sk = max([by[k] for k in SK_LABELS if k in by], key=lambda r: r["accuracy"], default=None)
    probe = by.get("laya_linear_probe")
    # 1. reliability (raw = before temperature scaling, the honest out-of-the-box view)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="#888", lw=1, label="perfect calibration")
    for r, c in ((a, "#c2410c"), (best_dl, "#2563eb"), (best_sk, "#16a34a"), (probe, "#7c3aed")):
        if r is None:
            continue
        bins = M.reliability_bins(r["_probs"], y_test)
        ax.plot([b[1] for b in bins], [b[2] for b in bins], marker="o", c=c,
                label=f"{r['model'][:48]} (ECE {r['ece']:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability after temperature scaling — {title}")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=150)
    plt.close(fig)
    # 2. accuracy vs CPU latency
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for r in base + [a]:
        if r.get("latency_ms"):
            is_a = r["family"].startswith("ADAR")
            ax.scatter(r["latency_ms"], r["accuracy"], s=60 if is_a else 30, c="#c2410c" if is_a else "#475569",
                       zorder=3)
            ax.annotate(r["model"][:28], (r["latency_ms"], r["accuracy"]), fontsize=6.5, xytext=(4, 2),
                        textcoords="offset points")
    if rj:
        cv = sorted(rj["curve"], key=lambda p: p["mean_cost_ms"])
        ax.plot([p["mean_cost_ms"] for p in cv], [p["accuracy"] for p in cv], c="#c2410c", lw=1.5,
                label="ADAR cascade (threshold sweep)")
        ax.scatter([p["mean_cost_ms"] for p in rj["reinforce"]], [p["accuracy"] for p in rj["reinforce"]],
                   marker="x", c="#9a3412", label="ADAR cascade (REINFORCE policy)", zorder=4)
        ax.legend(fontsize=7, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("CPU latency per input (ms, batch 1, log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Accuracy vs. cost — {title}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_latency.png"), dpi=150)
    plt.close(fig)
    # 3. risk-coverage
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for r, c in ((a, "#c2410c"), (best_dl, "#2563eb"), (best_sk, "#16a34a"), (probe, "#7c3aed")):
        if r is None:
            continue
        rc = M.risk_coverage(r["_probs"], y_test)
        ax.plot([x[0] for x in rc], [x[1] for x in rc], c=c, label=f"{r['model'][:48]} (AURC {r['aurc']:.4f})")
    ax.set_xlabel("coverage (fraction of inputs answered)")
    ax.set_ylabel("accuracy on answered inputs")
    ax.set_title(f"Selective prediction — {title}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=150)
    plt.close(fig)


def routing_analysis(pipe, name):
    """How the head routes between experts: mean alpha per expert vs. each expert's real accuracy."""
    t = pipe.task(name)
    f = os.path.join(pipe.run_dir, "head", name, f"adar_s{pipe.cfg['head_seeds'][0]}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if "alpha_test" not in d:
        return None
    kinds = pipe.cfg["experts"]
    out = {}
    for i, k in enumerate(kinds):
        p = np.load(os.path.join(pipe.run_dir, "experts", name, f"{k}.npz"))["test"]
        right = (p.argmax(1) == t.y_test)
        ep = d["expert_p_test"][:, i]
        out[k] = {"mean_alpha": float(d["alpha_test"][:, i].mean()), "predicted_p_correct": float(ep.mean()),
                  "actual_accuracy": float(right.mean()),
                  "noul_auc": _auc(ep, right)}
    return out


def _auc(score, label):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(label.astype(int), score))
    except ValueError:
        return None


def build_task_report(pipe, name):
    t = pipe.task(name)
    out_dir = pipe.d("results", name)
    rows, adar_rows, rj = collect(pipe, name)
    checks = qc_checks(rows, adar_rows, rj, t.y_test)
    plots(rows, adar_rows, rj, t.y_test, out_dir, name)
    routing = routing_analysis(pipe, name)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    import csv
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
        keys = sorted({k for r in clean for k in r}, key=lambda k: (k != "model", k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(clean)
    save_json(os.path.join(out_dir, "comparison.json"), clean)
    save_json(os.path.join(out_dir, "qc.json"), checks)
    if routing:
        save_json(os.path.join(out_dir, "routing.json"), routing)
    ordered = sorted(rows, key=lambda r: (r["family"].startswith(("ADAR", "Ablation")), r["accuracy"]))
    md = [f"# ADAR results — {name}", "", t.summary(), "",
          "All probabilistic models are temperature-scaled on the validation split (ECE raw = before). "
          "CPU latency = median batch-1 time on the run's CPU. Test rows are identical for every model.", "",
          table_md(ordered), "", "## QC gates (paired bootstrap, 95% CI)", ""]
    for c in checks:
        md.append(f"- {'PASS' if c['pass'] else 'FAIL'} — {c['check']}: {c['detail']}")
    if routing:
        md += ["", "## How ADAR routes between the syllabus experts (test set)", "",
               "| Expert | mean routing weight α | predicted P(correct) | actual accuracy | AUC of P(correct) |",
               "|---|---|---|---|---|"]
        for k, v in routing.items():
            md.append(f"| {EXPERT_LABELS[k]} | {v['mean_alpha']:.3f} | {v['predicted_p_correct']:.3f} | "
                      f"{v['actual_accuracy']:.3f} | {_fmt(v['noul_auc'], 3)} |")
    if rj:
        c = rj["chosen"]
        md += ["", "## Cascade (cheap experts first, Laya + ADAR only when needed)", "",
               f"- tier-0 (experts) accuracy {c['tier0_accuracy']:.4f} at {rj['cost0_ms']:.2f} ms",
               f"- always escalate: accuracy {c['always_tier1_accuracy']:.4f} at {c['always_tier1_cost_ms']:.1f} ms",
               f"- chosen threshold (picked on validation): accuracy {c['accuracy']:.4f}, Laya called on "
               f"{c['escalation_rate']:.1%} of inputs, mean {c['mean_cost_ms']:.1f} ms",
               f"- oracle router (upper bound): accuracy {c['oracle_accuracy']:.4f} escalating "
               f"{c['oracle_escalation_rate']:.1%}",
               "- REINFORCE policy points: " + "; ".join(
                   f"λ={p['lambda']}: acc {p['accuracy']:.4f} @ {p['escalation_rate']:.0%} escalated"
                   for p in rj["reinforce"])]
    md += ["", "![reliability](reliability.png) ![accuracy vs latency](accuracy_vs_latency.png) "
               "![risk coverage](risk_coverage.png)"]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    LOG.info("\n%s", "\n".join(md[:len(ordered) + 12]))
    n_pass = sum(c["pass"] for c in checks)
    return {"qc_pass": n_pass, "qc_total": len(checks),
            "adar_accuracy": adar_rows[0]["accuracy"] if adar_rows else None}


HF_DATASETS = {"ag_news": "fancyzhx/ag_news", "emotion": "dair-ai/emotion"}

MODEL_CARD = """---
license: apache-2.0
base_model: {base}
library_name: pytorch
pipeline_tag: text-classification
tags: [system-one, laya, decision-model, calibration, routing, text-classification, adar]
datasets: [{dataset}]
metrics: [accuracy, f1]
model-index:
- name: adar-head-{repo_task}
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: {task}
      type: {dataset}
      split: test
    metrics:
    - type: accuracy
      value: {acc:.4f}
      name: Accuracy (mean of 3 seeds)
    - type: f1
      value: {f1:.4f}
      name: Macro F1 (mean of 3 seeds)
---

# ADAR head for {task} (on top of {base})

**ADAR — Adaptive Deep Architecture Routing** is a System-One decision head. For every input it
reads the frozen Laya (System-One) embedding, one token per answer option (built from the option's
text description) and one token per syllabus expert (TextCNN / RNN / LSTM / GRU), mixes them with a
2-layer transformer and returns typed, calibrated outputs:

* `choice` — a calibrated distribution over {labels}
* `noul` per expert — P(this expert is right on this input), used to route between architectures
* `act` — P(the head's own answer is right), used to abstain or escalate

Trained with strictly proper scoring rules (log score + Brier), temperature-scaled on validation data.

## Results ({task} test set, {n_test} rows)

{table}

QC gates: {qc}

## Files

* `model.safetensors`, `config.json` — the head ({params} parameters) and its temperature
* `options.safetensors` — encoder embeddings of the option descriptions
* `experts/` — the syllabus experts (one model per stacking fold) and their vocabulary

## Usage

```python
from adar.inference import ADARClassifier
clf = ADARClassifier.from_pretrained("path/to/this/folder", device="cuda")
clf.predict(["NASA's new telescope captured images of a distant galaxy cluster."])
```

## Limitations

* The head is trained for this label set; new labels need new option descriptions *and* retraining
  of the experts (the Laya-only ablation head works with descriptions alone).
* Numbers come from a single train/val/test protocol; see `REPORT.md` for confidence intervals.
"""


def model_cards(pipe):
    for name in pipe.cfg["tasks"]:
        pub = os.path.join(pipe.run_dir, "publish", f"adar-head-{name}")
        res = os.path.join(pipe.run_dir, "results", name)
        if not (os.path.isdir(pub) and os.path.exists(os.path.join(res, "comparison.json"))):
            continue
        rows = load_json(os.path.join(res, "comparison.json"))
        checks = load_json(os.path.join(res, "qc.json"), [])
        cfg = load_json(os.path.join(pub, "config.json"))
        keep = [r for r in rows if r["key"] in ("adar", "adar_cascade", "laya_zero_shot", "laya_linear_probe",
                                                  "stacking_logreg", "expert_ensemble", "sk_logreg", "gru", "lstm",
                                                  "textcnn")]
        keep.sort(key=lambda r: -r["accuracy"])
        adar = next((r for r in rows if r["key"] == "adar"), {"accuracy": float("nan"), "macro_f1": float("nan")})
        card = MODEL_CARD.format(base=pipe.cfg["encoder"]["model_id"], dataset=HF_DATASETS.get(name, name), task=name,
                                 repo_task=name.replace("_", "-"), acc=adar["accuracy"], f1=adar["macro_f1"],
                                 labels=", ".join(cfg["labels"]), n_test=len(pipe.task(name).y_test),
                                 table=table_md(keep), params=cfg["params"],
                                 qc=f"{sum(c['pass'] for c in checks)}/{len(checks)} passed (see REPORT.md)")
        with open(os.path.join(pub, "README.md"), "w") as f:
            f.write(card)
        shutil.copy2(os.path.join(res, "REPORT.md"), os.path.join(pub, "REPORT.md"))
        for png in glob.glob(os.path.join(res, "*.png")):
            shutil.copy2(png, pub)


def summary_md(pipe):
    lines = ["# ADAR run summary", ""]
    for name in pipe.cfg["tasks"]:
        rep = os.path.join(pipe.run_dir, "results", name, "REPORT.md")
        if os.path.exists(rep):
            lines += [open(rep).read(), ""]
    for m in pipe.cfg["typed_decisions"]["models"]:
        f = os.path.join(pipe.run_dir, "typed_decisions", m.split("/")[-1], "metrics.json")
        if os.path.exists(f):
            r = load_json(f)
            pick = r["adar_cal"]["selected_on_train"]
            lines += [f"## typed-decisions — {m}", "",
                      "| Variant | Accuracy | Brier (soft gold) | Brier (one-hot) | ECE (10 bins) | KL |", "|---|---|---|---|---|---|"]
            for lbl, mm in (("as shipped", r["shipped"]["test"]),
                            (f"+ ADAR-Cal ({pick}, fitted on train)", r["adar_cal"][pick]["test"])):
                lines.append(f"| {lbl} | {mm['accuracy']:.4f} | {mm['brier_soft']:.4f} | {mm['brier_onehot']:.4f} | "
                             f"{mm['ece']:.4f} | {mm['kl']:.4f} |")
            lines += ["", f"p50 latency {r['latency_ms_p50']:.1f} ms per case (5 questions) on this GPU.", ""]
    rs = load_json(os.path.join(pipe.run_dir, "state.json"), {"stages": {}})
    lines += ["## Stage status", "", "```"] + [f"{v.get('status'):>11}  {k}" for k, v in rs["stages"].items()] + ["```"]
    with open(os.path.join(pipe.run_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def bundle(pipe):
    model_cards(pipe)
    summary_md(pipe)
    rd = pipe.run_dir
    zpath = os.path.join(os.path.dirname(rd.rstrip("/")), "adar_outputs.zip")
    keep_dirs = ("results", "publish", "typed_decisions", "router")
    with zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("SUMMARY.md", "state.json", "config.json", "run.log", "errors.log"):
            if os.path.exists(os.path.join(rd, f)):
                z.write(os.path.join(rd, f), f)
        for d in keep_dirs:
            for root, _, files in os.walk(os.path.join(rd, d)):
                for f in files:
                    full = os.path.join(root, f)
                    z.write(full, os.path.relpath(full, rd))
        for f in glob.glob(os.path.join(rd, "head", "*", "*.npz")) + glob.glob(os.path.join(rd, "head", "*", "*.json")):
            z.write(f, os.path.relpath(f, rd))
    os.replace(zpath + ".tmp", zpath)
    small = proof_bundle(pipe)
    hf = pipe.cfg["publish"].get("hf_repo_prefix")
    pushed = []
    if hf:
        pushed = push_to_hub(pipe, hf)
    LOG.info("bundle written: %s (%.1f MB); proof bundle: %s (%.1f MB)", zpath, os.path.getsize(zpath) / 1e6,
             small, os.path.getsize(small) / 1e6)
    return {"zip": zpath, "proof_zip": small, "pushed": pushed}


def proof_bundle(pipe):
    """Small zip with every number the project reports (no weights, no caches): what the
    Kaggle CLI downloads and tools/collect_results.py turns into results/kaggle_run/summary.json."""
    rd = pipe.run_dir
    zpath = os.path.join(os.path.dirname(rd.rstrip("/")), "adar_results_small.zip")
    with zipfile.ZipFile(zpath + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("SUMMARY.md", "state.json", "config.json", "run.log", "errors.log"):
            if os.path.exists(os.path.join(rd, f)):
                z.write(os.path.join(rd, f), f)
        pats = ["results/*/*", "router/*/router.json", "typed_decisions/*/*.json", "typed_decisions/*/*.jsonl",
                "experts/*/*.json", "sklearn/*/*.json", "emb/*/info.json", "head/*/*.json", "data/*.json",
                "publish/*/README.md", "publish/*/config.json"]
        for pat in pats:
            for f in sorted(glob.glob(os.path.join(rd, pat))):
                if os.path.isfile(f) and not f.endswith("vocab.json"):
                    z.write(f, os.path.relpath(f, rd))
    os.replace(zpath + ".tmp", zpath)
    return zpath


def push_to_hub(pipe, prefix):
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        LOG.warning("no HF_TOKEN (env or Kaggle secret); skipping upload")
        return []
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    out = []
    for name in pipe.cfg["tasks"]:
        pub = os.path.join(pipe.run_dir, "publish", f"adar-head-{name}")
        if not os.path.isdir(pub):
            continue
        repo = f"{prefix}-{name.replace('_', '-')}"
        api.create_repo(repo, private=pipe.cfg["publish"].get("private", True), exist_ok=True)
        api.upload_folder(folder_path=pub, repo_id=repo)
        out.append(repo)
        LOG.info("uploaded %s", repo)
    return out
