"""Build the leaderboard / Hugging Face submission files from a finished run.

    python tools/make_submissions.py                                  # from results/kaggle_run/summary.json
    python tools/make_submissions.py --metrics adar_run/typed_decisions/laya/metrics.json

Writes:
  submissions/typed_decisions/entry.json      values for the typed-decisions leaderboard form
  submissions/typed_decisions/ENTRY.md        the same, as a table to paste or screenshot
  submissions/huggingface/<repo>/README.md    model cards with a `model-index` results block,
                                              so the Hub shows the evaluation results on the model page
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_USER = "DEV-HV"
DATASETS = {"ag_news": ("AG News", "fancyzhx/ag_news"), "emotion": ("Emotion", "dair-ai/emotion")}
TD_MODEL = "convaiinnovations/laya"
TD_PUBLISHED = {"model": "convaiinnovations/laya-typed-decisions", "accuracy": 0.766, "brier": 0.062, "ece": 0.213}


def load(path):
    with open(path) as f:
        return json.load(f)


def adar_row(task):
    return next(r for r in task["rows"] if r["model"].startswith("ADAR (ours"))


def best_baseline(task):
    rows = [r for r in task["rows"] if not r["model"].startswith(("ADAR", "Average"))]
    return max(rows, key=lambda r: r["accuracy"])


def typed_decisions_entry(summary, metrics_path):
    td = summary["typed_decisions"]
    base = td[TD_MODEL]
    cal_key = next(k for k in base if k.startswith("adar_cal"))
    cal, shipped = base[cal_key], base["shipped"]
    check = td.get("convaiinnovations/laya-typed-decisions", {}).get("shipped")
    latency, latency_note = None, "copy latency_ms_p50 from typed_decisions/laya/metrics.json"
    if metrics_path:
        m = load(metrics_path)
        latency, latency_note = round(m["latency_ms_p50"], 1), "p50 per case (5 questions), Kaggle GPU T4, batch 1"
    entry = {
        "system_name": "Laya + ADAR-Cal",
        "base_model": TD_MODEL,
        "parameters": "421M (Laya) + one temperature per question type",
        "accuracy": round(cal["accuracy"], 4),
        "brier": round(cal["brier_soft"], 4),
        "ece": round(cal["ece"], 4),
        "kl": round(cal["kl"], 4),
        "latency_ms": latency,
        "latency_note": latency_note,
        "price": "free: open weights (Apache-2.0), self-hosted",
        "training_data_disclosure": "Temperatures per (question type, option count) fitted on the typed-decisions "
                                    "train split only; the variant was selected on train. Test labels were never "
                                    "used. Laya's answers are unchanged, so accuracy equals the base model's.",
        "link": "https://github.com/Harsh4-Dev/System-One-DLE",
        "baseline_same_model": {k: round(shipped[k], 4) for k in ("accuracy", "brier_soft", "ece", "kl")},
        "improvement": {
            "ece": f"{(cal['ece'] / shipped['ece'] - 1) * 100:+.0f}%",
            "brier": f"{(cal['brier_soft'] / shipped['brier_soft'] - 1) * 100:+.0f}%",
            "kl": f"{(cal['kl'] / shipped['kl'] - 1) * 100:+.0f}%",
        },
        "scorer_check": None if check is None else {
            "against": TD_PUBLISHED["model"],
            "published": {k: TD_PUBLISHED[k] for k in ("accuracy", "brier", "ece")},
            "ours": {"accuracy": round(check["accuracy"], 4), "brier": round(check["brier_soft"], 4),
                     "ece": round(check["ece"], 4)},
            "matches": abs(check["accuracy"] - TD_PUBLISHED["accuracy"]) < 0.01
            and abs(check["brier_soft"] - TD_PUBLISHED["brier"]) < 0.01
            and abs(check["ece"] - TD_PUBLISHED["ece"]) < 0.01,
        },
    }
    out = os.path.join(ROOT, "submissions", "typed_decisions")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "entry.json"), "w") as f:
        json.dump(entry, f, indent=2)
    b = entry["baseline_same_model"]
    md = ["# typed-decisions entry: Laya + ADAR-Cal", "",
          "| Field | Value |", "|---|---|"]
    for k in ("system_name", "base_model", "parameters", "accuracy", "brier", "ece", "kl", "latency_ms", "price", "link"):
        md.append(f"| {k} | {entry[k] if entry[k] is not None else '(see latency_note)'} |")
    md += ["", f"Training-data disclosure: {entry['training_data_disclosure']}", "",
           "| Metric (test, 2,000 decisions) | Laya as shipped | Laya + ADAR-Cal | Change |", "|---|---|---|---|",
           f"| ECE ↓ | {b['ece']} | {entry['ece']} | {entry['improvement']['ece']} |",
           f"| Brier ↓ | {b['brier_soft']} | {entry['brier']} | {entry['improvement']['brier']} |",
           f"| KL ↓ | {b['kl']} | {entry['kl']} | {entry['improvement']['kl']} |",
           f"| Accuracy ↑ | {b['accuracy']} | {entry['accuracy']} | unchanged |"]
    if entry["scorer_check"]:
        c = entry["scorer_check"]
        md += ["", f"Scorer check against {c['against']}: published {c['published']}, ours {c['ours']}, "
                   f"match = {c['matches']}"]
    with open(os.path.join(out, "ENTRY.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return entry


CARD = """---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: pytorch
pipeline_tag: text-classification
tags: [system-one, laya, adar, calibration, routing, text-classification]
datasets: [{hf_dataset}]
metrics: [accuracy, f1]
model-index:
- name: {repo}
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: {ds_name}
      type: {hf_dataset}
      split: test
    metrics:
    - type: accuracy
      value: {acc}
      name: Accuracy (mean of 3 seeds)
    - type: f1
      value: {f1}
      name: Macro F1 (mean of 3 seeds)
---

# ADAR head for {ds_name}

**ADAR (Adaptive Deep Architecture Routing)** is a 2.3M-parameter System-One decision head on top of the
frozen [Laya](https://huggingface.co/convaiinnovations/laya) encoder and four syllabus networks
(TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU). For every input it returns a calibrated answer (`choice`), the
probability that each expert is right (`noul`, used as per-input routing weights) and the probability
that its own answer is right (`act`, used to abstain or escalate).

## Results ({ds_name} test set, {n_test:,} rows, full Kaggle run)

| Model | Accuracy | Log-loss ↓ | CPU ms / input |
|---|---:|---:|---:|
{table}

{headline}

Quality checks (paired bootstrap, 95% CI): {qc_pass} of {qc_total} passed.
Code, notebook and full report: https://github.com/Harsh4-Dev/System-One-DLE

## Usage

```python
from huggingface_hub import snapshot_download
from adar.inference import ADARClassifier   # pip install laya; clone the GitHub repo for `adar`

clf = ADARClassifier.from_pretrained(snapshot_download("{repo_id}"))
clf.predict(["NASA's new telescope captured images of a distant galaxy cluster."])
```

## Limitations

* Trained for this label set; Laya stays frozen and only the head is trained.
* Numbers come from one train/validation/test protocol; see the repository for confidence intervals.
"""


def hf_cards(summary):
    written = []
    for task, (ds_name, hf_ds) in DATASETS.items():
        t = summary["tasks"].get(task)
        if not t:
            continue
        a = adar_row(t)
        keep = [r for r in t["rows"] if not r["model"].startswith("ADAR ablation")]
        keep.sort(key=lambda r: -r["accuracy"])
        lines = []
        for r in keep:
            name = r["model"].replace(" (ours, mean of 3 seeds)", " (ours, 3 seeds)")
            bold = r["model"].startswith("ADAR")
            row = f"| {name} | {r['accuracy'] * 100:.2f}% | {r['nll']:.3f} | {r['cpu_ms']:.1f} |"
            lines.append(row.replace(f"| {name} |", f"| **{name}** |") if bold else row)
        bb = best_baseline(t)
        headline = (f"ADAR: {a['accuracy'] * 100:.2f}% vs. best baseline {bb['model']} "
                    f"{bb['accuracy'] * 100:.2f}%.")
        repo = f"adar-head-{task.replace('_', '-')}"
        card = CARD.format(hf_dataset=hf_ds, repo=repo, ds_name=ds_name, acc=a["accuracy"], f1=a["macro_f1"],
                           n_test=summary["splits"][task]["test"], table="\n".join(lines), headline=headline,
                           qc_pass=t["qc_pass"], qc_total=t["qc_total"], repo_id=f"{HF_USER}/{repo}")
        out = os.path.join(ROOT, "submissions", "huggingface", repo)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write(card)
        written.append(out)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(ROOT, "results", "kaggle_run", "summary.json"))
    ap.add_argument("--metrics", default=None, help="typed_decisions/laya/metrics.json from the run (for latency)")
    a = ap.parse_args()
    s = load(a.summary)
    e = typed_decisions_entry(s, a.metrics)
    print("typed-decisions entry:", json.dumps({k: e[k] for k in ("accuracy", "brier", "ece", "kl")}),
          "| scorer check:", e["scorer_check"] and e["scorer_check"]["matches"])
    for p in hf_cards(s):
        print("model card:", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
