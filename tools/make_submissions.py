"""Build the leaderboard / Hugging Face submission files from a finished run.

    python tools/make_submissions.py                                  # from results/kaggle_run/summary.json

Writes:
  submissions/typed_decisions/DISCUSSION.md   the post for the typed-decisions leaderboard (a Discussion)
  submissions/typed_decisions/entry.json      the same numbers as JSON
  submissions/huggingface/<repo>/README.md    model cards with a `model-index` results block,
                                              so the Hub shows the evaluation results on the model page
"""
import argparse
import re
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_USER = "DEV-HV"
DATASETS = {"ag_news": ("AG News", "fancyzhx/ag_news"), "emotion": ("Emotion", "dair-ai/emotion")}
TD_MODEL = "convaiinnovations/laya"
TD_PUBLISHED = {"model": "convaiinnovations/laya-typed-decisions", "accuracy": 0.766}  # the board lists accuracy only


def load(path):
    with open(path) as f:
        return json.load(f)


def adar_row(task):
    return next(r for r in task["rows"] if r["model"].startswith("ADAR (ours"))


def best_baseline(task):
    rows = [r for r in task["rows"] if not r["model"].startswith(("ADAR", "Average"))]
    return max(rows, key=lambda r: r["accuracy"])


def _cal(entry):
    sel = entry.get("adar_cal_selected")
    if sel:
        return sel, entry["adar_cal_" + sel]
    k = next(k for k in entry if k in ("adar_cal_soft", "adar_cal_hard"))
    return k.split("_")[-1], entry[k]


def typed_decisions_post(summary):
    """Text for a Discussion on huggingface.co/datasets/LocalLLaMA/typed-decisions.

    The board takes new rows as Discussions ("score it on test with the full distributions and open a
    discussion with the numbers and the mode"). Our contribution: the full metrics for the
    laya-typed-decisions row (the card lists only its accuracy) and the calibration trade-off we measured.
    """
    td = summary["typed_decisions"]
    ft, base = td["convaiinnovations/laya-typed-decisions"], td[TD_MODEL]
    s = ft["shipped"]
    fsel, fcal = _cal(ft)
    bsel, bcal = _cal(base)
    lat = ft.get("latency_ms_p50")
    entry = {
        "row": "Laya typed-decisions (convaiinnovations/laya-typed-decisions)", "mode": "fitted on train (as published)",
        "accuracy": round(s["accuracy"], 4), "kl": round(s["kl"], 4), "brier": round(s["brier_soft"], 4),
        "ece": round(s["ece"], 4), "latency_ms_p50": None if lat is None else round(lat, 1),
        "latency_note": "batch 1, 5 questions per case, Kaggle GPU T4" if lat is not None else "see metrics.json",
        "matches_published_accuracy": abs(s["accuracy"] - TD_PUBLISHED["accuracy"]) < 0.005,
        "calibration_study": {
            "laya-typed-decisions": {"shipped": {k: round(s[k], 4) for k in ("ece", "brier_soft", "kl")},
                                     f"ADAR-Cal ({fsel})": {k: round(fcal[k], 4) for k in ("ece", "brier_soft", "kl")}},
            "laya (base, zero-shot)": {"shipped": {k: round(base["shipped"][k], 4) for k in ("accuracy", "ece", "brier_soft", "kl")},
                                      f"ADAR-Cal ({bsel})": {k: round(bcal[k], 4) for k in ("accuracy", "ece", "brier_soft", "kl")}},
        },
        "link": "https://github.com/Harsh4-Dev/System-One-DLE",
    }
    out = os.path.join(ROOT, "submissions", "typed_decisions")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "entry.json"), "w") as f:
        json.dump(entry, f, indent=2)
    match_line = ("Our accuracy matches the published 0.766, so the scorer agrees with the card" if entry["matches_published_accuracy"]
                  else f"Our accuracy ({entry['accuracy']}) differs from the published 0.766; please treat these numbers with care")
    lat_s = f"{entry['latency_ms_p50']} ms ({entry['latency_note']})" if lat is not None else "–"
    post = f"""Title: Full metrics for Laya typed-decisions + a calibration note

Hi! The leaderboard lists **Laya typed-decisions** (fitted on `train`) with accuracy only. We scored it on `test` with the full gold distributions (400 cases, 2,000 decisions) and can fill in the other columns:

| Model | Mode | Accuracy | KL from gold | Brier | ECE | p50 latency |
|---|---|---|---|---|---|---|
| [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | fitted on `train` | {entry['accuracy']} | {entry['kl']} | {entry['brier']} | {entry['ece']} | {lat_s} |

{match_line} (accuracy = argmax vs gold argmax; Brier = sum over the label set against the gold distribution; ECE = top-label confidence, 10 equal-width bins).

**Calibration note.** We also fitted one temperature per (question type, option count) on `train` (selected on `train` only):

| Model | Variant | ECE | Brier | KL |
|---|---|---|---|---|
| laya-typed-decisions | as shipped | {s['ece']:.4f} | {s['brier_soft']:.4f} | {s['kl']:.4f} |
| laya-typed-decisions | + temperatures ({fsel}) | {fcal['ece']:.4f} | {fcal['brier_soft']:.4f} | {fcal['kl']:.4f} |
| laya (base, zero-shot) | as shipped | {base['shipped']['ece']:.4f} | {base['shipped']['brier_soft']:.4f} | {base['shipped']['kl']:.4f} |
| laya (base) | + temperatures ({bsel}) | {bcal['ece']:.4f} | {bcal['brier_soft']:.4f} | {bcal['kl']:.4f} |

Because the gold labels are soft, lowering top-label ECE and lowering Brier/KL can pull in different directions; we found it useful to report both, as the card recommends. Temperatures never change the argmax, so accuracy is unchanged.

Code and the full evaluation (part of a student project, ADAR): https://github.com/Harsh4-Dev/System-One-DLE
"""
    with open(os.path.join(out, "DISCUSSION.md"), "w") as f:
        f.write(post)
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
      name: Accuracy (mean of {seeds} seeds)
    - type: f1
      value: {f1}
      name: Macro F1 (mean of {seeds} seeds)
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
            name = re.sub(r" \(ours, mean of (\d+) seeds\)", r" (ours, \1 seeds)", r["model"])
            bold = r["model"].startswith("ADAR")
            row = f"| {name} | {r['accuracy'] * 100:.2f}% | {r['nll']:.3f} | {r['cpu_ms']:.1f} |"
            lines.append(row.replace(f"| {name} |", f"| **{name}** |") if bold else row)
        bb = best_baseline(t)
        headline = (f"ADAR: {a['accuracy'] * 100:.2f}% vs. best baseline {bb['model']} "
                    f"{bb['accuracy'] * 100:.2f}%.")
        repo = f"adar-head-{task.replace('_', '-')}"
        m = re.search(r"mean of (\d+) seeds", a["model"])
        card = CARD.format(seeds=m.group(1) if m else 1, hf_dataset=hf_ds, repo=repo, ds_name=ds_name, acc=a["accuracy"], f1=a["macro_f1"],
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
    a = ap.parse_args()
    s = load(a.summary)
    e = typed_decisions_post(s)
    print("typed-decisions post:", json.dumps({k: e[k] for k in ("accuracy", "kl", "brier", "ece", "latency_ms_p50")}),
          "| matches published accuracy:", e["matches_published_accuracy"])
    for p in hf_cards(s):
        print("model card:", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
