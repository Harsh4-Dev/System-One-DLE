"""Build notebooks/adar_kaggle.ipynb from the tested adar/ package.

The notebook writes every adar/*.py file with %%writefile, so the code that runs on Kaggle is
byte-for-byte the code that was tested locally. Re-run this script after changing adar/.

    python tools/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__", "utils", "runstate", "data", "metrics", "experts", "baselines", "encoder", "head",
           "router", "laya_tasks", "report", "pipeline", "inference"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# ADAR — Adaptive Deep Architecture Routing with a System-One head (Laya)

This notebook trains and evaluates everything for the ADAR project on **Kaggle GPU T4 ×2**:

| Stage | What it produces |
|---|---|
| Syllabus experts | TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU (3-fold, out-of-fold predictions) |
| scikit-learn baselines | Naive Bayes, Logistic Regression, Linear SVM, Random Forest, MLPClassifier on TF-IDF |
| Laya embeddings | frozen `convaiinnovations/laya` encoder, cached in shards on both GPUs |
| Laya baselines | Laya zero-shot (typed choice), Laya + Logistic Regression probe, stacking baseline |
| **ADAR head** | 3 seeds + 3 ablations, temperature-scaled, publishable safetensors |
| Router | cascade (cheap experts first, Laya only when needed): threshold + REINFORCE policy |
| Report | comparison tables, QC gates with bootstrap CIs, reliability / latency / risk-coverage plots |
| typed-decisions | System One leaderboard evaluation of Laya checkpoints + ADAR-Cal |

### How to run
1. **Settings → Accelerator: GPU T4 ×2**, **Internet: On**.
2. (Optional, to upload the trained heads) **Add-ons → Secrets → `HF_TOKEN`** and set `HF_REPO_PREFIX` below.
3. **Save Version → Save & Run All (Commit)**. The run takes roughly 1.5–3 hours; outputs land in
   `/kaggle/working/adar_outputs.zip` and `/kaggle/working/adar_run/`.

### Saving progress & recovering from failures
* Every stage writes a checkpoint; expert and head training checkpoint **every epoch**, embeddings every shard.
* Re-running a cell skips finished work. A failing stage is logged to `adar_run/errors.log` and the
  run continues with independent stages (stages that depend on it are marked *blocked*).
* If a session dies or times out: open the notebook, **Add Input → Your Work → this notebook's last
  version** (its output), then run all again — the previous `adar_run/` is restored automatically and
  only unfinished stages run.
"""


def build():
    cells = [md(INTRO)]
    cells.append(code("""
import os, subprocess, sys, shutil
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi")
      else "no GPU visible - set Accelerator to GPU T4 x2 for the full run")
print("disk free (GB):", shutil.disk_usage("/kaggle/working" if os.path.exists("/kaggle/working") else ".").free / 1e9)
# laya = the open-source System One model (Apache-2.0); datasets = Hugging Face datasets
!pip install -q "laya>=0.3.23" datasets
"""))
    cells.append(md("## 1. The `adar` package\nThese cells write the project source (the same files as the GitHub repo)."))
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    cells.append(code("os.makedirs('adar', exist_ok=True)"))
    for m in MODULES:
        src = open(os.path.join(ROOT, "adar", f"{m}.py")).read()
        cells.append(code(f"%%writefile adar/{m}.py\n{src}"))
    cells.append(md("""
## 2. Configuration
`PROFILE = "full"` reproduces the reported numbers. `"quick"` is a ~30-minute check on subsets.
"""))
    cells.append(code("""
import importlib, sys
sys.path.insert(0, os.getcwd())
import adar.pipeline as P
importlib.reload(P)

PROFILE = os.environ.get("ADAR_PROFILE", "full")      # "full" | "quick" | "smoke" (local CPU test)
HF_REPO_PREFIX = None                                   # e.g. "your-hf-username/adar-head" to upload heads

CFG = {"publish": {"hf_repo_prefix": HF_REPO_PREFIX, "private": True}}
if PROFILE == "quick":
    CFG.update({"task_sizes": {
        "ag_news": {"n_train": 20000, "n_val": 3000, "n_test": None, "max_len": 128, "expert_max_len": 64},
        "emotion": {"n_train": 8000, "n_val": None, "n_test": None, "max_len": 64, "expert_max_len": 48}},
        "expert_epochs": 4, "typed_decisions": {"train_limit": 300, "test_limit": None}})
elif PROFILE == "smoke":   # runs on a laptop CPU with a local stand-in encoder; not for reporting
    CFG.update({"run_dir": os.environ.get("ADAR_RUN_DIR", "adar_run"), "tasks": ["ag_news"],
                "data_source": "csv", "local_data_dir": os.environ.get("ADAR_LOCAL_DATA"),
                "task_sizes": {"ag_news": {"n_train": 3000, "n_val": 600, "n_test": 1000, "max_len": 64,
                                           "expert_max_len": 48}},
                "encoder": {"backend": "hf", "model_id": os.environ.get("ADAR_STANDIN_ENCODER"),
                            "batch_size": 64, "shard_size": 2048},
                "expert_epochs": 2, "head_epochs": 6, "laya_zero_shot": False,
                "typed_decisions": {"enabled": False}, "time_budget_h": None})
pipe = P.Pipeline(CFG)
print(pipe.rs.summary() or "fresh run")
"""))
    cells.append(md("## 3. Syllabus experts (TextCNN / RNN / LSTM / GRU) and scikit-learn baselines"))
    cells.append(code("""
for name in pipe.cfg["tasks"]:
    pipe.rs.run(f"experts:{name}", pipe.stage_experts, name=name)
    pipe.rs.run(f"sklearn:{name}", pipe.stage_sklearn, name=name)
"""))
    cells.append(md("## 4. Frozen Laya embeddings (both GPUs) + Laya baselines"))
    cells.append(code("""
for name in pipe.cfg["tasks"]:
    pipe.rs.run(f"embed:{name}", pipe.stage_embed, name=name)
    pipe.rs.run(f"probe:{name}", pipe.stage_probe, deps=[f"embed:{name}", f"experts:{name}"], name=name)
    if pipe.cfg["laya_zero_shot"]:
        pipe.rs.run(f"zeroshot:{name}", pipe.stage_zeroshot, name=name)
"""))
    cells.append(md("## 5. Train the ADAR System-One head (3 seeds + ablations) and the router"))
    cells.append(code("""
for name in pipe.cfg["tasks"]:
    pipe.rs.run(f"head:{name}", pipe.stage_head, deps=[f"embed:{name}", f"experts:{name}"], name=name)
    pipe.rs.run(f"router:{name}", pipe.stage_router, deps=[f"head:{name}"], name=name)
pipe.free_encoders()
"""))
    cells.append(md("## 6. Comparison report + QC gates"))
    cells.append(code("""
from IPython.display import Markdown, Image, display
for name in pipe.cfg["tasks"]:
    pipe.rs.run(f"report:{name}", pipe.stage_report, deps=[f"head:{name}"], force=True, name=name)
    rep = os.path.join(pipe.run_dir, "results", name, "REPORT.md")
    if os.path.exists(rep):
        display(Markdown(open(rep).read().split("![reliability]")[0]))
        for png in ("reliability.png", "accuracy_vs_latency.png", "risk_coverage.png"):
            display(Image(os.path.join(pipe.run_dir, "results", name, png)))
"""))
    cells.append(md("""
## 7. typed-decisions leaderboard (System One benchmark)
Evaluates the shipped Laya checkpoints on `LocalLLaMA/typed-decisions` and adds **ADAR-Cal**
(per question-type temperatures fitted on the *train* split only — test is never used for fitting or selection).
"""))
    cells.append(code("""
td = pipe.cfg["typed_decisions"]
if td["enabled"]:
    for m in td["models"]:
        print(m, pipe.rs.run(f"typed_decisions:{m}", pipe.stage_typed_decisions, model_id=m))
"""))
    cells.append(md("## 8. Bundle: model cards, summary, zip (and optional Hugging Face upload)"))
    cells.append(code("""
print(pipe.rs.run("bundle", pipe.stage_bundle, force=True))
print(pipe.rs.summary())
display(Markdown(open(os.path.join(pipe.run_dir, "SUMMARY.md")).read()[:20000]))
"""))
    cells.append(md("""
## 9. Try the published head
Loads `adar_run/publish/adar-head-ag_news` exactly as someone downloading it would.
"""))
    cells.append(code("""
from adar.inference import ADARClassifier
pub = os.path.join(pipe.run_dir, "publish", f"adar-head-{pipe.cfg['tasks'][0]}")
if os.path.isdir(pub):
    clf = ADARClassifier.from_pretrained(pub)
    for r in clf.predict(["NASA's new telescope captured images of a distant galaxy cluster.",
                          "The striker scored twice as the home side won the derby."]):
        print(r)
"""))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"},
                                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                  "isGpuEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, "notebooks", "adar_kaggle.ipynb")
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
