# Running ADAR on Kaggle (GPU T4 ×2)

This guide takes you from the repository to a finished run with trained weights, comparison tables
and leaderboard files. You do not need to edit any code.

## 1. One-time account setup

1. Create a Kaggle account and **verify your phone number** (Settings → Phone verification).
   Kaggle only allows GPUs and internet access for verified accounts.
2. Optional, only if you want the notebook to upload the trained heads to Hugging Face:
   create a Hugging Face **write** token (see `GITHUB_AND_HF_UPLOAD.md`), then in any Kaggle notebook
   open **Add-ons → Secrets → Add a new secret**, name it `HF_TOKEN`, paste the token.

## 2. Import the notebook

1. On Kaggle: **Create → New Notebook**.
2. **File → Import Notebook** and upload `notebooks/adar_kaggle.ipynb` from this repository.
3. Right-hand panel → **Session options**:
   * **Accelerator: GPU T4 ×2**
   * **Internet: On** (needed to download Laya, the datasets and `pip install laya`)
4. If you added the `HF_TOKEN` secret, tick it under **Add-ons → Secrets** for this notebook.

## 3. Choose a profile

In the cell under **2. Configuration**:

| Setting | Use it for | Time on T4 ×2 (approx.) |
|---|---|---|
| `PROFILE = "quick"` | first check that everything works (subsets of both datasets) | 30–45 min |
| `PROFILE = "full"` (default) | the numbers you report and submit | 1.5–3 h |

To upload heads automatically set `HF_REPO_PREFIX = "<your-hf-username>/adar-head"`.
Repos are created **private**; you make them public later.

Recommended order: run `quick` once, look at the report, then run `full`.

## 4. Run

**Save Version → Save & Run All (Commit) → Save.**

A committed version runs on Kaggle's servers even if you close the browser (limit: 12 hours per
version). Follow it under **View Active Events** or the version's **Logs** tab.

You can also run cell by cell in an interactive session; the stages are the same.

## 5. What the notebook does, stage by stage

| Cell | Stage | Output folder in `adar_run/` |
|---|---|---|
| 3 | train TextCNN / RNN / LSTM / GRU (3 folds each) and 5 scikit-learn baselines | `experts/`, `sklearn/` |
| 4 | embed every text with the frozen Laya encoder on both GPUs; Laya probe, stacking baseline, Laya zero-shot | `emb/`, `probe/`, `zeroshot/` |
| 5 | train the ADAR head (3 seeds + 3 ablations) and the cascade router | `head/`, `router/`, `publish/` |
| 6 | comparison tables, QC gates, plots | `results/<task>/` |
| 7 | typed-decisions leaderboard evaluation (+ ADAR-Cal) | `typed_decisions/` |
| 8 | model cards, `SUMMARY.md`, `adar_outputs.zip`, optional Hugging Face upload | `publish/`, `/kaggle/working/adar_outputs.zip` |
| 9 | loads the published head and classifies two example sentences | – |

## 6. Get the results

Open the finished version → **Output** tab:

* `adar_outputs.zip` — everything you need: `SUMMARY.md`, `results/`, `publish/`, `typed_decisions/`.
* `adar_run/` — the full run directory including caches (only needed to resume).

Start with `SUMMARY.md`. Each task section ends with **QC gates**: every claim like
"ADAR beats X" is marked PASS only if its 95% bootstrap confidence interval excludes zero.

## 7. If something goes wrong

The notebook is built to survive failures:

* **Checkpoints everywhere.** Experts and the head save every epoch; embeddings save every shard
  of 4,096 texts; each finished stage is recorded in `adar_run/state.json`.
* **Re-running skips finished work.** Run all cells again and only unfinished stages run.
* **A failing stage does not stop the run.** Its traceback goes to `adar_run/errors.log`;
  stages that depend on it are marked *blocked*; independent stages continue.
* **GPU out of memory** halves the batch size automatically and retries.
* **Time limit.** The run stops starting new stages 15 minutes before 11.5 h, so the last
  version always has its outputs saved.

### Resume after a crash, timeout or a Kaggle restart

`/kaggle/working` is empty in a new session, so give the notebook its previous output:

1. In the notebook editor: **Add Input → Notebook Output Files → Your Work →** select this notebook
   (the latest version with an `adar_run` folder).
2. **Save & Run All** again. The first lines of the log say `Restoring previous run from ...`;
   finished stages print `[skip]`.

### Common errors

| Message | Cause | Fix |
|---|---|---|
| `ConnectionError` / `couldn't reach huggingface.co` | Internet is off | Session options → Internet: On |
| `no GPU visible` in the first cell | accelerator not set | Session options → GPU T4 ×2 |
| `data differs from the run being resumed` | the dataset came from a different source than the run you restored | keep `data_source` the same, or delete the restored `adar_run` and start fresh |
| `stage ... skipped: dependencies not done` | an earlier stage failed | read `adar_run/errors.log`, fix, run again |
| `laya loaded on cpu instead of cuda:1` | that GPU was out of memory | harmless; the run continues more slowly |

## 8. Before you trust or submit the numbers

1. All QC gates you want to claim show **PASS** in `SUMMARY.md`.
2. The **Laya zero-shot** row on AG News is in the same range as Laya's published figure (0.947).
   If it is far lower, something is wrong with the Laya download or the prompt; do not report.
3. The typed-decisions check in `LEADERBOARD_SUBMISSION.md` (step 2) passes.
