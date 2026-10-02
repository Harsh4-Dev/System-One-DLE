# Submitting ADAR results

Everything below comes out of one Kaggle run of `notebooks/adar_kaggle.ipynb`
(download `/kaggle/working/adar_outputs.zip`).

## 1. Course / internal comparison

| File | Use |
|---|---|
| `results/<task>/REPORT.md` | full comparison table (ADAR vs. scikit-learn vs. TextCNN/RNN/LSTM/GRU vs. Laya baselines), QC gates with bootstrap CIs, routing analysis |
| `results/<task>/comparison.csv` | same table as CSV, for your slides/report |
| `results/<task>/reliability.png`, `accuracy_vs_latency.png`, `risk_coverage.png` | the three figures |
| `SUMMARY.md` | all tasks + typed-decisions on one page |

## 2. Publish the trained head (Hugging Face)

`publish/adar-head-<task>/` is a complete model folder: `model.safetensors`, `config.json`,
`options.safetensors`, `experts/`, a generated `README.md` model card and `REPORT.md`.

* Automatic: add a Kaggle secret `HF_TOKEN` (a *write* token) and set `HF_REPO_PREFIX = "<user>/adar-head"`
  in the config cell. Repos are created **private**; make them public on the Hub when you are ready.
* Manual: `huggingface-cli upload <user>/adar-head-ag-news publish/adar-head-ag_news .`

Base model licence: Laya is Apache-2.0, so the head can be published under Apache-2.0 with attribution
(the model card's `base_model` field does this).

## 3. typed-decisions leaderboard (System One benchmark)

Benchmark: `LocalLLaMA/typed-decisions` (400 test cases, 2,000 typed decisions), leaderboard Space
`huggingface.co/spaces/mayafree/typed-decision-leaderboard`.

The notebook writes `typed_decisions/<checkpoint>/metrics.json` and `test_predictions.jsonl` for
`laya-typed-decisions` and base `laya`, each **as shipped** and with **ADAR-Cal** (per question-type
temperatures fitted on the *train* split; the variant is also selected on train, never on test).

Before submitting:

1. **Check the scorer.** The shipped `laya-typed-decisions` row must reproduce the published
   0.766 accuracy / 0.062 Brier / 0.213 ECE (within ~0.01). The notebook reports Brier against both the
   soft gold distribution and one-hot gold; the one that matches 0.062 is the leaderboard's definition.
   If neither matches, do not submit — the metric definitions differ.
2. **Only submit an improvement you can defend.** ADAR-Cal changes calibration, not accuracy. Submit it
   only if test ECE/Brier improve over the shipped row, and say in the entry: "temperatures fitted on
   the train split".
3. **Disclose training data.** `laya-typed-decisions` was fine-tuned on the train split by its authors,
   so its train-split temperatures can be too sharp; base `laya` + ADAR-Cal is the cleaner entry.
4. Fill the Space's form with: model name (e.g. `laya + ADAR-Cal`), parameter count (421M),
   accuracy, Brier, ECE, p50 latency (state the GPU: Kaggle T4) and a link to your model/repo.

## 4. RouterArena (next phase — not in this notebook)

RouterArena (ICLR 2026, `github.com/RouteWorks/RouterArena`) scores LLM routers on 8,400 queries.
ADAR maps onto it directly: replace the syllabus experts with the candidate LLMs and ask one
`noul` per LLM — "P(this model answers correctly)" — then pick the cheapest model above a threshold.

What it needs, which this repo does not do yet:

* **Training data that is not RouterArena's** (its rules reject routers fit on its data):
  RouterBench (`withmartian/routerbench`, ~405k precomputed model outcomes) is the usual choice.
* **A model pool + cost table** in RouterArena's config format.
* **API budget**: evaluation calls the chosen LLMs unless RouterArena's cache covers them.
* Submission = a PR with `router_inference/config/<name>.json` and the prediction files, then `/evaluate`.
  Test locally on the 809-query `sub_10` split first.

## 5. Write-up

A short arXiv preprint (method, the comparison tables, ablations, cascade curve, limitations) linked
from the model card and the leaderboard entries makes every number above citable.
