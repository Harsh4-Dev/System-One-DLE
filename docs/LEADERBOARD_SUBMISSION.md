# Preparing results for leaderboards and benchmarks

Everything here comes from one **full** Kaggle run (`docs/KAGGLE_SETUP.md`). Unzip
`adar_outputs.zip`; paths below are relative to it.

## What you can submit, and where

| Target | What ADAR contributes | Files | Status |
|---|---|---|---|
| Course / internal assessment | full comparison vs. scikit-learn and the syllabus models | `results/<task>/` | ready after the run |
| Hugging Face model hub | the trained ADAR heads | `publish/adar-head-<task>/` | ready after the run |
| **typed-decisions** (System One leaderboard) | Laya + ADAR-Cal calibration | `typed_decisions/<checkpoint>/` | ready after the run, see checks below |
| RouterArena (LLM routers) | ADAR as a router between LLMs | – | next phase, see the end |

---

## A. Course / internal comparison

| File | Use it for |
|---|---|
| `results/<task>/REPORT.md` | the full table, QC gates, routing analysis, cascade summary |
| `results/<task>/comparison.csv` | the same table for Excel / slides |
| `results/<task>/reliability.png` | "is the confidence honest?" figure |
| `results/<task>/accuracy_vs_latency.png` | "accuracy for the cost" figure with the cascade curve |
| `results/<task>/risk_coverage.png` | "accuracy when the model may abstain" figure |
| `SUMMARY.md` | all tasks on one page |

Quote only claims whose QC gate says **PASS**.

## B. typed-decisions leaderboard

* Benchmark: `LocalLLaMA/typed-decisions` — 400 test cases, 2,000 typed decisions (choice / score / yes-no).
* Leaderboard: Hugging Face Space `mayafree/typed-decision-leaderboard`.
* Published reference rows: Jev 1.13.0 — 0.727 accuracy; `laya-typed-decisions` — 0.766 accuracy,
  0.062 Brier, 0.213 ECE.

The notebook evaluates two Laya checkpoints, each **as shipped** and with **ADAR-Cal**:
one temperature per (question type, number of options), fitted on the **train** split only.
The ADAR-Cal variant (soft-label or hard-label objective) is also chosen on train. The test split
is never used for fitting or choosing.

### Checklist before submitting

- [ ] **1. Scorer matches the leaderboard.** In `typed_decisions/laya-typed-decisions/metrics.json`,
  `shipped.test.accuracy` should be ≈ 0.766 and `ece` ≈ 0.213 (±0.01).
  Brier is reported two ways: `brier_soft` (against the gold distribution) and `brier_onehot`.
  The one that reads ≈ 0.062 is the leaderboard's definition; use that one in your entry.
  **If accuracy or ECE do not match, stop:** the metric definitions differ and a submission would not be comparable.
- [ ] **2. There is a real improvement.** ADAR-Cal only changes confidence, never the chosen answer,
  so accuracy stays the same. Submit only if test ECE and Brier are lower than the shipped row.
- [ ] **3. Pick the clean entry.** `laya-typed-decisions` was trained on the train split by its
  authors, so temperatures fitted on that same split can come out too sharp. The base `laya`
  checkpoint + ADAR-Cal has no such overlap; prefer it if both improve.
- [ ] **4. Fill in the form** on the Space:

| Field | Value |
|---|---|
| Model / system name | e.g. `laya + ADAR-Cal` |
| Parameters | 421M (Laya; ADAR-Cal adds a handful of temperatures) |
| Accuracy, Brier, ECE | from `adar_cal.<selected>.test` in `metrics.json` |
| Latency | `latency_ms_p50` and the hardware: "Kaggle T4, batch 1, 5 questions per case" |
| Training data disclosure | "temperatures fitted on the typed-decisions train split" |
| Link | your Hugging Face model or GitHub repository |

- [ ] **5. Keep the evidence.** Keep `metrics.json` and `test_predictions.jsonl`; reviewers may ask
  how the numbers were produced.

## C. Hugging Face model hub

See `GITHUB_AND_HF_UPLOAD.md`. Before making a model public, open its `README.md` (the
generated model card) and check that the results table holds your Kaggle numbers.

## D. RouterArena (next phase, not done by this notebook)

RouterArena (ICLR 2026, `github.com/RouteWorks/RouterArena`) ranks routers that pick one LLM per
query. ADAR fits it directly: replace the syllabus experts by the candidate LLMs and ask one
"is this model right?" question per LLM, then pick the cheapest model that is likely right.

What is still needed:

1. **Training data that is not RouterArena's.** Its rules reject routers fitted on its data.
   RouterBench (`withmartian/routerbench`, about 405k recorded model answers) is the standard source.
2. **A model pool and cost table** in RouterArena's config format.
3. **An API budget.** Evaluation calls the chosen LLMs unless RouterArena's cache already has them.
4. **Submission:** a pull request with `router_inference/config/<name>.json` and the prediction
   files, then comment `/evaluate`. Test on the 809-query `sub_10` split first.

## E. Write-up

A short arXiv preprint makes every number above citable: the problem, the ADAR head, the
comparison tables, the ablations, the cascade curve and the limitations. Link it from the model
card and from each leaderboard entry.
