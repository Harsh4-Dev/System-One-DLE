# Preparing results for leaderboards and benchmarks

Everything here comes from one **full** Kaggle run (`docs/KAGGLE_SETUP.md`). Unzip
`adar_outputs.zip`; paths below are relative to it.

## What you can submit, and where

| Target | What ADAR contributes | Files | Status |
|---|---|---|---|
| Course / internal assessment | full comparison vs. scikit-learn and the syllabus models | `results/<task>/` | ready after the run |
| Hugging Face model hub | the trained ADAR heads | `publish/adar-head-<task>/` | ready after the run |
| **typed-decisions** (System One leaderboard) | missing KL / Brier / ECE for Laya typed-decisions + calibration study | `submissions/typed_decisions/DISCUSSION.md` | ready after the fresh run |
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

* Benchmark and leaderboard: the dataset card of `LocalLLaMA/typed-decisions` (400 test cases,
  2,000 typed decisions). New rows are added through **Discussions** on the dataset.
* Two tables: zero-shot general models, and models fitted on `train`. The card asks for KL or log-loss
  and Brier next to accuracy.
* Our submission is a Discussion that fills in the missing KL / Brier / ECE for
  *Laya typed-decisions* and reports our calibration study. We do not add a "Laya + ADAR-Cal" row:
  its accuracy (0.363) is below the board's prior baseline (0.470).

Full procedure and the reasoning: `submissions/README.md`. The post itself:
`submissions/typed_decisions/DISCUSSION.md` (regenerate after the fresh run).

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
