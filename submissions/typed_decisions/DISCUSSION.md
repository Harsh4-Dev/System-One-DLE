Reply to: https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/2 ("Laya benchmark results")

@codelion asked for KL and Brier from the full distributions for this row. We re-ran **Laya typed-decisions** independently on `test` (400 cases, 2,000 decisions, each whole case in one request) and scored it against the full gold distributions:

| Model | Mode | Accuracy | KL from gold | Brier | ECE | p50 latency |
|---|---|---|---|---|---|---|
| [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | fitted on `train` (as published) | 0.7665 | 0.117 | 0.0615 | 0.2137 | 148.3 ms (batch 1, 5 questions per case, Kaggle GPU T4) |

Our accuracy matches the published 0.766, so the scorer agrees with the card (accuracy = argmax vs gold argmax; KL from the gold distribution; Brier = sum over the label set against the gold distribution; ECE = top-label confidence, 10 equal-width bins). So the self-reported 0.766 now has an independent re-run.

**Calibration note.** We also fitted one temperature per (question type, option count) on `train` (selected on `train` only):

| Model | Variant | Accuracy | ECE | Brier | KL |
|---|---|---|---|---|---|
| laya-typed-decisions | as shipped | 0.7665 | 0.2137 | 0.0615 | 0.1170 |
| laya-typed-decisions | + temperatures (hard) | 0.7665 | 0.0626 | 0.1476 | 0.5264 |
| laya (base, zero-shot) | as shipped | 0.3630 | 0.1731 | 0.3154 | 0.5687 |
| laya (base) | + temperatures (soft) | 0.3630 | 0.0214 | 0.2365 | 0.4325 |

Because the gold labels are soft, lowering top-label ECE and lowering Brier/KL can pull in different directions, so we report both. Temperatures never change the argmax, so accuracy is unchanged. (We are not submitting the calibrated base model as a row; its accuracy is below the prior.)

We also opened a community PR on the model repo adding `.eval_results/typed-decisions.yaml` with these four values, so the row can show on the Hub leaderboard: <link to the PR>

Code, notebook and the full evaluation (student project, ADAR): https://github.com/Harsh4-Dev/System-One-DLE
