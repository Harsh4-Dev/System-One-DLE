Title: Full metrics for Laya typed-decisions + a calibration note

Hi! The leaderboard lists **Laya typed-decisions** (fitted on `train`) with accuracy only. We scored it on `test` with the full gold distributions (400 cases, 2,000 decisions) and can fill in the other columns:

| Model | Mode | Accuracy | KL from gold | Brier | ECE | p50 latency |
|---|---|---|---|---|---|---|
| [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) | fitted on `train` | 0.7665 | 0.117 | 0.0615 | 0.2137 | – |

Our accuracy matches the published 0.766, so the scorer agrees with the card (accuracy = argmax vs gold argmax; Brier = sum over the label set against the gold distribution; ECE = top-label confidence, 10 equal-width bins).

**Calibration note.** We also fitted one temperature per (question type, option count) on `train` (selected on `train` only):

| Model | Variant | ECE | Brier | KL |
|---|---|---|---|---|
| laya-typed-decisions | as shipped | 0.2137 | 0.0615 | 0.1170 |
| laya-typed-decisions | + temperatures (hard) | 0.0626 | 0.1476 | 0.5264 |
| laya (base, zero-shot) | as shipped | 0.1731 | 0.3154 | 0.5687 |
| laya (base) | + temperatures (soft) | 0.0214 | 0.2365 | 0.4325 |

Because the gold labels are soft, lowering top-label ECE and lowering Brier/KL can pull in different directions; we found it useful to report both, as the card recommends. Temperatures never change the argmax, so accuracy is unchanged.

Code and the full evaluation (part of a student project, ADAR): https://github.com/Harsh4-Dev/System-One-DLE
