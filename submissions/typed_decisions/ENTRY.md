# typed-decisions entry: Laya + ADAR-Cal

| Field | Value |
|---|---|
| system_name | Laya + ADAR-Cal |
| base_model | convaiinnovations/laya |
| parameters | 421M (Laya) + one temperature per question type |
| accuracy | 0.363 |
| brier | 0.2365 |
| ece | 0.0214 |
| kl | 0.4325 |
| latency_ms | (see latency_note) |
| price | free: open weights (Apache-2.0), self-hosted |
| link | https://github.com/Harsh4-Dev/System-One-DLE |

Training-data disclosure: Temperatures per (question type, option count) fitted on the typed-decisions train split only; the variant was selected on train. Test labels were never used. Laya's answers are unchanged, so accuracy equals the base model's.

| Metric (test, 2,000 decisions) | Laya as shipped | Laya + ADAR-Cal | Change |
|---|---|---|---|
| ECE ↓ | 0.1731 | 0.0214 | -88% |
| Brier ↓ | 0.3154 | 0.2365 | -25% |
| KL ↓ | 0.5687 | 0.4325 | -24% |
| Accuracy ↑ | 0.363 | 0.363 | unchanged |

Scorer check against convaiinnovations/laya-typed-decisions: published {'accuracy': 0.766, 'brier': 0.062, 'ece': 0.213}, ours {'accuracy': 0.7665, 'brier': 0.0615, 'ece': 0.2137}, match = True
