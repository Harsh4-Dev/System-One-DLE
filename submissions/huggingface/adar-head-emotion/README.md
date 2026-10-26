---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: pytorch
pipeline_tag: text-classification
tags: [system-one, laya, adar, calibration, routing, text-classification]
datasets: [dair-ai/emotion]
metrics: [accuracy, f1]
model-index:
- name: adar-head-emotion
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Emotion
      type: dair-ai/emotion
      split: test
    metrics:
    - type: accuracy
      value: 0.92
      name: Accuracy (mean of 3 seeds)
    - type: f1
      value: 0.8790498611638675
      name: Macro F1 (mean of 3 seeds)
---

# ADAR head for Emotion

**ADAR (Adaptive Deep Architecture Routing)** is a 2.3M-parameter System-One decision head on top of the
frozen [Laya](https://huggingface.co/convaiinnovations/laya) encoder and four syllabus networks
(TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU). For every input it returns a calibrated answer (`choice`), the
probability that each expert is right (`noul`, used as per-input routing weights) and the probability
that its own answer is right (`act`, used to abstain or escalate).

## Results (Emotion test set, 2,000 rows, full Kaggle run)

| Model | Accuracy | Log-loss ↓ | CPU ms / input |
|---|---:|---:|---:|
| Bi-LSTM | 92.25% | 0.192 | 3.3 |
| Bi-GRU | 92.20% | 0.174 | 3.0 |
| Average of syllabus experts | 92.20% | 0.178 | 9.2 |
| **ADAR cascade (escalates 0%)** | 92.20% | 0.178 | 9.6 |
| **ADAR (ours, 3 seeds)** | 92.00% | 0.170 | 387.6 |
| Bi-RNN | 90.80% | 0.212 | 1.8 |
| TextCNN | 90.55% | 0.215 | 1.1 |
| Linear SVM (TF-IDF) | 88.15% | 0.288 | 3.6 |
| Stacking LogReg (Laya emb + experts) | 87.95% | 0.343 | 384.6 |
| Logistic Regression (TF-IDF) | 86.75% | 0.356 | 1.8 |
| Random Forest (TF-IDF) | 86.50% | 0.360 | 57.0 |
| sklearn MLPClassifier (TF-IDF) | 84.60% | 0.425 | 5.5 |
| Naive Bayes (TF-IDF) | 76.05% | 0.677 | 1.3 |
| Laya encoder + Logistic Regression | 59.15% | 1.085 | 375.4 |
| Laya zero-shot (typed choice) | 58.05% | 1.211 | 653.2 |

ADAR: 92.00% vs. best baseline Bi-LSTM 92.25%.

Quality checks (paired bootstrap, 95% CI): 10 of 14 passed.
Code, notebook and full report: https://github.com/Harsh4-Dev/System-One-DLE

## Usage

```python
from huggingface_hub import snapshot_download
from adar.inference import ADARClassifier   # pip install laya; clone the GitHub repo for `adar`

clf = ADARClassifier.from_pretrained(snapshot_download("DEV-HV/adar-head-emotion"))
clf.predict(["NASA's new telescope captured images of a distant galaxy cluster."])
```

## Limitations

* Trained for this label set; Laya stays frozen and only the head is trained.
* Numbers come from one train/validation/test protocol; see the repository for confidence intervals.
