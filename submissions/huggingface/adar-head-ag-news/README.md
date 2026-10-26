---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: pytorch
pipeline_tag: text-classification
tags: [system-one, laya, adar, calibration, routing, text-classification]
datasets: [fancyzhx/ag_news]
metrics: [accuracy, f1]
model-index:
- name: adar-head-ag-news
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: AG News
      type: fancyzhx/ag_news
      split: test
    metrics:
    - type: accuracy
      value: 0.9324
      name: Accuracy (mean of 3 seeds)
    - type: f1
      value: 0.9324
      name: Macro F1 (mean of 3 seeds)
---

# ADAR head for AG News

**ADAR (Adaptive Deep Architecture Routing)** is a 2.3M-parameter System-One decision head on top of the
frozen [Laya](https://huggingface.co/convaiinnovations/laya) encoder and four syllabus networks
(TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU). For every input it returns a calibrated answer (`choice`), the
probability that each expert is right (`noul`, used as per-input routing weights) and the probability
that its own answer is right (`act`, used to abstain or escalate).

## Results (AG News test set, 7,600 rows, full Kaggle run)

| Model | Accuracy | Log-loss ↓ | CPU ms / input |
|---|---:|---:|---:|
| **ADAR (ours, 3 seeds)** | 93.24% | 0.198 | 483.3 |
| **ADAR cascade (escalates 36%)** | 93.14% | 0.209 | 185.7 |
| Laya zero-shot (typed choice) | 92.53% | 0.237 | 721.2 |
| Logistic Regression (TF-IDF) | 92.51% | 0.241 | 2.4 |
| Linear SVM (TF-IDF) | 92.51% | 0.237 | 2.6 |
| Average of syllabus experts | 92.46% | 0.235 | 16.0 |
| Bi-GRU | 92.41% | 0.236 | 5.7 |
| Bi-LSTM | 92.30% | 0.232 | 6.1 |
| Stacking LogReg (Laya emb + experts) | 92.12% | 0.232 | 480.6 |
| TextCNN | 91.74% | 0.259 | 1.2 |
| Bi-RNN | 91.50% | 0.263 | 3.1 |
| sklearn MLPClassifier (TF-IDF) | 91.24% | 0.275 | 26.7 |
| Naive Bayes (TF-IDF) | 90.76% | 0.301 | 2.6 |
| Laya encoder + Logistic Regression | 90.53% | 0.278 | 464.5 |
| Random Forest (TF-IDF) | 86.89% | 0.384 | 57.0 |

ADAR: 93.24% vs. best baseline Laya zero-shot (typed choice) 92.53%.

Quality checks (paired bootstrap, 95% CI): 14 of 14 passed.
Code, notebook and full report: https://github.com/Harsh4-Dev/System-One-DLE

## Usage

```python
from huggingface_hub import snapshot_download
from adar.inference import ADARClassifier   # pip install laya; clone the GitHub repo for `adar`

clf = ADARClassifier.from_pretrained(snapshot_download("DEV-HV/adar-head-ag-news"))
clf.predict(["NASA's new telescope captured images of a distant galaxy cluster."])
```

## Limitations

* Trained for this label set; Laya stays frozen and only the head is trained.
* Numbers come from one train/validation/test protocol; see the repository for confidence intervals.
