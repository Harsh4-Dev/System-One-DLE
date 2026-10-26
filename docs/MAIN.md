# ADAR — Adaptive Deep Architecture Routing

**A small trained "decision head" that sits on top of the open System-One model Laya and the
classic deep-learning architectures from our syllabus. For every input it answers three questions:
what is the answer, which architecture should I trust here, and how sure am I.**

This document explains the project in plain language: the problem, how ADAR fixes it, and what
improved. Setup and publishing are in the other documents:
[`KAGGLE_SETUP.md`](KAGGLE_SETUP.md), [`LEADERBOARD_SUBMISSION.md`](LEADERBOARD_SUBMISSION.md),
[`GITHUB_AND_HF_UPLOAD.md`](GITHUB_AND_HF_UPLOAD.md).

---

<!-- main-problem:start -->
## 1. The problem

All numbers in this document come from one full Kaggle run with the real Laya model
([`../results/kaggle_run/summary.json`](../results/kaggle_run/summary.json)).

### 1.1 The biggest model is not always the most accurate

On Emotion tweets, Laya answers **58.1%** correctly zero-shot, while a
1.2M-parameter Bi-LSTM from our syllabus reaches **92.2%**. On AG News
the order flips: Laya (92.5%) is level with the best syllabus network (Bi-GRU, 92.4%).
Which architecture is right depends on the data, and the course asks us to "identify the appropriate
deep architecture for a given application".

### 1.2 Choosing per input is worth more than choosing per dataset

On AG News a perfect per-input choice between the cheap experts and Laya + ADAR would reach
**94.87%** while calling Laya on only 8% of inputs; the experts alone reach
92.47%.

### 1.3 Strong models are expensive

On a CPU, Laya takes **706 ms** per news article; a TextCNN takes **1.1 ms**
(635× less). Running the big model on every input wastes most of that time on easy inputs.

### 1.4 Even System-One models are not perfectly calibrated

On the public typed-decisions benchmark, Laya's fine-tuned checkpoint has an expected calibration
error (ECE) of **0.214**: its confidence is off by about 21 points on average.
<!-- main-problem:end -->

---

## 2. The fix: ADAR

```
                 ┌──────────────────── cheap path ────────────────────┐
input text ──►  TextCNN · RNN · LSTM · GRU  ──►  router: "are the experts sure?"
                                                     │ yes → answer now
                                                     │ no  (only when the experts are unsure)
                                                     ▼
                 Laya encoder (frozen) ──►  ADAR head  ──►  answer
                                                          + P(answer is right)
                                                          + P(each expert is right)
```

### 2.1 The ADAR head, in simple words

Think of a meeting chaired by one person:

* **Laya reads the document.** Its frozen encoder turns the input into one vector, the `STATE` token.
* **Each answer option is described in words** ("sports news: games, matches, teams…"). Laya turns
  each description into an `OPTION` token, so the head knows what every label *means*.
* **Each syllabus expert gives its opinion.** TextCNN, RNN, LSTM and GRU each say how likely every
  option is and how sure they are. That becomes one `EXPERT` token each, and their votes are also
  attached to the option tokens.
* **The chair discusses.** A small 2-layer transformer lets these tokens look at each other (attention).
* **The chair decides** and gives three outputs:
  * `choice`: a probability for every option;
  * `noul` per expert: "is this expert right on *this* input?", which becomes a per-input
    **routing weight** for that expert's vote (the *adaptive routing* in ADAR);
  * `act`: "is my own answer right?", used to abstain or escalate.

The head is small (its parameter count is in section 3). Laya is not retrained.

### 2.2 Teaching the head to be honest

* The head is trained with **proper scoring rules** (log score + Brier score). These are the only
  kind of score you maximise by reporting your true belief, like a weather forecaster whose "70% rain"
  should mean rain on 7 days out of 10. Laya's own RLCD training rewards the same thing.
* After training, one **temperature** is fitted on validation data to fine-tune the confidence.
* The experts' opinions used for training are **out-of-fold**: each expert is trained 3 times on
  different thirds of the data, and the head only sees an expert's opinion on data that expert never
  trained on. Otherwise the head would learn to trust over-fitted experts.

### 2.3 Spending the expensive model only where it helps

The **cascade router** looks only at the cheap experts (how sure they are, whether they agree) and
predicts whether they are right. If yes, it answers immediately; if not, it calls Laya + ADAR.
Two versions:

* **Threshold router:** escalate when P(experts right) is below a threshold chosen on validation data.
* **REINFORCE router (policy gradient, Unit IV):** a small policy learns *escalate / don't* from
  reward = 1 if the final answer is right, minus a cost for every escalation.

---

<!-- main-results:start -->
## 3. The improvement

> One full run on Kaggle (GPU T4 ×2, 63 minutes) with the real Laya encoder. All models are temperature-scaled on validation data; a claim holds only where the paired-bootstrap 95% confidence interval excludes zero. Full reports: [`../results/kaggle_run/ag_news/REPORT.md`](../results/kaggle_run/ag_news/REPORT.md), [`../results/kaggle_run/emotion/REPORT.md`](../results/kaggle_run/emotion/REPORT.md).

### 3.1 AG News (7,600 test articles)

| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ | CPU ms / input |
|---|---:|---:|---:|---:|---:|
| Random Forest (TF-IDF) | 86.89% | 0.384 | 0.198 | 0.014 | 66.9 |
| Laya encoder + Logistic Regression | 90.53% | 0.278 | 0.145 | 0.012 | 472.2 |
| Naive Bayes (TF-IDF) | 90.76% | 0.301 | 0.147 | 0.026 | 3.0 |
| sklearn MLPClassifier (TF-IDF) | 91.24% | 0.275 | 0.138 | 0.018 | 27.2 |
| Bi-RNN | 91.50% | 0.263 | 0.133 | 0.019 | 3.4 |
| TextCNN | 91.71% | 0.258 | 0.128 | 0.019 | 1.1 |
| Stacking LogReg (Laya emb + experts) | 92.04% | 0.232 | 0.120 | 0.009 | 489.2 |
| Bi-LSTM | 92.30% | 0.232 | 0.121 | 0.014 | 6.4 |
| Bi-GRU | 92.41% | 0.237 | 0.122 | 0.017 | 6.2 |
| Average of syllabus experts | 92.47% | 0.235 | 0.120 | 0.016 | 17.1 |
| Logistic Regression (TF-IDF) | 92.51% | 0.241 | 0.120 | 0.015 | 2.8 |
| Linear SVM (TF-IDF) | 92.51% | 0.237 | 0.118 | 0.008 | 2.8 |
| Laya zero-shot (typed choice) | 92.53% | 0.237 | 0.120 | 0.021 | 706.3 |
| **ADAR (ours, mean of 3 seeds)** | **93.28% ± 0.16** | **0.197** | **0.103** | **0.009** | **491.7** |
| **ADAR cascade (Laya on 36% of inputs)** | **93.32%** | **0.208** | **0.106** | **0.011** | **189.2** |

What this shows:

* ADAR significantly beats Laya zero-shot, the best syllabus network and the best scikit-learn model.
* vs Laya zero-shot: **+0.93 points** (95% CI +0.42 to +1.47, significant)
* vs the best syllabus network: **+1.05 points** (95% CI +0.63 to +1.51, significant)
* vs the best scikit-learn model: **+0.95 points** (95% CI +0.47 to +1.43, significant)
* vs stacking: **+1.42 points** (95% CI +1.00 to +1.84, significant)
* Cascade: Laya called on **36%** of inputs, 93.32% accuracy at 189.2 ms vs 93.46% at 491.7 ms always calling it.
* Quality checks passed: **14/14** ([report](../results/kaggle_run/ag_news/REPORT.md)).
* **Most honest probabilities:** log-loss 0.197 vs 0.232 for the best baseline (Stacking LogReg (Laya emb + experts)), 15% lower.
* **The cascade** reaches 93.32% at 189 ms per input, against 93.28% at 492 ms when Laya + ADAR runs on everything (2.6× cheaper).
* **Stable:** 0.16 points standard deviation over the training seeds.

The REINFORCE router (Unit IV) finds a cheaper operating point: with λ = 0.05 it escalates 16% of articles and reaches 93.25% (threshold router: 36%, 93.32%).

### 3.2 Emotion (2,000 test tweets)

| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ | CPU ms / input |
|---|---:|---:|---:|---:|---:|
| Laya zero-shot (typed choice) | 58.05% | 1.211 | 0.590 | 0.088 | 653.2 |
| Laya encoder + Logistic Regression | 59.15% | 1.085 | 0.544 | 0.032 | 375.4 |
| Naive Bayes (TF-IDF) | 76.05% | 0.677 | 0.337 | 0.025 | 1.3 |
| sklearn MLPClassifier (TF-IDF) | 84.60% | 0.425 | 0.222 | 0.027 | 5.5 |
| Random Forest (TF-IDF) | 86.50% | 0.360 | 0.193 | 0.014 | 57.0 |
| Logistic Regression (TF-IDF) | 86.75% | 0.356 | 0.195 | 0.031 | 1.8 |
| Stacking LogReg (Laya emb + experts) | 87.95% | 0.343 | 0.179 | 0.018 | 384.6 |
| Linear SVM (TF-IDF) | 88.15% | 0.288 | 0.161 | 0.023 | 3.6 |
| TextCNN | 90.55% | 0.215 | 0.123 | 0.015 | 1.1 |
| Bi-RNN | 90.80% | 0.212 | 0.119 | 0.019 | 1.8 |
| **ADAR (ours, mean of 3 seeds)** | **92.00% ± 0.19** | **0.170** | **0.101** | **0.015** | **387.6** |
| Bi-GRU | 92.20% | 0.174 | 0.103 | 0.018 | 3.0 |
| Average of syllabus experts | 92.20% | 0.178 | 0.105 | 0.016 | 9.2 |
| **ADAR cascade (Laya on 0.1% of inputs)** | **92.20%** | **0.178** | **0.105** | **0.016** | **9.6** |
| Bi-LSTM | 92.25% | 0.192 | 0.110 | 0.016 | 3.3 |

What this shows:

* ADAR significantly beats Laya zero-shot and the best scikit-learn model; ties the best syllabus network (difference not significant).
* vs Laya zero-shot: **+34.20 points** (95% CI +31.80 to +36.65, significant)
* vs the best syllabus network: **+0.00 points** (95% CI -0.75 to +0.75, not significant)
* vs the best scikit-learn model: **+4.10 points** (95% CI +2.90 to +5.40, significant)
* vs stacking: **+4.30 points** (95% CI +2.90 to +5.65, significant)
* Cascade: Laya called on **0.1%** of inputs, 92.20% accuracy at 9.6 ms vs 92.25% at 387.6 ms always calling it.
* Quality checks passed: **10/14** ([report](../results/kaggle_run/emotion/REPORT.md)).
* Laya is weak on tweets, and the router learned it: it calls Laya on 0.1% of inputs.

### 3.3 Which parts matter (ablations, seed 0)

| Variant | AG News Δ accuracy (95% CI) | Emotion Δ accuracy (95% CI) |
|---|---|---|
| without the experts (Laya only) | -1.64 (-2.12 to -1.18) **significant** | -27.30 (-29.60 to -25.10) **significant** |
| without Laya (experts only) | -0.83 (-1.28 to -0.41) **significant** | -0.25 (-1.20 to +0.65) |
| without attention | -0.26 (-0.57 to -0.00) | -0.20 (-1.20 to +0.80) |

### 3.4 What the head learned about the architectures (AG News)

| Expert | Average routing weight α | Real accuracy | AUC of “is this expert right?” |
|---|---:|---:|---:|
| TextCNN | 0.161 | 91.7% | 0.910 |
| Bi-GRU | 0.160 | 92.4% | 0.899 |
| Bi-LSTM | 0.150 | 92.3% | 0.900 |
| Bi-RNN | 0.067 | 91.5% | 0.913 |

Without being told, the head gives Bi-RNN the lowest weight (the plain RNN, the architecture the syllabus flags for vanishing gradients).

## 4. Limits, stated plainly

* Laya is **frozen**; ADAR trains only the head.
* Only text classification is covered so far.
* On Emotion, ADAR **ties** the best syllabus network (+0.00 points, CI -0.75 to +0.75).
* Attention's accuracy gain is **not significant**; it mainly helps the probabilities.
* The cascade threshold, chosen on validation, gives 93.32% on test against 93.46% when always escalating.
* For typed-decisions, ADAR-Cal **re-calibrates** Laya; it does not change Laya's answers or accuracy.
<!-- main-results:end -->

## 5. How it maps to the syllabus

| Unit | In ADAR |
|---|---|
| I | TextCNN, backpropagation, initialisation, BatchNorm, gradient clipping against exploding gradients, AdamW with a cosine schedule |
| II | CNNs for classification; training a new head on a frozen pretrained network (fine-tuning) |
| III | RNN, LSTM, GRU; embeddings; attention and a transformer head; a BERT-family encoder (Laya) |
| IV | REINFORCE policy-gradient router (reward = correct − λ·cost) |

## 6. Next steps

1. Publish both heads on Hugging Face with the generated model cards (`GITHUB_AND_HF_UPLOAD.md`, `../submissions/huggingface/`).
2. Post the typed-decisions Discussion (`../submissions/typed_decisions/DISCUSSION.md`; why and how: `../submissions/README.md`).
3. Next phase: RouterArena, with LLMs as the experts (`LEADERBOARD_SUBMISSION.md`, part D).
