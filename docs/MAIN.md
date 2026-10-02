# ADAR — Adaptive Deep Architecture Routing

**A small trained "decision head" that sits on top of the open System-One model Laya and the
classic deep-learning architectures from our syllabus. For every input it answers three questions:
what is the answer, which architecture should I trust here, and how sure am I.**

This document explains the project in plain language: the problem, how ADAR fixes it, and what
improved. Setup and publishing are in the other documents:
[`KAGGLE_SETUP.md`](KAGGLE_SETUP.md), [`LEADERBOARD_SUBMISSION.md`](LEADERBOARD_SUBMISSION.md),
[`GITHUB_AND_HF_UPLOAD.md`](GITHUB_AND_HF_UPLOAD.md).

---

## 1. The problem

### 1.1 Models say "I'm sure" when they are not

A classifier gives an answer and a confidence. Out of the box that confidence is often not honest:
on our AG News test, **21–23% of the wrong answers from TextCNN, LSTM and GRU came with more than
90% confidence**. You cannot use such a number to decide when to trust the model.

### 1.2 Choosing the architecture is guesswork

The course asks us to "identify the appropriate deep architecture for a given application".
In practice you train a CNN, an LSTM and a GRU, keep the best one, and throw the rest away.
But they are good on *different* inputs:

| On the AG News test set | |
|---|---|
| best single expert (TextCNN) | 89.5% correct |
| at least one of the four experts is right | **92.7%** of inputs |
| the experts disagree | 7.7% of inputs |

Picking the right expert *for each input* is worth up to 3 points of accuracy, and keeping one
model throws that away.

### 1.3 Strong models are expensive

System-One models such as Laya (and TypeSafe's Jev) read text with a large encoder and return
typed, calibrated decisions. They are strong, but on a laptop CPU the encoder costs tens to hundreds
of milliseconds per input, against **under 1 ms** for a TextCNN. Running the big model on every
input wastes most of that time on easy inputs.

### 1.4 Even System-One models are not perfectly calibrated

On the public typed-decisions leaderboard, Laya's fine-tuned checkpoint has an expected
calibration error (ECE) of **0.213**: its probabilities are often off by about 20 points.

---

## 2. The fix: ADAR

```
                 ┌────────────── cheap path (~5 ms) ──────────────┐
input text ──►  TextCNN · RNN · LSTM · GRU  ──►  router: "are the experts sure?"
                                                     │ yes → answer now
                                                     │ no  (about 1 in 10 inputs)
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

The head has about **2 million parameters**. Laya (421M) is not retrained.

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

## 3. The improvement

> **Where these numbers come from.** Laya's weights could not be downloaded in the environment
> where the code was built and checked, so this check used **DistilBERT as a stand-in encoder**,
> on **30,000 AG News training rows** and the **full 7,600-row test set**. The Kaggle notebook
> produces the same tables with the real Laya encoder on the full datasets. Report those.
> All models are temperature-scaled on validation data; claims hold only where the paired-bootstrap
> 95% confidence interval excludes zero. Full report: [`qc/ag_news_standin/REPORT.md`](qc/ag_news_standin/REPORT.md).

### 3.1 Accuracy and honesty

| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ | CPU ms / input |
|---|---|---|---|---|---|
| Random Forest (TF-IDF) | 86.2% | 0.399 | 0.202 | 0.016 | 68.3 |
| Bi-RNN | 88.4% | 0.354 | 0.177 | 0.024 | 1.3 |
| Stand-in encoder + Logistic Regression | 89.0% | 0.319 | 0.163 | 0.014 | 19.7 |
| Bi-LSTM | 89.3% | 0.329 | 0.167 | 0.025 | 1.0 |
| Stacking (encoder + experts → Logistic Regression) | 89.3% | 0.311 | 0.160 | 0.016 | 24.8 |
| Bi-GRU | 89.4% | 0.323 | 0.164 | 0.030 | 2.4 |
| TextCNN | 89.5% | 0.332 | 0.164 | 0.028 | 0.4 |
| Average of the 4 experts | 89.7% | 0.318 | 0.159 | 0.026 | 5.1 |
| Naive Bayes (TF-IDF) | 89.8% | 0.328 | 0.159 | 0.025 | 1.8 |
| Logistic Regression (TF-IDF) | 90.6% | 0.298 | 0.147 | 0.020 | 0.8 |
| Linear SVM (TF-IDF) | 90.7% | 0.287 | 0.143 | 0.009 | 3.4 |
| sklearn MLPClassifier (TF-IDF) | 90.9% | 0.289 | 0.144 | 0.019 | 18.1 |
| **ADAR cascade** (Laya-side called on 9% of inputs) | **91.2%** | 0.295 | 0.144 | 0.021 | **7.0** |
| **ADAR** (mean of 3 seeds) | **91.7% ± 0.1** | **0.244** | **0.127** | **0.007** | 26.1 |

What this shows:

* **ADAR is the most accurate model**: +0.8 points over the best scikit-learn model
  (95% CI +0.3 to +1.3) and +2.2 points over the best syllabus model (CI +1.6 to +2.7).
* **ADAR's probabilities are the most honest**: NLL 15% lower and Brier 11% lower than the best
  baseline, and the lowest calibration error (0.007).
* **The cascade gets most of the gain for a quarter of the cost**: 91.2% at 7 ms per input,
  more accurate than every baseline, while calling the expensive path on only 9% of inputs.
* **It is stable**: three training seeds differ by 0.06 points.

### 3.2 Which parts matter (ablations)

| Variant | Accuracy | NLL | Verdict |
|---|---|---|---|
| full ADAR | 91.7% | 0.244 | – |
| without the experts (encoder only) | 90.8% | 0.276 | experts add +0.9 points (significant) |
| without the encoder (experts only) | 89.7% | 0.303 | encoder adds +2.0 points (significant) |
| without attention (no transformer layers) | 91.5% | 0.255 | +0.2 points, **not significant**; attention mainly helps the probabilities |

### 3.3 What the head learned about the architectures

| Expert | Average routing weight | Real accuracy |
|---|---|---|
| Bi-GRU | 0.187 | 89.4% |
| TextCNN | 0.184 | 89.5% |
| Bi-LSTM | 0.117 | 89.3% |
| Bi-RNN | **0.065** | 88.4% |

Without being told, the head trusts the plain RNN least. It is the weakest expert here, and the
architecture the syllabus flags for vanishing gradients. Its per-input "is this expert right?" predictions separate right from wrong
with an AUC of about 0.89.

---

## 4. Limits, stated plainly

* The numbers above use a **stand-in encoder**. The Laya numbers come from the Kaggle run.
* Laya is **frozen**; ADAR trains only the head.
* Attention's accuracy gain is **not significant** at this scale.
* The cascade threshold chosen on validation lost **0.47 points** on test against always using the
  expensive path (the target was at most 0.5).
* The REINFORCE router never escalates once the cost weight λ is 0.4 or higher.
* For typed-decisions, ADAR-Cal **re-calibrates** Laya; it does not change Laya's answers.
* Only text classification is covered so far.

## 5. How it maps to the syllabus

| Unit | In ADAR |
|---|---|
| I | TextCNN, backpropagation, initialisation, BatchNorm, gradient clipping against exploding gradients, AdamW with a cosine schedule |
| II | CNNs for classification; training a new head on a frozen pretrained network (fine-tuning) |
| III | RNN, LSTM, GRU; embeddings; attention and a transformer head; a BERT-family encoder (Laya) |
| IV | REINFORCE policy-gradient router (reward = correct − λ·cost) |

## 6. Next steps

1. Run the notebook on Kaggle (`KAGGLE_SETUP.md`) to get the real Laya numbers on AG News and Emotion.
2. Publish the heads on Hugging Face (`GITHUB_AND_HF_UPLOAD.md`).
3. Submit to typed-decisions if ADAR-Cal improves calibration (`LEADERBOARD_SUBMISSION.md`).
4. Next phase: RouterArena, with LLMs as the experts.
