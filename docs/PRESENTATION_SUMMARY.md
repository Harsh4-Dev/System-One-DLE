# ADAR: Adaptive Deep Architecture Routing (presentation summary)

**One line:** ADAR is a small trained head (2.3M parameters) on top of the open System-One model **Laya**
and four syllabus networks (TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU). For every input it decides **which
model to trust**, **how sure it is**, and **whether the expensive model is needed at all**.

All numbers come from one full Kaggle run (GPU T4 ×2, 63 min, real Laya model):
`results/kaggle_run/summary.json`. A result counts as "better" only if its 95% bootstrap confidence
interval excludes zero.

## 1. The problem (from our own run)

| Problem | Evidence |
|---|---|
| The biggest model is not always right | Laya gets **58.1%** on Emotion tweets; a 1.2M-parameter Bi-LSTM gets **92.2%** |
| Big models are slow | On a CPU: Laya **706 ms** per input, TextCNN **1.1 ms** (635× faster) |
| Confidence is not honest | Laya's fine-tuned checkpoint has calibration error (ECE) **0.214** on the typed-decisions benchmark |
| Picking one model per dataset wastes accuracy | A perfect per-input choice would reach **94.87%** on AG News, calling Laya on only 8% of inputs |

The course objective is to "identify the appropriate deep architecture for a given application".
ADAR makes that choice **per input** instead of once per dataset.

## 2. How it works

1. **Experts answer first.** The four syllabus networks run in 17 ms in total.
2. **Router:** "Are the experts sure?" If yes, answer immediately. If not, escalate.
3. **On escalation:** the frozen Laya encoder reads the text, and the **ADAR head** combines Laya's view with the experts' votes.
4. **Inside the head:** tokens for the input (STATE), each answer option and each expert go through a 2-layer transformer. It outputs three typed answers, like a System-One model:
   - **choice**: a probability for every option (the answer);
   - **noul × 4**: "is this expert right on this input?", used as that expert's routing weight;
   - **act**: "is my own answer right?", used to abstain or escalate.
5. **Training for honesty:**
   - **Out-of-fold opinions:** an expert's opinion is only used on data it never trained on.
   - **Loss = log score + Brier score:** proper scoring rules, so the head is rewarded only for honest probabilities.
   - **Temperature scaling** on validation data.
6. **Two routers:** a threshold on P(experts right), and a **REINFORCE** policy with reward = correct − λ·cost.

## 3. Results

**Setup:** two datasets, compared against 13 baselines. Every model is calibrated on validation data, and ADAR is averaged over 3 seeds.
- **AG News:** 114k train, 7,600 test, 4 news topics.
- **Emotion:** 16k train, 2,000 test, 6 emotions.

| | AG News | Emotion |
|---|---|---|
| **ADAR** | **93.28%** ± 0.16 | **92.00%** ± 0.19 |
| ADAR cascade | 93.32% (Laya on 36%) | 92.20% (Laya on 0.1%) |
| Laya zero-shot | 92.53% | 58.05% |
| Best syllabus network | 92.41% (Bi-GRU) | 92.25% (Bi-LSTM) |
| Best scikit-learn model | 92.51% (Log. regression) | 88.15% (Linear SVM) |
| Quality checks passed | **14 / 14** | 10 / 14 |

- **AG News:** ADAR significantly beats all three reference models.
  - vs Laya: **+0.93 pts** (CI +0.42 to +1.47)
  - vs Bi-GRU: +1.05 pts
  - vs scikit-learn: +0.95 pts
- **Honesty:** ADAR has the lowest log-loss of every model: 0.197, against 0.232 for the best baseline (**15% lower**).
- **Cost:** the AG News cascade matches ADAR's accuracy at 189 ms per input instead of 492 ms (**2.6× cheaper**).
- **Emotion:** ADAR beats Laya by **+34 pts** and the best scikit-learn model by +4.1 pts, but only **ties** Bi-LSTM (+0.00, CI −0.75 to +0.75). The router learned that Laya doesn't help on tweets: it calls Laya on 0.1% of inputs, giving 92.2% at 9.6 ms instead of 388 ms per input (**41× cheaper**).
- **Ablations (AG News):** each part was removed and the head retrained.
  - Without the experts: −1.64 pts (significant).
  - Without Laya: −0.83 pts (significant).
  - Without attention: −0.26 pts (not significant). Attention mainly improves the probabilities: log-loss 0.202 without it, 0.197 with it.
- **What the head learned:** it gives the plain RNN the lowest routing weight (0.067), without being told. Its "is this expert right?" predictions reach AUCs of 0.90–0.91.
- **REINFORCE router:** at λ = 0.05 it calls Laya on 16% of articles and reaches 93.25%, about as accurate as the threshold router (93.32%) with less than half the escalations (36%).

## 4. Public benchmark: typed-decisions

- **Our scorer matches the leaderboard:** it reproduces the published accuracy for Laya typed-decisions (**0.7665** vs 0.766).
- **We add the metrics the board leaves out:** Brier 0.0615, KL 0.117, ECE 0.214.
- **Calibration study:** ADAR's calibration step uses temperatures fitted on the train split only. On base Laya it lowers:
  - ECE from **0.173 to 0.021** (−88%)
  - Brier from 0.315 to 0.236
  - KL from 0.569 to 0.433

  Accuracy is unchanged (0.363), because calibration never changes the chosen answer.

## 5. Where we submit, and why

| Target | Decision | Why |
|---|---|---|
| typed-decisions Discussion | **Submit now** | Fills in the missing metrics for Laya; read by the System-One community |
| Hugging Face model pages | **Publish now** | Both trained heads, with model cards generated from this run |
| RouterArena (ICLR 2026) | **Next phase** | Ranked leaderboard for exactly what ADAR does; needs LLM experts and an API budget |
| "Laya + ADAR-Cal" leaderboard row | **Do not submit** | Accuracy 0.363 is below the board's 0.470 baseline (always predicting the label frequencies) |

## 6. Syllabus fit and how we differ from other projects

| Unit | What ADAR uses |
|---|---|
| I | TextCNN, BatchNorm, gradient clipping, AdamW with a cosine schedule |
| II | CNN classifiers; a new head trained on a frozen pretrained network |
| III | RNN, LSTM, GRU, embeddings, attention, a transformer head, a BERT-family encoder |
| IV | REINFORCE policy gradient for routing |

Most class projects implement one architecture from one unit (e.g. R-CNN, DCGAN, a BERT classifier,
DQN). ADAR **combines architectures from all four units and decides between them per input**.

## 7. Limits (say these plainly)

- Text classification only so far; Laya is frozen and only the head is trained.
- On Emotion, ADAR ties the best syllabus network rather than beating it.
- Attention's accuracy gain is not significant.
- ADAR-Cal improves calibration, not accuracy.
- **Next steps:** RouterArena with LLMs as experts, an image task with CNN experts, an echo-state-network expert, and a short arXiv paper.

## 8. Demo (real output from the run)

```python
clf = ADARClassifier.from_pretrained("adar-head-ag-news")
clf.predict(["NASA's new telescope captured images of a distant galaxy cluster."])
# -> sci_tech 0.927 · P(answer right) 0.966 · escalate: no
# "Real Madrid beat Barcelona 3-1 ..." -> sports 0.990 · P(answer right) 0.998
```

**Links:**
- Deck: https://claude.ai/artifact/Apmff51782q7q3aasjaLHJ
- Explainer: https://claude.ai/artifact/Hjvfc4zov1kefnSsrZggYL
- Report: `docs/ADAR_Project_Report.docx`
- Code: github.com/Harsh4-Dev/System-One-DLE
