# ADAR results — ag_news

ag_news: train=114000 val=6000 test=7600 classes=4 folds=3

All probabilistic models are temperature-scaled on the validation split (ECE raw = before). CPU latency = median batch-1 time on the run's CPU. Test rows are identical for every model.

| Model | Family | Syllabus unit | Accuracy | Macro-F1 | NLL | Brier | ECE (TS) | ECE (raw) | AURC | CPU ms/sample | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest (TF-IDF) | scikit-learn | - | 0.8689 | 0.8681 | 0.3842 | 0.1980 | 0.0137 | 0.2183 | 0.0382 | 66.89 | - |
| Laya encoder + Logistic Regression | System One + sklearn | III | 0.9053 | 0.9052 | 0.2778 | 0.1450 | 0.0123 | 0.0207 | 0.0215 | 472.15 | 394.78M |
| Naive Bayes (TF-IDF) | scikit-learn | - | 0.9076 | 0.9073 | 0.3011 | 0.1471 | 0.0259 | 0.0455 | 0.0285 | 2.95 | - |
| sklearn MLPClassifier (TF-IDF) | scikit-learn | - | 0.9124 | 0.9122 | 0.2750 | 0.1383 | 0.0178 | 0.0183 | 0.0232 | 27.24 | - |
| Bi-RNN | Deep learning (syllabus) | III | 0.9150 | 0.9149 | 0.2634 | 0.1333 | 0.0187 | 0.0178 | 0.0209 | 3.39 | 3.91M |
| TextCNN | Deep learning (syllabus) | I-II | 0.9171 | 0.9169 | 0.2580 | 0.1284 | 0.0192 | 0.0171 | 0.0210 | 1.11 | 4.04M |
| Stacking LogReg (Laya emb + experts) | Ensemble | I-III | 0.9204 | 0.9203 | 0.2323 | 0.1203 | 0.0091 | 0.0188 | 0.0154 | 489.21 | - |
| Bi-LSTM | Deep learning (syllabus) | III | 0.9230 | 0.9228 | 0.2324 | 0.1211 | 0.0140 | 0.0159 | 0.0157 | 6.40 | 4.11M |
| Bi-GRU | Deep learning (syllabus) | III | 0.9241 | 0.9239 | 0.2365 | 0.1218 | 0.0175 | 0.0179 | 0.0164 | 6.16 | 4.04M |
| Average of syllabus experts | Ensemble | I-III | 0.9247 | 0.9246 | 0.2349 | 0.1200 | 0.0164 | 0.0173 | 0.0166 | 17.06 | 16.09M |
| Logistic Regression (TF-IDF) | scikit-learn | - | 0.9251 | 0.9250 | 0.2410 | 0.1202 | 0.0149 | 0.0149 | 0.0184 | 2.79 | - |
| Linear SVM (TF-IDF) | scikit-learn | - | 0.9251 | 0.9250 | 0.2366 | 0.1179 | 0.0076 | 0.0228 | 0.0194 | 2.76 | - |
| Laya zero-shot (typed choice) | System One | III | 0.9253 | 0.9252 | 0.2366 | 0.1203 | 0.0214 | 0.0641 | 0.0167 | 706.27 | 394.78M |
| ADAR ablation: laya only | Ablation | - | 0.9182 | 0.9182 | 0.2380 | 0.1255 | 0.0084 | 0.0142 | 0.0158 | 473.92 | 2.33M |
| ADAR ablation: experts only | Ablation | - | 0.9263 | 0.9262 | 0.2266 | 0.1173 | 0.0119 | 0.0113 | 0.0152 | 19.40 | 2.33M |
| ADAR ablation: no attention | Ablation | - | 0.9320 | 0.9320 | 0.2020 | 0.1048 | 0.0091 | 0.0100 | 0.0120 | 490.80 | 1.28M |
| **ADAR (ours, mean of 3 seeds)** | ADAR (ours) | I-IV | 0.9328 ± 0.0016 | 0.9328 ± 0.0016 | 0.1973 ± 0.0030 | 0.1033 ± 0.0018 | 0.0092 ± 0.0008 | 0.0093 | 0.0115 | 491.73 | 2.33M |
| **ADAR cascade (escalates 36%)** | ADAR (ours) | I-IV | 0.9332 | 0.9332 | 0.2077 | 0.1057 | 0.0111 | 0.0111 | 0.0141 | 189.19 | - |

## QC gates (paired bootstrap, 95% CI)

- PASS — ADAR accuracy > best scikit-learn model (Logistic Regression (TF-IDF)): delta=+0.95pp, 95% CI [+0.47, +1.43]
- PASS — ADAR accuracy > best syllabus deep model (Bi-GRU): delta=+1.05pp, 95% CI [+0.63, +1.51]
- PASS — ADAR accuracy > best expert ensemble (Average of syllabus experts): delta=+0.99pp, 95% CI [+0.57, +1.42]
- PASS — ADAR accuracy > best Laya zero-shot (Laya zero-shot (typed choice)): delta=+0.93pp, 95% CI [+0.42, +1.47]
- PASS — ADAR accuracy > best Laya linear probe (Laya encoder + Logistic Regression): delta=+2.93pp, 95% CI [+2.43, +3.46]
- PASS — ADAR accuracy > best stacking baseline (Stacking LogReg (Laya emb + experts)): delta=+1.42pp, 95% CI [+1.00, +1.84]
- PASS — ADAR NLL lower than every baseline (best: Stacking LogReg (Laya emb + experts)): ADAR 0.1949 vs 0.2323
- PASS — ADAR BRIER lower than every baseline (best: Linear SVM (TF-IDF)): ADAR 0.1023 vs 0.1179
- PASS — ADAR ECE (after TS) <= best baseline ECE + 0.005 (best: Linear SVM (TF-IDF)): ADAR 0.0088 vs 0.0076
- PASS — full ADAR >= ADAR ablation: laya only (accuracy; NLL lower): delta=+1.64pp CI [+1.18, +2.12], NLL 0.1949 vs 0.2380
- PASS — full ADAR >= ADAR ablation: experts only (accuracy; NLL lower): delta=+0.83pp CI [+0.41, +1.28], NLL 0.1949 vs 0.2266
- PASS — full ADAR >= ADAR ablation: no attention (accuracy; NLL lower): delta=+0.26pp CI [+0.00, +0.57], NLL 0.1949 vs 0.2020
- PASS — ADAR stable across seeds (accuracy std < 0.5pp): std=0.16pp over 3 seeds
- PASS — Cascade keeps accuracy within 0.5pp of always-Laya while skipping Laya calls: acc 0.9332 vs 0.9346; Laya called on 36% of inputs; mean cost 189.2 vs 491.7 ms

## How ADAR routes between the syllabus experts (test set)

| Expert | mean routing weight α | predicted P(correct) | actual accuracy | AUC of P(correct) |
|---|---|---|---|---|
| TextCNN | 0.161 | 0.920 | 0.917 | 0.910 |
| Bi-RNN | 0.067 | 0.919 | 0.915 | 0.913 |
| Bi-LSTM | 0.150 | 0.925 | 0.923 | 0.900 |
| Bi-GRU | 0.160 | 0.925 | 0.924 | 0.899 |

## Cascade (cheap experts first, Laya + ADAR only when needed)

- tier-0 (experts) accuracy 0.9247 at 17.06 ms
- always escalate: accuracy 0.9346 at 491.7 ms
- chosen threshold (picked on validation): accuracy 0.9332, Laya called on 36.3% of inputs, mean 189.2 ms
- oracle router (upper bound): accuracy 0.9487 escalating 7.5%
- REINFORCE policy points: λ=0.05: acc 0.9325 @ 16% escalated; λ=0.1: acc 0.9274 @ 2% escalated; λ=0.2: acc 0.9247 @ 0% escalated; λ=0.4: acc 0.9247 @ 0% escalated; λ=0.8: acc 0.9247 @ 0% escalated; λ=1.6: acc 0.9247 @ 0% escalated

![reliability](reliability.png) ![accuracy vs latency](accuracy_vs_latency.png) ![risk coverage](risk_coverage.png)
