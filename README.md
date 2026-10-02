# ADAR — Adaptive Deep Architecture Routing with a System-One head

ADAR puts a small trained **System-One decision head** on top of the open-source System-One model
[Laya](https://huggingface.co/convaiinnovations/laya) (Apache-2.0, ModernBERT-large, 421M) and the
classic syllabus architectures (TextCNN, RNN, LSTM, GRU). For every input it decides, with calibrated
probabilities, *what the answer is*, *which architecture to trust*, and *whether to escalate*.

```
input text ──► syllabus experts (TextCNN / RNN / LSTM / GRU)  ~1 ms on a laptop CPU
     │                    │ their probabilities
     ▼                    ▼
 frozen Laya ──► [STATE] [OPT_1 … OPT_K] [EXP_1 … EXP_E]  ──► 2-layer transformer ──►
 encoder           input   option text +    how sure each
                           expert votes     expert is
        ──► choice : calibrated P(label)                 (log score + Brier, temperature-scaled)
        ──► noul   : P(expert e is right) per expert     → routing weights α_e(x)
        ──► act    : P(this answer is right)            → abstain / escalate
```

A cascade router (threshold or **REINFORCE** policy) runs the cheap experts first and calls
Laya + ADAR only when they are unsure.

## Repository

| Path | What |
|---|---|
| `notebooks/adar_kaggle.ipynb` | **the notebook to run on Kaggle (GPU T4 ×2)** — trains, evaluates, saves weights, builds reports |
| `adar/` | the package (the notebook writes these exact files) |
| `tests/` | unit tests: resume/failure handling, metrics, head structure, typed-decisions scorer |
| `tools/build_notebook.py` | regenerates the notebook from `adar/` |
| `docs/MAIN.md` | **start here**: the problem, the fix, the improvement, in plain language (also `docs/ADAR_explained.html`) |
| `docs/KAGGLE_SETUP.md` | running the notebook on Kaggle, resuming after failures |
| `docs/LEADERBOARD_SUBMISSION.md` | preparing typed-decisions / course / Hugging Face submissions; RouterArena plan |
| `docs/GITHUB_AND_HF_UPLOAD.md` | making the repo public and publishing the trained heads |
| `docs/LOCAL_QC.md` | what was verified before the Kaggle run, and what was not |
| `docs/qc/ag_news_standin/` | the local QC report (stand-in encoder) with figures |

## Run it

1. Follow `docs/KAGGLE_SETUP.md`, or in short: upload `notebooks/adar_kaggle.ipynb` to Kaggle → **Accelerator: GPU T4 ×2**, **Internet: On**.
2. **Save Version → Save & Run All**. Expect roughly 1.5–3 hours.
3. Download `adar_outputs.zip`: comparison tables and plots (`results/`), the publishable head
   (`publish/`), typed-decisions results, `SUMMARY.md`.

Interrupted? Attach the previous version's output as an input and run again; finished stages are
skipped (see the notebook's first cell).

Local tests: `pip install torch scikit-learn scipy transformers safetensors pytest && python -m pytest -q tests`

## Syllabus coverage

| Unit | Where |
|---|---|
| I | TextCNN, backpropagation, initialisation, BatchNorm, gradient clipping (exploding gradients), AdamW + cosine schedule |
| II | CNN text classification, fine-tuning a head on a frozen pretrained network |
| III | RNN / LSTM / GRU experts, embeddings, attention + transformer head, BERT-family encoder (Laya) |
| IV | REINFORCE policy-gradient router (reward = correct − λ·cost) |

## Honest scope

* ADAR is a head on a **frozen** Laya encoder; Laya itself is not retrained.
* Claims are made only where the paired-bootstrap 95% CI excludes zero (see each `REPORT.md`).
* The typed-decisions "ADAR-Cal" entry is a calibration of existing Laya checkpoints, fitted on the train split.
