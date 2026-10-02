"""ADAR: Adaptive Deep Architecture Routing with a System-One decision head.

Modules
-------
utils      logging, seeding, atomic saves, retries, OOM back-off
runstate   resumable stage manager (Kaggle-safe checkpoint/restore)
data       dataset loading + label descriptions + stacking folds
metrics    accuracy / macro-F1 / NLL / Brier / ECE / temperature scaling
experts    syllabus experts: TextCNN, RNN, LSTM, GRU (train/resume/predict)
baselines  scikit-learn baselines on TF-IDF and on Laya embeddings
encoder    frozen Laya encoder (or a HF stand-in) with sharded multi-GPU caching
head       the ADAR System-One head (model, training, save/load)
router     cascade router (threshold + REINFORCE policy) and cost curves
laya_tasks Laya zero-shot baseline + typed-decisions leaderboard evaluation
report     comparison tables, plots, model card, output bundle
pipeline   stage functions the notebook calls
"""

__version__ = "0.1.0"
