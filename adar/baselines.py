"""scikit-learn baselines (the 'standard models' comparison).

TF-IDF features: Multinomial NB, Logistic Regression, Linear SVM (Platt-calibrated), Random Forest,
sklearn MLPClassifier. Embedding features: Logistic Regression on the frozen Laya encoder
(a linear probe - the strongest 'no new architecture' baseline).
"""
import os
import pickle
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .utils import LOG, load_json, save_json, save_npz

TFIDF_MODELS = {
    "sk_naive_bayes": lambda: MultinomialNB(alpha=0.1),
    "sk_logreg": lambda: LogisticRegression(C=10.0, max_iter=2000),
    "sk_linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=0.5), cv=3, method="sigmoid"),
    "sk_random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, min_samples_leaf=1, random_state=0),
    # MLP on all 200k TF-IDF columns is impractically slow (dense 200k x 256 weight updates);
    # it gets the 20k most informative columns (chi-squared), selected on its own training rows.
    "sk_mlp": lambda: make_pipeline(SelectKBest(chi2, k=20000),
                                    MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=15,
                                                  n_iter_no_change=3, random_state=0, batch_size=256)),
}
# Random forest / sklearn-MLP are slow on 100k+ sparse rows; cap their training rows (reported).
ROW_CAP = {"sk_random_forest": 30000, "sk_mlp": 40000}


def _latency(fn, texts, n=50):
    times = []
    for t in texts[:n]:
        t0 = time.perf_counter()
        fn([t])
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_tfidf_baselines(task, out_dir, models=tuple(TFIDF_MODELS), seed=0):
    os.makedirs(out_dir, exist_ok=True)
    vec_path = os.path.join(out_dir, "tfidf.pkl")
    if os.path.exists(vec_path):
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
    else:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        vec.fit(task.x_train)
        with open(vec_path, "wb") as f:
            pickle.dump(vec, f)
    xtr, xva, xte = vec.transform(task.x_train), vec.transform(task.x_val), vec.transform(task.x_test)
    res = {}
    rng = np.random.RandomState(seed)
    for name in models:
        npz = os.path.join(out_dir, f"{name}.npz")
        if os.path.exists(npz):
            res[name] = load_json(npz.replace(".npz", ".json"))
            LOG.info("baseline %s already done", name)
            continue
        idx = np.arange(len(task.y_train))
        cap = ROW_CAP.get(name)
        if cap and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        t0 = time.time()
        clf = TFIDF_MODELS[name]()
        clf.fit(xtr[idx], task.y_train[idx])
        fit_s = time.time() - t0
        pv, pt = clf.predict_proba(xva), clf.predict_proba(xte)
        lat = _latency(lambda t: clf.predict_proba(vec.transform(t)), task.x_test)
        save_npz(npz, val=pv.astype(np.float32), test=pt.astype(np.float32))
        info = {"name": name, "train_rows": int(len(idx)), "fit_seconds": round(fit_s, 1), "cpu_latency_ms": lat,
                "val_acc": float((pv.argmax(1) == task.y_val).mean())}
        save_json(npz.replace(".npz", ".json"), info)
        res[name] = info
        LOG.info("baseline %s: val_acc=%.4f fit=%.0fs rows=%d", name, info["val_acc"], fit_s, len(idx))
    return {k: v["val_acc"] for k, v in res.items()}


def run_embedding_probe(task, emb, out_dir, name="laya_linear_probe"):
    """Logistic regression on frozen encoder embeddings, trained on the whole train pool."""
    npz = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(npz):
        return load_json(npz.replace(".npz", ".json"))
    scaler = StandardScaler().fit(emb["train"])
    t0 = time.time()
    clf = LogisticRegression(C=0.5, max_iter=3000)
    clf.fit(scaler.transform(emb["train"]), task.y_train)
    pv = clf.predict_proba(scaler.transform(emb["val"]))
    pt = clf.predict_proba(scaler.transform(emb["test"]))
    lat = _latency(lambda z: clf.predict_proba(scaler.transform(np.asarray(z))), list(emb["test"]))
    save_npz(npz, val=pv.astype(np.float32), test=pt.astype(np.float32))
    info = {"name": name, "fit_seconds": round(time.time() - t0, 1), "cpu_latency_ms_head_only": lat,
            "val_acc": float((pv.argmax(1) == task.y_val).mean())}
    save_json(npz.replace(".npz", ".json"), info)
    LOG.info("%s: val_acc=%.4f", name, info["val_acc"])
    return info
