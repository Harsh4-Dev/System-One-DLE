"""Datasets, label descriptions and stacking folds.

Each dataset becomes a `TaskData` with a train pool (split into K folds so every expert's
predictions on train are out-of-fold), a validation split (early stopping, temperature scaling,
router thresholds) and the official test split (touched only for final reporting).
"""
import csv
import io
import os
from dataclasses import dataclass, field

import numpy as np

from .utils import LOG, retry

# Option descriptions are what the System-One head (and Laya) reads for each label.
TASKS = {
    "ag_news": {
        "hf": [("fancyzhx/ag_news", None), ("ag_news", None)],
        "labels": ["world", "sports", "business", "sci_tech"],
        "descriptions": {
            "world": "world news: international affairs, politics, governments, conflicts, diplomacy",
            "sports": "sports news: games, matches, teams, players, leagues, championships",
            "business": "business news: companies, markets, economy, earnings, stocks, trade",
            "sci_tech": "science and technology news: computers, internet, software, space, research",
        },
        "instruction": "Which news category does this article belong to?",
        "github_csv": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{split}.csv",
    },
    "emotion": {
        "hf": [("dair-ai/emotion", "split"), ("dair-ai/emotion", None)],
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "descriptions": {
            "sadness": "sadness: feeling sad, unhappy, lonely, hurt, hopeless or disappointed",
            "joy": "joy: feeling happy, glad, cheerful, pleased, content or excited",
            "love": "love: feeling loving, caring, affectionate, romantic or tender",
            "anger": "anger: feeling angry, annoyed, irritated, furious or resentful",
            "fear": "fear: feeling afraid, scared, anxious, nervous or worried",
            "surprise": "surprise: feeling surprised, amazed, shocked or astonished",
        },
        "instruction": "Which emotion does the writer express?",
    },
}


@dataclass
class TaskData:
    name: str
    labels: list
    descriptions: dict
    instruction: str
    x_train: list
    y_train: np.ndarray
    x_val: list
    y_val: np.ndarray
    x_test: list
    y_test: np.ndarray
    folds: np.ndarray = field(default=None)

    @property
    def n_classes(self):
        return len(self.labels)

    def option_texts(self):
        return [self.descriptions[l] for l in self.labels]

    def split(self, name):
        return {"train": (self.x_train, self.y_train), "val": (self.x_val, self.y_val),
                "test": (self.x_test, self.y_test)}[name]

    def fingerprint(self):
        """Hash of every split's texts+labels. Cached embeddings/predictions are only reused when it
        matches, so a session that loaded the data from a different source can never silently
        mix rows."""
        import hashlib
        h = hashlib.sha256()
        for xs, ys in (self.split("train"), self.split("val"), self.split("test")):
            for x, y in zip(xs, ys.tolist()):
                h.update(x.encode("utf-8", "ignore"))
                h.update(str(y).encode())
            h.update(b"|")
        return h.hexdigest()[:16]

    def summary(self):
        return (f"{self.name}: train={len(self.x_train)} val={len(self.x_val)} test={len(self.x_test)} "
                f"classes={self.n_classes} folds={len(np.unique(self.folds))}")


@retry(times=3)
def _hf_load(path, config):
    from datasets import load_dataset
    return load_dataset(path, config) if config else load_dataset(path)


def _from_hf(spec):
    last = None
    for path, config in spec["hf"]:
        try:
            ds = _hf_load(path, config)
            out = {}
            for split in ds:
                out[split] = (list(ds[split]["text"]), np.asarray(ds[split]["label"], dtype=np.int64))
            LOG.info("loaded %s from Hugging Face (%s)", path, ", ".join(f"{k}={len(v[0])}" for k, v in out.items()))
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            LOG.warning("HF load %s failed: %s", path, e)
    raise RuntimeError(f"could not load from Hugging Face: {last}")


def _ag_news_rows(text_iter):
    xs, ys = [], []
    for row in csv.reader(text_iter):
        if len(row) < 3:
            continue
        ys.append(int(row[0]) - 1)  # CSV labels are 1..4 in the same order as the HF dataset
        xs.append((row[1] + " " + row[2]).replace("\\", " ").strip())
    return xs, np.asarray(ys, dtype=np.int64)


@retry(times=3)
def _download(url):
    import urllib.request
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8")


def _ag_news_fallback(spec, local_dir=None):
    out = {}
    for split in ("train", "test"):
        if local_dir and os.path.exists(os.path.join(local_dir, f"{split}.csv")):
            with open(os.path.join(local_dir, f"{split}.csv"), newline="", encoding="utf-8") as f:
                out[split] = _ag_news_rows(f)
        else:
            out[split] = _ag_news_rows(io.StringIO(_download(spec["github_csv"].format(split=split))))
    LOG.info("loaded ag_news from CSV (train=%d test=%d)", len(out["train"][0]), len(out["test"][0]))
    return out


def _stratified_take(y, n, rng):
    """Indices of a class-stratified random subset of size ~n (all rows when n is None/too big)."""
    idx = np.arange(len(y))
    if n is None or n >= len(y):
        return rng.permutation(idx)
    take = []
    for c in np.unique(y):
        ci = idx[y == c]
        k = max(1, int(round(n * len(ci) / len(y))))
        take.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return rng.permutation(np.concatenate(take))


def load_task(name, n_train=None, n_val=None, n_test=None, n_folds=3, seed=0, local_dir=None, source="auto"):
    """Load a task and build the train pool / folds / val / test splits.

    n_train / n_val / n_test cap the split sizes (stratified); None keeps everything.
    When the dataset has no official validation split, n_val rows are held out of train.
    """
    spec = TASKS[name]
    raw = None
    if source in ("auto", "hf") and not local_dir:
        try:
            raw = _from_hf(spec)
        except Exception as e:  # noqa: BLE001
            if source == "hf":
                raise
            LOG.warning("falling back from HF for %s: %s", name, e)
    if raw is None:
        if name != "ag_news":
            raise RuntimeError(f"{name}: no fallback source; enable internet / HF access")
        raw = _ag_news_fallback(spec, local_dir)

    rng = np.random.RandomState(seed)
    x_tr_all, y_tr_all = raw["train"]
    if "validation" in raw:
        x_va_all, y_va_all = raw["validation"]
        vi = _stratified_take(y_va_all, n_val, rng)
        x_val, y_val = [x_va_all[i] for i in vi], y_va_all[vi]
        pool = np.arange(len(y_tr_all))
    else:
        n_val = n_val or max(1000, len(y_tr_all) // 20)
        vi = _stratified_take(y_tr_all, n_val, rng)
        x_val, y_val = [x_tr_all[i] for i in vi], y_tr_all[vi]
        pool = np.setdiff1d(np.arange(len(y_tr_all)), vi)
    ti = pool[_stratified_take(y_tr_all[pool], n_train, rng)]
    x_train, y_train = [x_tr_all[i] for i in ti], y_tr_all[ti]
    x_te_all, y_te_all = raw["test"]
    ei = np.sort(_stratified_take(y_te_all, n_test, rng))
    x_test, y_test = [x_te_all[i] for i in ei], y_te_all[ei]

    # stratified fold ids for out-of-fold expert predictions
    folds = np.zeros(len(y_train), dtype=np.int64)
    for c in np.unique(y_train):
        ci = np.where(y_train == c)[0]
        ci = ci[rng.permutation(len(ci))]
        folds[ci] = np.arange(len(ci)) % n_folds

    td = TaskData(name, list(spec["labels"]), dict(spec["descriptions"]), spec["instruction"],
                  x_train, y_train, x_val, y_val, x_test, y_test, folds)
    LOG.info(td.summary())
    return td
