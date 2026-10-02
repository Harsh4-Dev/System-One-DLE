"""Metrics and calibration. All probabilistic outputs are (N, K) arrays of probabilities."""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

EPS = 1e-12


def to_logits(p):
    return np.log(np.clip(p, EPS, 1.0))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))))


def brier(p, y):
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(p, y, bins=15):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def reliability_bins(p, y, bins=10):
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf <= edges[1])
        if sel.any():
            rows.append((float((edges[i] + edges[i + 1]) / 2), float(conf[sel].mean()),
                         float(correct[sel].mean()), int(sel.sum())))
    return rows


def risk_coverage(p, y, points=50):
    """Accuracy on the most-confident fraction `coverage` of inputs (selective classification)."""
    order = np.argsort(-p.max(1))
    correct = (p.argmax(1) == y)[order].astype(float)
    cum = np.cumsum(correct) / np.arange(1, len(y) + 1)
    covs = np.linspace(1.0 / points, 1.0, points)
    return [(float(c), float(cum[max(0, int(round(c * len(y))) - 1)])) for c in covs]


def aurc(p, y):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-p.max(1))
    err = (p.argmax(1) != y)[order].astype(float)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(y) + 1)))


def fit_temperature(logits, y, lo=0.05, hi=20.0):
    """Single temperature T minimising validation NLL of softmax(logits / T)."""
    def f(log_t):
        return nll(softmax(logits / np.exp(log_t)), y)
    r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


def apply_temperature(p_or_logits, t, is_prob=True):
    z = to_logits(p_or_logits) if is_prob else p_or_logits
    return softmax(z / t)


def all_metrics(p, y):
    pred = p.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "nll": nll(p, y),
        "brier": brier(p, y),
        "ece": ece(p, y),
        "aurc": aurc(p, y),
    }


def bootstrap_ci(p, y, metric="accuracy", n=1000, seed=0):
    rng = np.random.RandomState(seed)
    pred = p.argmax(1)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if metric == "accuracy":
            vals.append((pred[i] == y[i]).mean())
        else:
            vals.append(all_metrics(p[i], y[i])[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(p_a, p_b, y, n=2000, seed=0):
    """95% CI of accuracy(a) - accuracy(b) on the same examples."""
    rng = np.random.RandomState(seed)
    ca = (p_a.argmax(1) == y).astype(float)
    cb = (p_b.argmax(1) == y).astype(float)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        d.append(ca[i].mean() - cb[i].mean())
    return float(np.mean(ca - cb)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
