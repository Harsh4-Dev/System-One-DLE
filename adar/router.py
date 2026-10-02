"""Cost-aware routing between the cheap tier and the System-One tier.

Tier 0 ("System 1, cheap"): the syllabus experts (TextCNN/RNN/LSTM/GRU) - ~1 ms on a laptop CPU.
Tier 1 ("System-One + ADAR"): Laya encoder + ADAR head - the expensive, more accurate path.

The router reads only tier-0 signals and predicts P(tier 0 is right). Two policies:
  * threshold  - escalate when P(tier0 right) < tau (tau chosen on validation)
  * REINFORCE  - a stochastic policy trained with policy gradients (Unit IV) to maximise
                 reward = 1[final answer right] - lambda * cost(escalation)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import apply_temperature, fit_temperature, to_logits
from .utils import seed_everything


def tier0_probs(expert_probs, kinds, temperature=1.0):
    p = np.mean([expert_probs[k] for k in kinds], 0)
    return apply_temperature(p, temperature) if temperature != 1.0 else p


def router_features(expert_probs, kinds):
    ps = np.stack([expert_probs[k] for k in kinds], 1)  # N,E,K
    pm = ps.mean(1)
    srt = np.sort(pm, 1)[:, ::-1]
    ent = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum(1) / np.log(pm.shape[1])
    am = ps.argmax(2)
    agree = (am == pm.argmax(1)[:, None]).mean(1)
    feats = [srt[:, 0], srt[:, 0] - srt[:, 1], ent, agree, ps.max(2).min(1), ps.max(2).max(1)]
    feats += [ps[:, e].max(1) for e in range(ps.shape[1])]
    return np.stack(feats, 1).astype(np.float32)


class ThresholdRouter:
    def fit(self, f_train, correct0_train):
        self.scaler = StandardScaler().fit(f_train)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(self.scaler.transform(f_train), correct0_train)
        return self

    def p_correct(self, f):
        return self.clf.predict_proba(self.scaler.transform(f))[:, 1]


def cascade_curve(p_c0, pred0, pred1, y, cost0, cost1, n=41):
    """Accuracy / escalation-rate / mean cost for escalation thresholds swept over quantiles of p_c0."""
    rows = []
    qs = np.unique(np.quantile(p_c0, np.linspace(0, 1, n)))
    for tau in np.concatenate([[-1.0], qs, [2.0]]):
        esc = p_c0 < tau
        pred = np.where(esc, pred1, pred0)
        rows.append({"tau": float(tau), "escalation_rate": float(esc.mean()), "accuracy": float((pred == y).mean()),
                     "mean_cost_ms": float(cost0 + esc.mean() * cost1)})
    return rows


def choose_tau(p_c0_val, pred0_val, pred1_val, y_val, tolerance=0.002):
    """Smallest escalation rate whose validation accuracy is within `tolerance` of always-escalate."""
    target = (pred1_val == y_val).mean() - tolerance
    best_tau = 2.0
    for tau in np.sort(np.unique(np.concatenate([p_c0_val, [2.0]]))):
        esc = p_c0_val < tau
        acc = (np.where(esc, pred1_val, pred0_val) == y_val).mean()
        if acc >= target:
            return float(tau)
    return best_tau


class PolicyNet(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_reinforce(f, correct0, correct1, cost_frac, lam, epochs=300, lr=1e-2, seed=0, batch=512):
    """REINFORCE with a moving-average baseline. Action 1 = escalate to tier 1.
    reward = 1[chosen tier is right] - lam * cost_frac * action."""
    seed_everything(seed)
    scaler = StandardScaler().fit(f)
    X = torch.as_tensor(scaler.transform(f), dtype=torch.float32)
    c0 = torch.as_tensor(correct0, dtype=torch.float32)
    c1 = torch.as_tensor(correct1, dtype=torch.float32)
    pol = PolicyNet(X.shape[1])
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    baseline = 0.0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        logit = pol(X[idx])
        dist = torch.distributions.Bernoulli(logits=logit)
        a = dist.sample()
        r = torch.where(a > 0, c1[idx], c0[idx]) - lam * cost_frac * a
        adv = r - baseline
        baseline = 0.9 * baseline + 0.1 * r.mean().item()
        loss = -(dist.log_prob(a) * adv).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def act(f_new):
        with torch.no_grad():
            return (torch.sigmoid(pol(torch.as_tensor(scaler.transform(f_new), dtype=torch.float32))) > 0.5).numpy()
    return act


def run_router(expert_probs_oof, expert_probs_val, expert_probs_test, kinds, y_tr, y_va, y_te,
               p1_val, p1_test, cost0_ms, cost1_ms, lambdas=(0.05, 0.1, 0.2, 0.4, 0.8, 1.6), seed=0):
    """Fit the routers and produce test-set operating points."""
    t0 = fit_temperature(to_logits(tier0_probs(expert_probs_val, kinds)), y_va)
    p0 = {s: tier0_probs(d, kinds, t0) for s, d in
          (("train", expert_probs_oof), ("val", expert_probs_val), ("test", expert_probs_test))}
    f = {s: router_features(d, kinds) for s, d in
         (("train", expert_probs_oof), ("val", expert_probs_val), ("test", expert_probs_test))}
    r = ThresholdRouter().fit(f["train"], (p0["train"].argmax(1) == y_tr).astype(int))
    pc_val, pc_test = r.p_correct(f["val"]), r.p_correct(f["test"])
    pred0_val, pred0_test = p0["val"].argmax(1), p0["test"].argmax(1)
    pred1_val, pred1_test = p1_val.argmax(1), p1_test.argmax(1)
    curve = cascade_curve(pc_test, pred0_test, pred1_test, y_te, cost0_ms, cost1_ms)
    oracle_esc = pred0_test != y_te
    tau = choose_tau(pc_val, pred0_val, pred1_val, y_va)
    esc = pc_test < tau
    pred = np.where(esc, pred1_test, pred0_test)
    # cascade probabilities (for calibration metrics): tier-1 probs where escalated, else tier-0
    p_casc = np.where(esc[:, None], p1_test, p0["test"])
    chosen = {"tau": tau, "escalation_rate": float(esc.mean()), "accuracy": float((pred == y_te).mean()),
              "mean_cost_ms": float(cost0_ms + esc.mean() * cost1_ms),
              "always_tier1_accuracy": float((pred1_test == y_te).mean()),
              "always_tier1_cost_ms": float(cost0_ms + cost1_ms),
              "tier0_accuracy": float((pred0_test == y_te).mean()),
              "oracle_escalation_rate": float(oracle_esc.mean()),
              "oracle_accuracy": float((np.where(oracle_esc, pred1_test, pred0_test) == y_te).mean())}
    rl = []
    cost_frac = cost1_ms / (cost0_ms + cost1_ms)
    c0v, c1v = (pred0_val == y_va).astype(float), (pred1_val == y_va).astype(float)
    for lam in lambdas:
        act = train_reinforce(f["val"], c0v, c1v, cost_frac, lam, seed=seed)
        a = act(f["test"])
        pr = np.where(a, pred1_test, pred0_test)
        rl.append({"lambda": lam, "escalation_rate": float(a.mean()), "accuracy": float((pr == y_te).mean()),
                   "mean_cost_ms": float(cost0_ms + a.mean() * cost1_ms)})
    return {"tier0_temperature": t0, "chosen": chosen, "curve": curve, "reinforce": rl,
            "p_cascade_test": p_casc, "p_tier0_test": p0["test"], "p_tier0_val": p0["val"]}
