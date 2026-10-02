"""Unit tests for the parts that must not silently break: resume/failure handling, metrics,
the head's structure, and the typed-decisions scorer. Run: python -m pytest -q tests"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adar import experts, head, laya_tasks, metrics, router  # noqa: E402
from adar.encoder import cached_embed  # noqa: E402
from adar.runstate import RunState  # noqa: E402


# ---------------------------------------------------------------- run state ---------------
def test_runstate_skip_fail_block(tmp_path):
    rs = RunState(str(tmp_path), continue_on_error=True)
    calls = []
    assert rs.run("a", lambda: calls.append(1) or {"ok": 1}) == {"ok": 1}
    assert rs.run("a", lambda: calls.append(2)) == {"ok": 1}  # skipped, cached result
    assert calls == [1]
    rs.run("b", lambda: 1 / 0)
    assert rs.status("b") == "failed"
    assert "ZeroDivisionError" in open(tmp_path / "errors.log").read()
    rs.run("c", lambda: 3, deps=["b"])
    assert rs.status("c") == "blocked"
    rs2 = RunState(str(tmp_path))  # new session reads the same state
    assert rs2.done("a") and not rs2.done("b")
    assert rs2.run("b", lambda: 5) == 5  # a failed stage is retried on the next run


def test_runstate_raises_without_continue(tmp_path):
    rs = RunState(str(tmp_path), continue_on_error=False)
    with pytest.raises(ZeroDivisionError):
        rs.run("x", lambda: 1 / 0)


def test_restore_from_inputs(tmp_path):
    prev = tmp_path / "input" / "nb" / "adar_run"
    (prev / "emb").mkdir(parents=True)
    (prev / "state.json").write_text(json.dumps({"stages": {"embed:x": {"status": "done"}}}))
    (prev / "emb" / "a.npy").write_text("cached")
    cur = tmp_path / "working" / "adar_run"
    src = RunState.restore_from_inputs(str(cur), str(tmp_path / "input" / "*" / "**" / "adar_run"))
    assert src is not None and (cur / "emb" / "a.npy").exists()
    assert RunState(str(cur)).done("embed:x")


# ---------------------------------------------------------------- embedding cache ---------
class FakeEncoder:
    def __init__(self, fail_after=None):
        self.device, self.calls, self.fail_after = "cpu", 0, fail_after

    def embed(self, texts, batch_size=8):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated crash")
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_cached_embed_resumes_after_crash(tmp_path):
    texts = [f"text {i}" * (i % 5 + 1) for i in range(50)]
    with pytest.raises(RuntimeError):
        cached_embed([FakeEncoder(fail_after=2)], texts, str(tmp_path), "train", shard_size=10)
    assert len([f for f in os.listdir(tmp_path / "train") if f.startswith("shard_")]) == 2
    enc = FakeEncoder()
    out = cached_embed([enc], texts, str(tmp_path), "train", shard_size=10)
    assert enc.calls == 3  # only the 3 missing shards were recomputed
    assert np.allclose(out[:, 0], [len(t) for t in texts])
    with pytest.raises(RuntimeError, match="different data"):
        cached_embed([enc], texts[:40], str(tmp_path), "train", shard_size=10)


# ---------------------------------------------------------------- expert resume ------------
def test_expert_training_resumes(tmp_path, monkeypatch):
    rng = np.random.RandomState(0)
    ids = rng.randint(2, 50, size=(64, 12))
    lens = np.full(64, 12)
    y = rng.randint(0, 3, 64)
    ck = str(tmp_path / "e.pt")
    real = experts.predict_proba
    n = {"calls": 0}

    def crash_on_second(*a, **k):
        n["calls"] += 1
        if n["calls"] == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(experts, "predict_proba", crash_on_second)
    with pytest.raises(KeyboardInterrupt):
        experts.train_one("gru", ids, lens, y, ids, lens, y, 3, 50, "cpu", ck, epochs=4, patience=10)
    assert torch.load(ck, weights_only=False)["epoch"] == 1
    monkeypatch.setattr(experts, "predict_proba", real)
    model, hist = experts.train_one("gru", ids, lens, y, ids, lens, y, 3, 50, "cpu", ck, epochs=4, patience=10)
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]


# ---------------------------------------------------------------- metrics -----------------
def test_metrics_basics():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    m = metrics.all_metrics(p, y)
    assert m["accuracy"] == 0.75
    assert np.isclose(m["brier"], np.mean([0.02, 0.08, 0.72, 0.5]))
    perfect = np.eye(2)[y]
    assert metrics.ece(perfect, y) == 0.0


def test_temperature_recovers_scale():
    rng = np.random.RandomState(0)
    z = rng.randn(4000, 5) * 2
    y = np.array([rng.choice(5, p=metrics.softmax(r)) for r in z])
    t = metrics.fit_temperature(z * 3.0, y)  # logits made 3x too sharp
    assert abs(t - 3.0) < 0.3


# ---------------------------------------------------------------- head -------------------
def _head_inputs(B=6, K=4, E=3, D=16):
    torch.manual_seed(0)
    return torch.randn(B, D), torch.randn(K, D), torch.softmax(torch.randn(B, E, K), -1)


def test_head_shapes_and_option_equivariance():
    z, o, p = _head_inputs()
    m = head.ADARHead(16, 16, 3, d=32).eval()
    out = m(z, o, p)
    assert out["logits"].shape == (6, 4) and out["expert_logit"].shape == (6, 3) and out["act_logit"].shape == (6,)
    perm = torch.tensor([2, 0, 3, 1])
    out_p = m(z, o[perm], p[:, :, perm])
    assert torch.allclose(out["logits"][:, perm], out_p["logits"], atol=1e-5)  # options are a set, not a list


def test_head_ablation_variants_run():
    z, o, p = _head_inputs()
    for var in ({"use_experts": False}, {"use_state": False}, {"use_transformer": False}):
        m = head.ADARHead(16, 16, 3, d=32, **var).eval()
        out = m(z, o, None if var.get("use_experts") is False else p)
        assert torch.isfinite(out["logits"]).all()


def test_head_learns_and_save_load(tmp_path):
    rng = np.random.RandomState(0)
    K, D, N = 3, 8, 600
    centers = rng.randn(K, D) * 2
    y = rng.randint(0, K, N)
    z = (centers[y] + rng.randn(N, D)).astype(np.float32)
    p = np.stack([np.eye(K)[(y + (rng.rand(N) < 0.3)) % K] * 0.8 + 0.2 / K for _ in range(2)], 1).astype(np.float32)
    opts = rng.randn(K, D).astype(np.float32)
    model, hist = head.train_head(z[:500], p[:500], y[:500], z[500:], p[500:], y[500:], opts,
                                  str(tmp_path / "h.pt"), "cpu", epochs=15, batch_size=64)
    acc = (head.predict(model, z[500:], opts, p[500:], "cpu")["probs"].argmax(1) == y[500:]).mean()
    assert acc > 0.85
    head.save_head(model, str(tmp_path / "pub"), {"labels": ["a", "b", "c"]}, opts)
    m2, o2, cfg = head.load_head(str(tmp_path / "pub"))
    a = head.predict(model, z[500:], opts, p[500:], "cpu")["probs"]
    b = head.predict(m2, z[500:], o2, p[500:], "cpu")["probs"]
    assert np.allclose(a, b, atol=1e-6) and cfg["temperature"] == pytest.approx(float(model.temperature))


# ---------------------------------------------------------------- router -----------------
def test_cascade_curve_endpoints():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 3, 200)
    pred0 = np.where(rng.rand(200) < 0.7, y, (y + 1) % 3)
    pred1 = np.where(rng.rand(200) < 0.9, y, (y + 1) % 3)
    pc = rng.rand(200)
    curve = router.cascade_curve(pc, pred0, pred1, y, 1.0, 10.0)
    assert curve[0]["escalation_rate"] == 0 and curve[0]["accuracy"] == (pred0 == y).mean()
    assert curve[-1]["escalation_rate"] == 1 and curve[-1]["accuracy"] == (pred1 == y).mean()


# ---------------------------------------------------------------- typed-decisions ---------
def test_typed_decision_vectors_and_metrics():
    qc = {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}
    qn = {"type": "noul", "instructions": "?"}
    qs = {"type": "score", "criteria": ["lo", "mid", "hi"]}
    assert laya_tasks._pred_vector({"probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}}, qc).tolist() == [0.2, 0.7, 0.1]
    assert laya_tasks._pred_vector({"noul": 0.8}, qn).tolist() == pytest.approx([0.2, 0.8])
    assert laya_tasks._gold_vector({"probabilities": {"0": 1, "2": 1}}, qs).tolist() == [0.5, 0, 0.5]
    decs = [{"type": "noul", "k": 2, "pred": [0.1, 0.9], "gold": [0.0, 1.0]},
            {"type": "noul", "k": 2, "pred": [0.6, 0.4], "gold": [0.0, 1.0]}]
    m = laya_tasks.td_metrics(decs)
    assert m["accuracy"] == 0.5 and m["brier_soft"] == pytest.approx((0.02 + 0.72) / 2)
    temps = laya_tasks.fit_td_temperatures(decs * 20, min_count=10)
    assert "noul" in temps and "noul:2" in temps
