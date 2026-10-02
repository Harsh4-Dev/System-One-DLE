"""Syllabus experts: TextCNN (Unit I/II), vanilla RNN, LSTM, GRU (Unit III).

Each expert is trained K times (one per stacking fold) so that its predictions on the training
pool are out-of-fold - the ADAR head and router never see an expert's predictions on data that
expert was trained on. Val/test predictions are the average over the K fold models.
Training checkpoints every epoch and resumes after an interruption.
"""
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LOG, count_params, load_json, save_json, save_npz, seed_everything, torch_save

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts, max_size=30000, min_freq=2):
        c = Counter()
        for t in texts:
            c.update(tokenize(t))
        words = [w for w, n in c.most_common(max_size) if n >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, texts, max_len):
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        lens = np.ones(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            toks = [self.stoi.get(w, self.UNK) for w in tokenize(t)][:max_len] or [self.UNK]
            ids[i, :len(toks)] = toks
            lens[i] = len(toks)
        return ids, lens

    def __len__(self):
        return len(self.itos)


class TextCNN(nn.Module):
    """Kim (2014) CNN for sentences, with BatchNorm after each convolution."""

    def __init__(self, vocab_size, n_classes, emb=128, channels=128, kernels=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, channels, k, padding=k // 2) for k in kernels])
        self.bns = nn.ModuleList([nn.BatchNorm1d(channels) for _ in kernels])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(channels * len(kernels), n_classes)

    def forward(self, ids, lens):
        mask = (ids != 0).unsqueeze(1)  # B,1,T
        x = self.emb(ids).transpose(1, 2)  # B,E,T
        feats = []
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(x)))[:, :, : ids.size(1)]
            h = h.masked_fill(~mask, -1e4)
            feats.append(h.max(dim=2).values)
        return self.out(self.drop(torch.cat(feats, 1)))


class RNNClassifier(nn.Module):
    """Bidirectional RNN / LSTM / GRU with mean+max pooling over valid time steps."""

    CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(self, cell, vocab_size, n_classes, emb=128, hidden=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        kw = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.rnn = self.CELLS[cell](emb, hidden, batch_first=True, bidirectional=True, **kw)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, n_classes)

    def forward(self, ids, lens):
        x = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))
        mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lens[:, None]).unsqueeze(-1)
        mean = (h * mask).sum(1) / lens[:, None].clamp(min=1)
        mx = h.masked_fill(~mask, -1e4).max(1).values
        return self.out(self.drop(torch.cat([mean, mx], 1)))


EXPERT_KINDS = ("textcnn", "rnn", "lstm", "gru")
EXPERT_UNITS = {"textcnn": "Unit I-II (CNN)", "rnn": "Unit III (RNN)", "lstm": "Unit III (LSTM)",
                "gru": "Unit III (GRU)"}


def build_expert(kind, vocab_size, n_classes):
    if kind == "textcnn":
        return TextCNN(vocab_size, n_classes)
    return RNNClassifier(kind, vocab_size, n_classes)


@torch.no_grad()
def predict_proba(model, ids, lens, device, batch_size=1024):
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        b_ids = torch.as_tensor(ids[i:i + batch_size], device=device)
        b_len = torch.as_tensor(lens[i:i + batch_size], device=device)
        out.append(torch.softmax(model(b_ids, b_len).float(), -1).cpu().numpy())
    return np.concatenate(out)


def train_one(kind, ids, lens, y, val_ids, val_lens, y_val, n_classes, vocab_size, device, ckpt_path,
              epochs=5, batch_size=256, lr=2e-3, patience=2, seed=0):
    """Train one expert; resumes from ckpt_path. Returns the best (val-NLL) model."""
    seed_everything(seed)
    model = build_expert(kind, vocab_size, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start, best, best_state, bad, hist = 0, float("inf"), None, 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best, best_state, bad, hist = ck["epoch"], ck["best"], ck["best_state"], ck["bad"], ck["hist"]
        LOG.info("  resume %s from epoch %d (best val nll %.4f)", os.path.basename(ckpt_path), start, best)
        if ck.get("finished"):
            model.load_state_dict(best_state)
            return model, hist
    t_ids = torch.as_tensor(ids, device=device)
    t_lens = torch.as_tensor(lens, device=device)
    t_y = torch.as_tensor(y, device=device)
    for ep in range(start, epochs):
        model.train()
        perm = torch.randperm(len(y), device=device)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(y), batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:
                continue
            loss = F.cross_entropy(model(t_ids[b], t_lens[b]), t_y[b])
            if not torch.isfinite(loss):
                LOG.warning("  non-finite loss in %s epoch %d; skipping batch", kind, ep)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards against exploding gradients
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        pv = predict_proba(model, val_ids, val_lens, device)
        v_nll = float(-np.mean(np.log(np.clip(pv[np.arange(len(y_val)), y_val], 1e-12, 1))))
        v_acc = float((pv.argmax(1) == y_val).mean())
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1), "val_nll": v_nll, "val_acc": v_acc,
                     "seconds": round(time.time() - t0, 1)})
        LOG.info("  %s ep%d loss=%.4f val_nll=%.4f val_acc=%.4f (%.0fs)", kind, ep + 1, tot / max(nb, 1),
                 v_nll, v_acc, time.time() - t0)
        if v_nll < best - 1e-4:
            best, bad = v_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        stop = bad >= patience or ep + 1 == epochs
        torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "best": best,
                    "best_state": best_state, "bad": bad, "hist": hist, "finished": stop}, ckpt_path)
        if stop:
            break
    model.load_state_dict(best_state)
    return model, hist


def measure_cpu_latency(model, ids, lens, n=64, threads=4):
    """Median per-sample latency (ms) at batch size 1 on CPU - the 'runs on a laptop' number."""
    import copy
    m = copy.deepcopy(model).cpu().eval()
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    times = []
    with torch.no_grad():
        for i in range(min(n, len(ids))):
            a = torch.as_tensor(ids[i:i + 1])
            b = torch.as_tensor(lens[i:i + 1])
            t0 = time.perf_counter()
            m(a, b)
            times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    return float(np.median(times[5:] if len(times) > 10 else times))


def run_experts(task, out_dir, device, kinds=EXPERT_KINDS, epochs=5, max_len=64, batch_size=256, seed=0):
    """Train every expert on every fold; write experts/<kind>.npz with oof/val/test probabilities."""
    os.makedirs(out_dir, exist_ok=True)
    vocab_path = os.path.join(out_dir, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = Vocab(load_json(vocab_path))
    else:
        vocab = Vocab.build(task.x_train)
        save_json(vocab_path, vocab.itos)
    tr_ids, tr_lens = vocab.encode(task.x_train, max_len)
    va_ids, va_lens = vocab.encode(task.x_val, max_len)
    te_ids, te_lens = vocab.encode(task.x_test, max_len)
    n_folds = int(task.folds.max()) + 1
    summary = {}
    for kind in kinds:
        final = os.path.join(out_dir, f"{kind}.npz")
        if os.path.exists(final):
            LOG.info("expert %s already trained", kind)
            summary[kind] = load_json(os.path.join(out_dir, f"{kind}.json"))
            continue
        oof = np.zeros((len(task.y_train), task.n_classes), dtype=np.float32)
        val_p, test_p, hists, lat, params = [], [], [], None, None
        for f in range(n_folds):
            fold_npz = os.path.join(out_dir, f"{kind}_fold{f}.npz")
            tr = task.folds != f
            if os.path.exists(fold_npz):
                d = np.load(fold_npz)
                oof[~tr], fv, ft = d["oof"], d["val"], d["test"]
                meta = load_json(fold_npz.replace(".npz", ".json"))
                hists.append(meta["hist"])
                lat, params = meta["latency_ms"], meta["params"]
            else:
                model, hist = train_one(kind, tr_ids[tr], tr_lens[tr], task.y_train[tr], va_ids, va_lens, task.y_val,
                                        task.n_classes, len(vocab), device,
                                        os.path.join(out_dir, "ckpt", f"{kind}_fold{f}.pt"),
                                        epochs=epochs, batch_size=batch_size, seed=seed + f)
                oof[~tr] = predict_proba(model, tr_ids[~tr], tr_lens[~tr], device)
                fv = predict_proba(model, va_ids, va_lens, device)
                ft = predict_proba(model, te_ids, te_lens, device)
                lat = measure_cpu_latency(model, te_ids, te_lens)
                params = count_params(model)
                save_npz(fold_npz, oof=oof[~tr], val=fv, test=ft)
                save_json(fold_npz.replace(".npz", ".json"), {"hist": hist, "latency_ms": lat, "params": params})
                torch_save(model.state_dict(), os.path.join(out_dir, "weights", f"{kind}_fold{f}.pt"))
                hists.append(hist)
            val_p.append(fv)
            test_p.append(ft)
        val_m, test_m = np.mean(val_p, 0), np.mean(test_p, 0)
        save_npz(final, oof=oof, val=val_m, test=test_m)
        info = {"kind": kind, "params": params, "cpu_latency_ms": lat, "folds": n_folds, "history": hists,
                "oof_acc": float((oof.argmax(1) == task.y_train).mean()),
                "val_acc": float((val_m.argmax(1) == task.y_val).mean())}
        save_json(os.path.join(out_dir, f"{kind}.json"), info)
        summary[kind] = info
        LOG.info("expert %s: oof_acc=%.4f val_acc=%.4f params=%d cpu=%.2fms", kind, info["oof_acc"],
                 info["val_acc"], params, lat)
    return {k: {"oof_acc": v["oof_acc"], "val_acc": v["val_acc"]} for k, v in summary.items()}


def load_expert_probs(out_dir, kinds):
    d = {k: np.load(os.path.join(out_dir, f"{k}.npz")) for k in kinds}
    return ({k: d[k]["oof"] for k in kinds}, {k: d[k]["val"] for k in kinds}, {k: d[k]["test"] for k in kinds})
