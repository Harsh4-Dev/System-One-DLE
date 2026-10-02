"""The ADAR System-One head.

One decision = one short token sequence, in the spirit of Laya's typed decision head:

    [STATE]  the frozen System-One (Laya) embedding of the input
    [OPT_k]  one token per answer option, built from the encoder embedding of the option's
             *description* plus every expert's probability for that option
    [EXP_e]  one token per syllabus expert (TextCNN / RNN / LSTM / GRU) carrying how sure it is

A 2-layer transformer lets the tokens attend to each other (Unit III: attention). Outputs:

    choice   logits_k = scorer(OPT_k) + bilinear(STATE, OPT_k) + sum_e alpha_e(x) * log p_e,k
             (alpha = per-input expert routing weights: the "adaptive routing" in ADAR)
    noul     P(expert e is right) for every expert          -> which architecture to trust
    act      P(the head's own answer is right)              -> escalate / abstain gate

Training minimises strictly proper scoring rules (log score + Brier), the objective Laya's RLCD
rewards, then a single temperature is fitted on validation data.
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import fit_temperature
from .utils import LOG, count_params, save_json, seed_everything, torch_save


def expert_summary(p):
    """Per-expert features: max prob, top1-top2 margin, normalised entropy, agreement with others."""
    B, E, K = p.shape
    top2 = p.topk(min(2, K), -1).values
    mx = top2[..., 0]
    margin = top2[..., 0] - (top2[..., 1] if K > 1 else 0)
    ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(K)
    am = p.argmax(-1)
    agree = (am[:, :, None] == am[:, None, :]).float().sum(-1).sub(1) / max(E - 1, 1)
    return torch.stack([mx, margin, ent, agree], -1)


class ADARHead(nn.Module):
    def __init__(self, state_dim, opt_dim, n_experts, d=256, layers=2, heads=4, dropout=0.1,
                 use_state=True, use_experts=True, use_transformer=True):
        super().__init__()
        self.cfg = dict(state_dim=state_dim, opt_dim=opt_dim, n_experts=n_experts, d=d, layers=layers, heads=heads,
                        dropout=dropout, use_state=use_state, use_experts=use_experts, use_transformer=use_transformer)
        self.use_state, self.use_experts, self.use_transformer = use_state, use_experts and n_experts > 0, use_transformer
        E = max(n_experts, 1)
        self.state_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Dropout(dropout), nn.Linear(state_dim, d))
        self.state_null = nn.Parameter(torch.zeros(d))
        self.opt_proj = nn.Sequential(nn.LayerNorm(opt_dim), nn.Linear(opt_dim, d))
        self.opt_evidence = nn.Linear(2 * E, d)
        self.exp_feat = nn.Linear(4, d)
        self.exp_id = nn.Embedding(E, d)
        self.type_emb = nn.Embedding(3, d)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
            self.mixer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.scorer = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.bil_s = nn.Linear(d, d, bias=False)
        self.bil_o = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, E)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(math.e - 1))  # softplus(bias) = 1 -> plain log-linear pool at init
        self.expert_noul = nn.Linear(d, 1)
        self.act_head = nn.Sequential(nn.Linear(d + 3, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("temperature", torch.ones(()))

    def forward(self, z, opts, p_exp=None, exp_mask=None):
        """z: (B, Ds) state embeddings; opts: (K, Do) option embeddings; p_exp: (B, E, K) expert probs;
        exp_mask: (B, E) 1 = expert available (used for expert-dropout and missing experts)."""
        B, K = z.size(0), opts.size(0)
        s0 = self.state_proj(z) if self.use_state else self.state_null.expand(B, -1)
        o0 = self.opt_proj(opts).unsqueeze(0).expand(B, -1, -1)  # B,K,d
        tokens = [s0.unsqueeze(1) + self.type_emb.weight[0]]
        o = o0 + self.type_emb.weight[1]
        if self.use_experts:
            E = p_exp.size(1)
            m = torch.ones(B, E, device=z.device) if exp_mask is None else exp_mask.float()
            p_eff = p_exp * m[..., None] + (1 - m[..., None]) / K  # missing expert -> uniform evidence
            logp = p_eff.clamp_min(1e-6).log()
            ev = torch.cat([logp, p_eff], 1).transpose(1, 2)  # B,K,2E
            o = o + self.opt_evidence(ev)
            ex = self.exp_feat(expert_summary(p_eff) * m[..., None]) + self.exp_id.weight[:E] + self.type_emb.weight[2]
            tokens += [o, ex]
        else:
            tokens += [o]
        h = torch.cat(tokens, 1)
        if self.use_transformer:
            h = self.mixer(h)
        h = self.norm(h)
        h_state, h_opt = h[:, 0], h[:, 1:1 + K]
        logits = self.scorer(h_opt).squeeze(-1)
        logits = logits + (self.bil_s(s0).unsqueeze(1) * self.bil_o(o0)).sum(-1) / math.sqrt(s0.size(-1))
        alpha = None
        expert_logit = None
        if self.use_experts:
            alpha = F.softplus(self.gate(h_state))[:, :E] * m
            logits = logits + (alpha.unsqueeze(-1) * logp).sum(1)
            expert_logit = self.expert_noul(h[:, 1 + K:1 + K + E]).squeeze(-1)
        p = torch.softmax(logits.detach().float(), -1)
        top2 = p.topk(min(2, K), -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, -1],
                             -(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(K)], -1)
        act_logit = self.act_head(torch.cat([h_state, feats], -1)).squeeze(-1)
        return {"logits": logits, "alpha": alpha, "expert_logit": expert_logit, "act_logit": act_logit}


def head_loss(out, y, p_exp, w_brier=1.0, w_noul=0.2, w_act=0.1):
    logits = out["logits"]
    loss = F.cross_entropy(logits, y)
    p = torch.softmax(logits, -1)
    loss = loss + w_brier * ((p - F.one_hot(y, p.size(-1)).float()) ** 2).sum(-1).mean()
    if out["expert_logit"] is not None:
        right = (p_exp.argmax(-1) == y[:, None]).float()
        loss = loss + w_noul * F.binary_cross_entropy_with_logits(out["expert_logit"], right)
    head_right = (logits.argmax(-1) == y).float().detach()
    loss = loss + w_act * F.binary_cross_entropy_with_logits(out["act_logit"], head_right)
    return loss


@torch.no_grad()
def predict(model, z, opts, p_exp, device, batch_size=2048, calibrated=True):
    model.eval()
    o = torch.as_tensor(opts, device=device)
    outs = {"probs": [], "expert_p": [], "act_p": [], "alpha": []}
    for i in range(0, len(z), batch_size):
        zb = torch.as_tensor(z[i:i + batch_size], device=device)
        pb = torch.as_tensor(p_exp[i:i + batch_size], device=device) if p_exp is not None else None
        out = model(zb, o, pb)
        t = model.temperature if calibrated else 1.0
        outs["probs"].append(torch.softmax(out["logits"].float() / t, -1).cpu().numpy())
        outs["act_p"].append(torch.sigmoid(out["act_logit"]).cpu().numpy())
        if out["expert_logit"] is not None:
            outs["expert_p"].append(torch.sigmoid(out["expert_logit"]).cpu().numpy())
            outs["alpha"].append(out["alpha"].cpu().numpy())
    return {k: (np.concatenate(v) if v else None) for k, v in outs.items()}


def train_head(z_tr, p_tr, y_tr, z_va, p_va, y_va, opts, ckpt_path, device, variant=None, seed=0,
               epochs=30, batch_size=512, lr=1e-3, wd=0.01, patience=5, expert_dropout=0.15, d=256):
    """Train one ADAR head (resumable). Returns (model, history). p_* may be None for no-expert variants."""
    variant = dict(variant or {})
    seed_everything(seed)
    n_exp = p_tr.shape[1] if p_tr is not None else 0
    model = ADARHead(z_tr.shape[1], opts.shape[1], n_exp, d=d, **variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, math.ceil(len(y_tr) / batch_size))
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (0.05 * total)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    start, best, best_state, bad, hist = 0, float("inf"), None, 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, best_state, bad, hist = ck["epoch"], ck["best"], ck["best_state"], ck["bad"], ck["hist"]
        if ck.get("finished"):
            model.load_state_dict(best_state)
            return model, hist
        LOG.info("  resume head from epoch %d", start)
    Z = torch.as_tensor(z_tr, device=device)
    P = torch.as_tensor(p_tr, device=device) if n_exp else None
    Y = torch.as_tensor(y_tr, device=device)
    O = torch.as_tensor(opts, device=device)
    for ep in range(start, epochs):
        model.train()
        perm = torch.randperm(len(Y), device=device)
        tot, nb = 0.0, 0
        for i in range(0, len(Y), batch_size):
            b = perm[i:i + batch_size]
            pb = P[b] if n_exp else None
            mask = None
            if n_exp and expert_dropout > 0:
                mask = (torch.rand(len(b), n_exp, device=device) > expert_dropout).float()
            out = model(Z[b], O, pb, mask)
            loss = head_loss(out, Y[b], pb if n_exp else None)
            if not torch.isfinite(loss):
                LOG.warning("  non-finite head loss; skipping batch")
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
        pv = predict(model, z_va, opts, p_va, device, calibrated=False)["probs"]
        v_nll = float(-np.mean(np.log(np.clip(pv[np.arange(len(y_va)), y_va], 1e-12, 1))))
        v_acc = float((pv.argmax(1) == y_va).mean())
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), "val_nll": v_nll, "val_acc": v_acc})
        if v_nll < best - 1e-5:
            best, bad = v_nll, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        stop = bad >= patience or ep + 1 == epochs
        torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "best_state": best_state, "bad": bad, "hist": hist,
                    "finished": stop}, ckpt_path)
        if stop:
            break
    LOG.info("  head %s seed %d: best val_nll=%.4f val_acc=%.4f epochs=%d", variant or "full", seed, best,
             max(h["val_acc"] for h in hist), len(hist))
    model.load_state_dict(best_state)
    # temperature scaling on validation logits
    logits = np.log(np.clip(predict(model, z_va, opts, p_va, device, calibrated=False)["probs"], 1e-12, 1))
    model.temperature.fill_(fit_temperature(logits, y_va))
    return model, hist


def head_cpu_latency_ms(model, dim, opts, n_exp, n=50):
    m = ADARHead(**model.cfg)
    m.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    m.eval()
    o = torch.as_tensor(opts)
    z = torch.randn(1, dim)
    pe = torch.softmax(torch.randn(1, n_exp, opts.shape[0]), -1) if n_exp else None
    ts = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            m(z, o, pe)
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts[5:]))


def save_head(model, out_dir, extra_config, option_emb):
    """Write a publishable head: model.safetensors + options.safetensors + config.json."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    save_file({"option_embeddings": torch.as_tensor(option_emb).contiguous()},
              os.path.join(out_dir, "options.safetensors"))
    cfg = {"architecture": "ADARHead", "head_config": model.cfg, "temperature": float(model.temperature),
           "params": count_params(model), **extra_config}
    save_json(os.path.join(out_dir, "config.json"), cfg)
    return cfg


def load_head(out_dir, device="cpu"):
    from safetensors.torch import load_file
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    model = ADARHead(**cfg["head_config"])
    model.load_state_dict(load_file(os.path.join(out_dir, "model.safetensors")))
    opts = load_file(os.path.join(out_dir, "options.safetensors"))["option_embeddings"].numpy()
    return model.to(device).eval(), opts, cfg
