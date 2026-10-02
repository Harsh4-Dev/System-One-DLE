"""Load a published ADAR head folder and classify raw text.

    from adar.inference import ADARClassifier
    clf = ADARClassifier.from_pretrained("publish/adar-head-ag_news", device="cuda")
    clf.predict(["Stocks rallied after the central bank held rates."])
"""
import json
import os

import torch

from .encoder import FrozenEncoder
from .experts import Vocab, build_expert
from .head import load_head


class ADARClassifier:
    def __init__(self, head, options, cfg, encoder, experts, vocab, device):
        self.head, self.options, self.cfg = head, options, cfg
        self.encoder, self.experts, self.vocab, self.device = encoder, experts, vocab, device
        self.labels = cfg["labels"]

    @classmethod
    def from_pretrained(cls, path, device=None, encoder_device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        head, options, cfg = load_head(path, device)
        e = cfg["encoder"]
        encoder = FrozenEncoder(e["backend"], e["model_id"], encoder_device or device, e["max_len"], e.get("subfolder"))
        with open(os.path.join(path, "experts", "vocab.json")) as f:
            vocab = Vocab(json.load(f))
        experts = {}
        for kind in cfg["experts"]["kinds"]:
            models = []
            for fold in range(cfg["experts"]["folds"]):
                m = build_expert(kind, len(vocab), len(cfg["labels"]))
                m.load_state_dict(torch.load(os.path.join(path, "experts", f"{kind}_fold{fold}.pt"), map_location="cpu"))
                models.append(m.to(device).eval())
            experts[kind] = models
        return cls(head, options, cfg, encoder, experts, vocab, device)

    @torch.no_grad()
    def expert_probs(self, texts):
        ids, lens = self.vocab.encode(texts, self.cfg["experts"]["max_len"])
        ids, lens = torch.as_tensor(ids, device=self.device), torch.as_tensor(lens, device=self.device)
        out = []
        for kind in self.cfg["experts"]["kinds"]:
            ps = [torch.softmax(m(ids, lens).float(), -1) for m in self.experts[kind]]
            out.append(torch.stack(ps).mean(0))
        return torch.stack(out, 1)  # B,E,K

    @torch.no_grad()
    def predict(self, texts, escalate_below=0.5):
        z = torch.as_tensor(self.encoder.embed(texts), device=self.device)
        p_exp = self.expert_probs(texts) if self.head.use_experts else None
        out = self.head(z, torch.as_tensor(self.options, device=self.device), p_exp)
        probs = torch.softmax(out["logits"].float() / self.head.temperature, -1).cpu().numpy()
        act = torch.sigmoid(out["act_logit"]).cpu().numpy()
        res = []
        for i in range(len(texts)):
            r = {"label": self.labels[int(probs[i].argmax())],
                 "probabilities": {l: round(float(p), 4) for l, p in zip(self.labels, probs[i])},
                 "p_head_correct": round(float(act[i]), 4),
                 "escalate": bool(act[i] < escalate_below)}
            if out["expert_logit"] is not None:
                kinds = self.cfg["experts"]["kinds"]
                r["p_expert_correct"] = {k: round(float(v), 4)
                                         for k, v in zip(kinds, torch.sigmoid(out["expert_logit"][i]).cpu().numpy())}
                r["routing_weights"] = {k: round(float(v), 4) for k, v in zip(kinds, out["alpha"][i].cpu().numpy())}
            res.append(r)
        return res
