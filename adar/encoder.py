"""Frozen System-One encoder with sharded, resumable, multi-GPU embedding cache.

backend="laya": the encoder inside a Laya checkpoint (ModernBERT-large fine-tuned by RLCD),
                loaded with `laya.load`, exactly as `laya.shortlist.embed_fn_from_agent` uses it.
backend="hf"  : any Hugging Face encoder (used as a stand-in for local smoke tests).

Pooled representation = [mean over tokens ; first token]  ->  2 * hidden_size dims.
"""
import os
import queue
import threading
import time

import numpy as np
import torch

from .utils import LOG, is_oom, load_json, save_json


class FrozenEncoder:
    def __init__(self, backend, model_id, device, max_len=128, subfolder=None):
        self.backend, self.model_id, self.device, self.max_len = backend, model_id, torch.device(device), max_len
        if backend == "laya":
            import laya
            kw = {"subfolder": subfolder} if subfolder else {}
            self.agent = laya.load(model_id, device=str(self.device), **kw)
            if self.agent.device != self.device:  # laya falls back to CPU when the GPU is out of memory
                LOG.warning("laya loaded on %s instead of %s", self.agent.device, self.device)
                self.device = self.agent.device
            self.tok = self.agent.tok
            self.model = self.agent.model.encoder
        elif backend == "hf":
            from transformers import AutoModel, AutoTokenizer
            self.agent = None
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
        else:
            raise ValueError(backend)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden = int(self.model.config.hidden_size)
        self.dim = 2 * self.hidden

    def _tokenize(self, texts):
        return self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

    @torch.inference_mode()
    def _embed_batch(self, texts):
        enc = self._tokenize(texts)
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            h = self.model(input_ids=ids, attention_mask=am).last_hidden_state
        h = h.float()
        m = am.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return torch.cat([mean, h[:, 0]], -1).cpu().numpy()

    def embed(self, texts, batch_size=64):
        """Embed texts (length-sorted batches to minimise padding; output in input order)."""
        texts = ["" if t is None else str(t) for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        i, bs = 0, batch_size
        while i < len(texts):
            idx = order[i:i + bs]
            try:
                out[idx] = self._embed_batch([texts[j] for j in idx])
                i += len(idx)
            except RuntimeError as e:
                if not is_oom(e) or bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                LOG.warning("encoder OOM on %s; batch_size -> %d", self.device, bs)
        return out

    def cpu_latency_ms(self, texts, n=20, threads=4, question=None):
        """Median batch-1 latency on CPU (loads a CPU copy for the measurement).

        Returns {"encoder": ms, "zero_shot": ms or None}; zero_shot times a full Laya typed
        decision (`agent.system_one`) when `question` is given and the backend is Laya."""
        old = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            cpu = FrozenEncoder(self.backend, self.model_id, "cpu", self.max_len) if self.device.type != "cpu" else self

            def med(fn):
                times = []
                for t in texts[:n]:
                    t0 = time.perf_counter()
                    fn(t)
                    times.append((time.perf_counter() - t0) * 1000)
                return float(np.median(times[2:] if len(times) > 5 else times))

            out = {"encoder": med(lambda t: cpu._embed_batch([t])), "zero_shot": None}
            if question is not None and cpu.agent is not None:
                out["zero_shot"] = med(lambda t: cpu.agent.system_one(t, question))
            if cpu is not self:
                del cpu
            return out
        finally:
            torch.set_num_threads(old)


def _shard_path(cache_dir, split, i):
    return os.path.join(cache_dir, split, f"shard_{i:05d}.npy")


def cached_embed(encoders, texts, cache_dir, split, shard_size=4096, batch_size=64, meta=None):
    """Embed `texts` with resumable shards; work is spread over every encoder (one per GPU)."""
    os.makedirs(os.path.join(cache_dir, split), exist_ok=True)
    man_path = os.path.join(cache_dir, split, "manifest.json")
    man = load_json(man_path)
    want = {"n": len(texts), "shard_size": shard_size, **(meta or {})}
    if man is not None and man.get("spec") != want:
        raise RuntimeError(f"embedding cache {cache_dir}/{split} was built for different data/model: "
                           f"{man.get('spec')} != {want}. Delete it or change run_dir.")
    save_json(man_path, {"spec": want})
    n_shards = (len(texts) + shard_size - 1) // shard_size
    todo = [i for i in range(n_shards) if not os.path.exists(_shard_path(cache_dir, split, i))]
    if todo:
        LOG.info("embedding %s: %d/%d shards to do on %d device(s)", split, len(todo), n_shards, len(encoders))
        q = queue.Queue()
        for i in todo:
            q.put(i)
        errors = []
        t0 = time.time()

        def worker(enc):
            while not errors:
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    emb = enc.embed(texts[i * shard_size:(i + 1) * shard_size], batch_size=batch_size)
                    p = _shard_path(cache_dir, split, i)
                    np.save(p + ".tmp.npy", emb.astype(np.float16))
                    os.replace(p + ".tmp.npy", p)
                    done = n_shards - q.qsize()
                    LOG.info("  %s shard %d done on %s (%d/%d, %.0fs)", split, i, enc.device, done, n_shards,
                             time.time() - t0)
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                    return

        threads = [threading.Thread(target=worker, args=(e,)) for e in encoders]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
    return np.concatenate([np.load(_shard_path(cache_dir, split, i)).astype(np.float32) for i in range(n_shards)])
