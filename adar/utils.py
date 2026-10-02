import functools
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager

import numpy as np

LOG = logging.getLogger("adar")


def setup_logging(log_path=None, level=logging.INFO):
    LOG.setLevel(level)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    LOG.propagate = False
    return LOG


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)  # mkstemp creates 0600 files; outputs are meant to be shared
    os.replace(tmp, path)


def save_json(path, obj):
    atomic_write_bytes(path, json.dumps(obj, indent=2, default=_json_default).encode())


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".npz")
    os.close(fd)
    np.savez(tmp, **arrays)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def torch_save(obj, path):
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".pt")
    os.close(fd)
    torch.save(obj, tmp)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def retry(times=4, base_delay=2.0, exceptions=(Exception,)):
    """Retry with exponential back-off (2s, 4s, 8s, ...) - for flaky downloads."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            for attempt in range(times + 1):
                try:
                    return fn(*a, **kw)
                except exceptions as e:  # noqa: PERF203
                    if attempt == times:
                        raise
                    delay = base_delay * (2 ** attempt)
                    LOG.warning("%s failed (%s: %s); retry %d/%d in %.0fs",
                                fn.__name__, type(e).__name__, e, attempt + 1, times, delay)
                    time.sleep(delay)
        return wrapped
    return deco


def is_oom(err):
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or type(err).__name__ == "OutOfMemoryError"


def run_with_oom_backoff(fn, batch_size, min_batch=1, what="step"):
    """Call fn(batch_size); on CUDA OOM free memory, halve the batch and try again."""
    import torch
    bs = batch_size
    while True:
        try:
            return fn(bs), bs
        except RuntimeError as e:
            if not is_oom(e) or bs <= min_batch:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs = max(min_batch, bs // 2)
            LOG.warning("OOM during %s; retrying with batch_size=%d", what, bs)


@contextmanager
def timer(name, sink=None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    LOG.info("%s took %.1fs", name, dt)
    if sink is not None:
        sink[name] = dt


def copytree_merge(src, dst):
    """Copy src into dst, keeping files that already exist in dst."""
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            target = os.path.join(out, f)
            if not os.path.exists(target):
                shutil.copy2(os.path.join(root, f), target)


def device_list():
    import torch
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return [torch.device("cpu")]


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))
