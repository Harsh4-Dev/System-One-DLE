"""Resumable stage manager.

Every stage writes a `done` record into `<run_dir>/state.json` when it finishes, so re-running the
notebook (or a fresh Kaggle session restored from a previous version's output) skips finished
work. A failing stage records its traceback and, with `continue_on_error`, lets independent
stages keep running; stages that depend on it are skipped instead of crashing later.
"""
import glob
import os
import time
import traceback

from .utils import LOG, copytree_merge, load_json, save_json


class StageFailed(RuntimeError):
    pass


class RunState:
    def __init__(self, run_dir, continue_on_error=True, time_budget_h=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "state.json")
        self.state = load_json(self.path, {"stages": {}})
        self.continue_on_error = continue_on_error
        self.t_start = time.time()
        self.time_budget_s = None if time_budget_h is None else time_budget_h * 3600

    # ---- restore ---------------------------------------------------------------------------
    @staticmethod
    def restore_from_inputs(run_dir, search_glob="/kaggle/input/*/**/adar_run"):
        """Copy a previous run (attached as a Kaggle input dataset / notebook output) into run_dir.

        Kaggle wipes /kaggle/working between sessions. Attach the previous version's output as an
        input and this picks up all finished stages, cached embeddings and checkpoints.
        """
        candidates = [p for p in glob.glob(search_glob, recursive=True)
                      if os.path.isfile(os.path.join(p, "state.json")) and os.path.abspath(p) != os.path.abspath(run_dir)]
        if not candidates:
            return None
        src = max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "state.json")))
        LOG.info("Restoring previous run from %s", src)
        copytree_merge(src, run_dir)
        return src

    # ---- bookkeeping -----------------------------------------------------------------------
    def _save(self):
        save_json(self.path, self.state)

    def status(self, name):
        return self.state["stages"].get(name, {}).get("status")

    def done(self, name):
        return self.status(name) == "done"

    def out_of_time(self, margin_s=900):
        if self.time_budget_s is None:
            return False
        return time.time() - self.t_start > self.time_budget_s - margin_s

    def reset(self, name):
        self.state["stages"].pop(name, None)
        self._save()

    def run(self, stage, fn, deps=(), force=False, **kwargs):
        """Run fn(**kwargs) as stage `stage` unless already done. Returns fn's result or None."""
        if self.done(stage) and not force:
            LOG.info("[skip] %s already done", stage)
            return self.state["stages"][stage].get("result")
        missing = [d for d in deps if not self.done(d)]
        if missing:
            msg = f"stage {stage} skipped: dependencies not done: {missing}"
            LOG.warning(msg)
            self.state["stages"][stage] = {"status": "blocked", "reason": msg}
            self._save()
            if not self.continue_on_error:
                raise StageFailed(msg)
            return None
        if self.out_of_time():
            LOG.warning("time budget nearly used; not starting %s (re-run to resume)", stage)
            return None
        LOG.info("[run ] %s", stage)
        t0 = time.time()
        self.state["stages"][stage] = {"status": "running", "started": t0}
        self._save()
        try:
            result = fn(**kwargs)
        except KeyboardInterrupt:
            self.state["stages"][stage] = {"status": "interrupted"}
            self._save()
            raise
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            LOG.error("stage %s FAILED: %s\n%s", stage, e, tb)
            self.state["stages"][stage] = {"status": "failed", "error": repr(e), "traceback": tb[-4000:]}
            self._save()
            with open(os.path.join(self.run_dir, "errors.log"), "a") as f:
                f.write(f"\n==== {stage} @ {time.ctime()} ====\n{tb}\n")
            if not self.continue_on_error:
                raise
            return None
        rec = {"status": "done", "seconds": round(time.time() - t0, 1)}
        if isinstance(result, (dict, list, str, int, float)) or result is None:
            rec["result"] = result
        self.state["stages"][stage] = rec
        self._save()
        LOG.info("[done] %s (%.1fs)", stage, rec["seconds"])
        return result

    def summary(self):
        rows = []
        for k, v in self.state["stages"].items():
            rows.append(f"{v.get('status', '?'):>11}  {k}  {v.get('seconds', '')}")
        return "\n".join(rows)
