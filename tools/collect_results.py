"""Turn a run's proof bundle (adar_results_small.zip, or its extracted folder) into
results/kaggle_run/summary.json, the single source every deck, report and submission number is read from.

    python tools/collect_results.py runs/2026-10-26/adar_results_small.zip
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _row(r):
    out = {"model": r["model"], "family": r.get("family"), "unit": r.get("unit"), "accuracy": r["accuracy"],
           "macro_f1": r.get("macro_f1"), "nll": r.get("nll"), "brier": r.get("brier"), "ece_ts": r.get("ece"),
           "ece_raw": r.get("ece_raw"), "aurc": r.get("aurc"), "cpu_ms": r.get("latency_ms"),
           "params_M": (r["params"] / 1e6) if isinstance(r.get("params"), (int, float)) and r.get("params") else None,
           "key": r.get("key")}
    for k in ("accuracy", "nll", "brier", "macro_f1", "ece"):
        if r.get(k + "_std") is not None:
            out[k + "_std"] = r[k + "_std"]
    return out


def collect(src, out_dir):
    tmp = None
    if zipfile.is_zipfile(src):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(src) as z:
            for name in z.namelist():  # refuse absolute paths / traversal
                if name.startswith("/") or ".." in name.split("/"):
                    raise ValueError(f"unsafe path in bundle: {name}")
            z.extractall(tmp)
        rd = tmp
    else:
        rd = src
    cfg = _load(os.path.join(rd, "config.json"), {})
    state = _load(os.path.join(rd, "state.json"), {"stages": {}})
    stages = state.get("stages", {})
    S = {
        "source": "fresh Kaggle run (proof bundle adar_results_small.zip)",
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "encoder": cfg.get("encoder", {}),
        "standin_encoder": cfg.get("encoder", {}).get("backend") != "laya",
        "stage_status": {k: v.get("status") for k, v in stages.items()},
        "stage_seconds": {k: v.get("seconds") for k, v in stages.items() if v.get("seconds") is not None},
        "tasks": {}, "splits": {}, "typed_decisions": {},
    }
    if S["stage_status"].get("bundle") == "running":
        # the bundle stage writes this zip while it is still running
        S["stage_status"]["bundle"] = "done"
    S["total_minutes"] = round(sum(S["stage_seconds"].values()) / 60, 1)
    bundle = stages.get("bundle", {}).get("result") or {}
    S["hf_uploads"] = bundle.get("pushed", [])
    for fp in glob.glob(os.path.join(rd, "data", "*_fingerprint.json")):
        m = re.search(r"(\w+): train=(\d+) val=(\d+) test=(\d+) classes=(\d+)", _load(fp)["summary"])
        if m:
            S["splits"][m.group(1)] = dict(zip(("train", "val", "test", "classes"), map(int, m.groups()[1:])))
    for task_dir in sorted(glob.glob(os.path.join(rd, "results", "*"))):
        task = os.path.basename(task_dir)
        comp = _load(os.path.join(task_dir, "comparison.json"))
        if comp is None:
            continue
        qc = _load(os.path.join(task_dir, "qc.json"), [])
        t = {"rows": [_row(r) for r in comp],
             "qc": [f"{'PASS' if c['pass'] else 'FAIL'} — {c['check']}: {c['detail']}" for c in qc],
             "routing": _load(os.path.join(task_dir, "routing.json")),
             "router": _load(os.path.join(rd, "router", task, "router.json"))}
        t["qc_pass"], t["qc_total"] = sum(c["pass"] for c in qc), len(qc)
        S["tasks"][task] = t
        dst = os.path.join(out_dir, task)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(task_dir, "*")):
            shutil.copy2(f, dst)
    for mf in glob.glob(os.path.join(rd, "typed_decisions", "*", "metrics.json")):
        m = _load(mf)
        cal = m["adar_cal"]
        sel = cal["selected_on_train"]
        S["typed_decisions"][m["model"]] = {
            "shipped": m["shipped"]["test"], "shipped_train": m["shipped"]["train"],
            "adar_cal_" + sel: cal[sel]["test"], "adar_cal_selected": sel,
            "adar_cal_variants": {k: {"test": v["test"], "train": v["train"], "temperatures": v["temperatures"]}
                                  for k, v in cal.items() if k != "selected_on_train"},
            "latency_ms_p50": m.get("latency_ms_p50"), "latency_ms_p90": m.get("latency_ms_p90"),
        }
        dst = os.path.join(out_dir, "typed_decisions", os.path.basename(os.path.dirname(mf)))
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(mf, dst)
    for f in ("SUMMARY.md", "run.log", "state.json", "config.json"):
        if os.path.exists(os.path.join(rd, f)):
            shutil.copy2(os.path.join(rd, f), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="adar_results_small.zip or its extracted folder")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "kaggle_run"))
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out, "summary.json")):
        old = os.path.join(os.path.dirname(os.path.abspath(a.out)), "previous_runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        shutil.copytree(a.out, old)
        print("previous results archived to", old)
        shutil.rmtree(a.out)
    S = collect(a.bundle, a.out)
    bad = {k: v for k, v in S["stage_status"].items() if v != "done"}
    print("stages:", len(S["stage_status"]), "not done:", bad or "none")
    for t, v in S["tasks"].items():
        adar = next((r for r in v["rows"] if r["model"].startswith("ADAR (ours")), None)
        print(f"{t}: QC {v['qc_pass']}/{v['qc_total']}  ADAR accuracy {adar and adar['accuracy']:.4f}")
    for m, v in S["typed_decisions"].items():
        print(f"typed-decisions {m}: shipped acc {v['shipped']['accuracy']:.4f} ece {v['shipped']['ece']:.4f}")
    if S["standin_encoder"]:
        print("WARNING: this run used a stand-in encoder, not Laya; do not report it")


if __name__ == "__main__":
    main()
