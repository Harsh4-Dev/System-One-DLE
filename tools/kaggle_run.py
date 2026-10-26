"""Run the ADAR notebook on Kaggle (GPU T4 x2) from the command line and collect the results.

Needs the Kaggle CLI (`pip install kaggle`) and an API token in the environment
(KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY). See docs/KAGGLE_CLI_SETUP.md.

    python tools/kaggle_run.py --user <kaggle-username>            # push, wait, download, collect
    python tools/kaggle_run.py --user <kaggle-username> --no-push  # only wait for / fetch the last version
    python tools/kaggle_run.py --user <kaggle-username> --quick    # ~30-45 min check on subsets

After it finishes: results/kaggle_run/summary.json holds every number, and the submission files and
the Word report are regenerated from it.
"""
import argparse
import datetime
import json
import os

import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if out.strip():
        print(out.strip()[-3000:], flush=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {' '.join(cmd)}")
    return out


def build_kernel(user, slug, title, quick, folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    with open(os.path.join(ROOT, "notebooks", "adar_kaggle.ipynb")) as f:
        nb = json.load(f)
    if quick:  # the config cell reads ADAR_PROFILE; set it at the top of the notebook
        nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": ["import os\n", "os.environ['ADAR_PROFILE'] = 'quick'\n"]})
    with open(os.path.join(folder, "adar_kaggle.ipynb"), "w") as f:
        json.dump(nb, f, indent=1)
    meta = {"id": f"{user}/{slug}", "title": title, "code_file": "adar_kaggle.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def wait(ref, poll, max_hours):
    t0 = time.time()
    while True:
        out = sh(["kaggle", "kernels", "status", ref], check=False).lower()
        if "complete" in out:
            return "complete"
        if any(w in out for w in ("error", "cancel")):
            return "failed"
        if time.time() - t0 > max_hours * 3600:
            return "timeout"
        print(f"  running for {int((time.time() - t0) / 60)} min; next check in {poll}s", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), required=not os.environ.get("KAGGLE_USERNAME"))
    ap.add_argument("--slug", default="adar-system-one")
    ap.add_argument("--title", default="ADAR System-One")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--deck", help="Slides deck project/ folder to rebuild from the fresh summary (tools/report/build_slides.py)")
    a = ap.parse_args()
    ref = f"{a.user}/{a.slug}"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(ROOT, "runs", stamp)
    os.makedirs(run_dir, exist_ok=True)
    if not a.no_push:
        folder = os.path.join(ROOT, "build", "kaggle_kernel")
        build_kernel(a.user, a.slug, a.title, a.quick, folder)
        sh(["kaggle", "kernels", "push", "-p", folder, "--accelerator", "NvidiaTeslaT4"])
        time.sleep(30)
    status = wait(ref, a.poll, a.max_hours)
    print("kernel status:", status)
    sh(["kaggle", "kernels", "output", ref, "-p", run_dir, "-o", "--file-pattern",
        r"(adar_results_small\.zip|.*\.log|SUMMARY\.md)$"], check=False)
    proof = os.path.join(run_dir, "adar_results_small.zip")
    if not os.path.exists(proof):
        cand = [os.path.join(dp, f) for dp, _, fs in os.walk(run_dir) for f in fs if f == "adar_results_small.zip"]
        proof = cand[0] if cand else None
    if not proof:
        sys.exit(f"no adar_results_small.zip in the kernel output (status {status}); see the log in {run_dir}")
    sh([sys.executable, os.path.join(ROOT, "tools", "collect_results.py"), proof])
    sh([sys.executable, os.path.join(ROOT, "tools", "make_submissions.py")])
    figs = os.path.join(ROOT, "build", "figs")
    os.makedirs(figs, exist_ok=True)
    summary = os.path.join(ROOT, "results", "kaggle_run", "summary.json")
    sh([sys.executable, os.path.join(ROOT, "tools", "report", "make_figures.py"), summary, figs], check=False)
    logs = [os.path.join(dp, f) for dp, _, fs in os.walk(run_dir) for f in fs if f.endswith(".log")]
    kernel_log = os.path.join(ROOT, "results", "kaggle_run", "kernel.log")
    if logs:
        shutil.copy2(max(logs, key=os.path.getsize), kernel_log)
    sh(["node", os.path.join(ROOT, "tools", "report", "build_report.js"), summary, figs,
        os.path.join(ROOT, "docs", "ADAR_Project_Report.docx")] + ([kernel_log] if logs else []), check=False)
    if a.deck:
        sh([sys.executable, os.path.join(ROOT, "tools", "report", "build_slides.py"), summary, a.deck]
           + (["--log", kernel_log] if logs else []), check=False)
    print("\nDone. Every number now comes from", os.path.relpath(summary, ROOT), "| raw files in", os.path.relpath(run_dir, ROOT))
    if status != "complete":
        print("WARNING: the kernel did not finish cleanly; check results/kaggle_run/state.json for failed stages")


if __name__ == "__main__":
    main()
