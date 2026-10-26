"""Writes the README "Results" section from results/kaggle_run/summary.json (between the
<!-- results:start --> and <!-- results:end --> markers).

    python tools/report/readme_results.py [summary.json] [README.md]
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gate(task, needle):
    for q in task["qc"]:
        if needle in q:
            m = re.search(r"delta=([+-][\d.]+)pp,? (?:95% )?CI \[([+-][\d.]+), ([+-][\d.]+)\]", q)
            if m:
                d, lo, hi = (float(x) for x in m.groups())
                return q.startswith("PASS"), d, lo, hi
    return None


def table(task):
    rows = sorted((r for r in task["rows"] if not r["model"].startswith("ADAR ablation")), key=lambda r: r["accuracy"])
    out = ["| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ | CPU ms / input |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        acc = f"{r['accuracy'] * 100:.2f}%" + (f" ± {r['accuracy_std'] * 100:.2f}" if r.get("accuracy_std") else "")
        cells = [r["model"], acc, f"{r['nll']:.3f}", f"{r['brier']:.3f}", f"{r['ece_ts']:.3f}", f"{r['cpu_ms']:.1f}"]
        if r["model"].startswith("ADAR"):
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def bullets(task):
    out = []
    for label, needle in (("Laya zero-shot", "best Laya zero-shot"), ("the best syllabus network", "best syllabus deep model"),
                          ("the best scikit-learn model", "best scikit-learn model"), ("stacking", "best stacking baseline")):
        g = gate(task, needle)
        if g:
            ok, d, lo, hi = g
            word = "significant" if ok else ("ADAR significantly behind" if hi < 0 else "not significant")
            out.append(f"- vs {label}: **{d:+.2f} points** (95% CI {lo:+.2f} to {hi:+.2f}, {word})")
    c = task.get("router", {}).get("chosen")
    if c:
        out.append(f"- Cascade: Laya called on **{c['escalation_rate']:.0%}** of inputs, {c['accuracy'] * 100:.2f}% accuracy at "
                   f"{c['mean_cost_ms']:.1f} ms vs {c['always_tier1_accuracy'] * 100:.2f}% at {c['always_tier1_cost_ms']:.1f} ms always calling it.")
    out.append(f"- Quality checks passed: **{task['qc_pass']}/{task['qc_total']}** ([report](results/kaggle_run/{{task}}/REPORT.md)).")
    return "\n".join(out)


def section(S):
    td = S["typed_decisions"]
    ft, base = td["convaiinnovations/laya-typed-decisions"], td["convaiinnovations/laya"]
    bc = base["adar_cal_" + base["adar_cal_selected"]]
    sp = S["splits"]
    parts = ["## Results", "",
             f"> One full run of [`notebooks/adar_kaggle.ipynb`](notebooks/adar_kaggle.ipynb) on Kaggle (GPU T4 ×2, "
             f"{S['total_minutes']:.0f} min) with the real Laya encoder. Every number below is read from "
             "[`results/kaggle_run/summary.json`](results/kaggle_run/summary.json), built only from the run's proof bundle. "
             "Probabilities are temperature-scaled on validation data; a claim counts only when its paired-bootstrap 95% CI excludes zero.",
             "", f"### AG News ({sp['ag_news']['test']:,} test articles, 4 topics)", "", table(S["tasks"]["ag_news"]), "",
             bullets(S["tasks"]["ag_news"]).replace("{task}", "ag_news"), "",
             "<table>\n<tr>\n<td width=\"50%\"><img src=\"results/kaggle_run/ag_news/accuracy_vs_latency.png\" alt=\"Accuracy versus CPU latency with the ADAR cascade curve\"></td>\n"
             "<td width=\"50%\"><img src=\"results/kaggle_run/ag_news/reliability.png\" alt=\"Reliability diagram after temperature scaling\"></td>\n</tr>\n<tr>\n"
             "<td align=\"center\"><sub>Accuracy for the cost, with the cascade curve.</sub></td>\n"
             "<td align=\"center\"><sub>Confidence vs. real accuracy (closer to the diagonal is more honest).</sub></td>\n</tr>\n</table>", "",
             f"### Emotion ({sp['emotion']['test']:,} test tweets, 6 emotions)", "", table(S["tasks"]["emotion"]), "",
             bullets(S["tasks"]["emotion"]).replace("{task}", "emotion"), "",
             "### typed-decisions (public System-One benchmark)", "",
             "| Model | Accuracy | Brier | KL | ECE |", "|---|---:|---:|---:|---:|",
             f"| Laya typed-decisions (board lists accuracy 0.766 only) | {ft['shipped']['accuracy']:.3f} | {ft['shipped']['brier_soft']:.3f} | {ft['shipped']['kl']:.3f} | {ft['shipped']['ece']:.3f} |",
             f"| Laya (base) | {base['shipped']['accuracy']:.3f} | {base['shipped']['brier_soft']:.3f} | {base['shipped']['kl']:.3f} | {base['shipped']['ece']:.3f} |",
             f"| Laya (base) + ADAR-Cal (temperatures fitted on train) | {bc['accuracy']:.3f} | {bc['brier_soft']:.3f} | {bc['kl']:.3f} | {bc['ece']:.3f} |",
             "", "Submission plan and the post itself: [`submissions/README.md`](submissions/README.md)."]
    return "\n".join(parts)


def main():
    summary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "kaggle_run", "summary.json")
    readme = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "README.md")
    S = json.load(open(summary))
    txt = open(readme).read()
    a, b = "<!-- results:start -->", "<!-- results:end -->"
    if a not in txt:
        sys.exit("README has no results markers")
    new = txt[:txt.index(a) + len(a)] + "\n" + section(S) + "\n" + txt[txt.index(b):]
    open(readme, "w").write(new)
    print("README results updated from", summary)


if __name__ == "__main__":
    main()
