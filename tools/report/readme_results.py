"""Writes the number-bearing sections of README.md and docs/MAIN.md from results/kaggle_run/summary.json
(between <!-- name:start --> and <!-- name:end --> markers), so no result is typed by hand.

    python tools/report/readme_results.py [summary.json] [README.md] [MAIN.md]
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
    c = (task.get("router") or {}).get("chosen")
    for r in rows:
        if c and r["model"].startswith("ADAR cascade"):
            r = dict(r, model=f"ADAR cascade (Laya on {esc(c['escalation_rate'])} of inputs)")
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
            out.append(f"* vs {label}: **{d:+.2f} points** (95% CI {lo:+.2f} to {hi:+.2f}, {word})")
    c = task.get("router", {}).get("chosen")
    if c:
        out.append(f"* Cascade: Laya called on **{esc(c['escalation_rate'])}** of inputs, {c['accuracy'] * 100:.2f}% accuracy at "
                   f"{c['mean_cost_ms']:.1f} ms vs {c['always_tier1_accuracy'] * 100:.2f}% at {c['always_tier1_cost_ms']:.1f} ms always calling it.")
    out.append(f"* Quality checks passed: **{task['qc_pass']}/{task['qc_total']}** ([report](results/kaggle_run/{{task}}/REPORT.md)).")
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


def row(task, prefix):
    return next((r for r in task["rows"] if r["model"].startswith(prefix)), None)


def best_expert(task):
    return max((r for r in task["rows"] if r["model"] in ("TextCNN", "Bi-RNN", "Bi-LSTM", "Bi-GRU")), key=lambda r: r["accuracy"])


def verdict(task):
    win, tie, lose = [], [], []
    for name, needle in (("Laya zero-shot", "best Laya zero-shot"), ("the best syllabus network", "best syllabus deep model"),
                         ("the best scikit-learn model", "best scikit-learn model")):
        g = gate(task, needle)
        if g:
            (win if g[0] else lose if g[3] < 0 else tie).append(name)
    j = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    parts = (["significantly beats " + j(win)] if win else []) + (["ties " + j(tie) + " (difference not significant)"] if tie else []) \
        + (["is significantly behind " + j(lose)] if lose else [])
    return "ADAR " + "; ".join(parts) + "."


def esc(v):
    return f"{v * 100:.1f}%" if 0 < v < 0.01 or 0.99 < v < 1 else f"{v * 100:.0f}%"


def main_problem(S):
    ag, em = S["tasks"]["ag_news"], S["tasks"]["emotion"]
    lz_em, be_em, be_ag = row(em, "Laya zero-shot"), best_expert(em), best_expert(ag)
    lz_ag, cnn = row(ag, "Laya zero-shot"), row(ag, "TextCNN")
    td = S["typed_decisions"]["convaiinnovations/laya-typed-decisions"]["shipped"]
    c = ag["router"]["chosen"]
    return f"""## 1. The problem

All numbers in this document come from one full Kaggle run with the real Laya model
([`../results/kaggle_run/summary.json`](../results/kaggle_run/summary.json)).

### 1.1 The biggest model is not always the most accurate

On Emotion tweets, Laya answers **{lz_em['accuracy'] * 100:.1f}%** correctly zero-shot, while a
{be_em['params_M']:.1f}M-parameter {be_em['model']} from our syllabus reaches **{be_em['accuracy'] * 100:.1f}%**. On AG News
the order flips: Laya ({lz_ag['accuracy'] * 100:.1f}%) is level with the best syllabus network ({be_ag['model']}, {be_ag['accuracy'] * 100:.1f}%).
Which architecture is right depends on the data, and the course asks us to "identify the appropriate
deep architecture for a given application".

### 1.2 Choosing per input is worth more than choosing per dataset

On AG News a perfect per-input choice between the cheap experts and Laya + ADAR would reach
**{c['oracle_accuracy'] * 100:.2f}%** while calling Laya on only {esc(c['oracle_escalation_rate'])} of inputs; the experts alone reach
{c['tier0_accuracy'] * 100:.2f}%.

### 1.3 Strong models are expensive

On a CPU, Laya takes **{lz_ag['cpu_ms']:.0f} ms** per news article; a TextCNN takes **{cnn['cpu_ms']:.1f} ms**
({lz_ag['cpu_ms'] / cnn['cpu_ms']:.0f}× less). Running the big model on every input wastes most of that time on easy inputs.

### 1.4 Even System-One models are not perfectly calibrated

On the public typed-decisions benchmark, Laya's fine-tuned checkpoint has an expected calibration
error (ECE) of **{td['ece']:.3f}**: its confidence is off by about {td['ece'] * 100:.0f} points on average."""


def main_results(S):
    ag, em = S["tasks"]["ag_news"], S["tasks"]["emotion"]
    a, cas = row(ag, "ADAR (ours"), row(ag, "ADAR cascade")
    base = [r for r in ag["rows"] if not r["model"].startswith("ADAR")]
    bn = min(base, key=lambda r: r["nll"])
    c, ce = ag["router"]["chosen"], em["router"]["chosen"]
    rf = [p for p in ag["router"]["reinforce"] if p["escalation_rate"] > 0]
    out = ["## 3. The improvement", "",
           f"> One full run on Kaggle (GPU T4 ×2, {S['total_minutes']:.0f} minutes) with the real Laya encoder. All models are "
           "temperature-scaled on validation data; a claim holds only where the paired-bootstrap 95% confidence interval excludes zero. "
           "Full reports: [`../results/kaggle_run/ag_news/REPORT.md`](../results/kaggle_run/ag_news/REPORT.md), "
           "[`../results/kaggle_run/emotion/REPORT.md`](../results/kaggle_run/emotion/REPORT.md).", "",
           f"### 3.1 AG News ({S['splits']['ag_news']['test']:,} test articles)", "", table(ag), "", "What this shows:", "",
           f"* {verdict(ag)}", *bullets(ag).replace("{task}", "ag_news").replace("(results/", "(../results/").splitlines(),
           f"* **Most honest probabilities:** log-loss {a['nll']:.3f} vs {bn['nll']:.3f} for the best baseline ({bn['model']}), "
           f"{(1 - a['nll'] / bn['nll']) * 100:.0f}% lower.",
           f"* **The cascade** reaches {cas['accuracy'] * 100:.2f}% at {cas['cpu_ms']:.0f} ms per input, against {a['accuracy'] * 100:.2f}% at "
           f"{a['cpu_ms']:.0f} ms when Laya + ADAR runs on everything ({a['cpu_ms'] / cas['cpu_ms']:.1f}× cheaper).",
           f"* **Stable:** {a['accuracy_std'] * 100:.2f} points standard deviation over the training seeds.", ""]
    if rf:
        p = min(rf, key=lambda p: p["escalation_rate"] if p["accuracy"] >= c["accuracy"] - 0.001 else 9)
        out += [f"The REINFORCE router (Unit IV) finds a cheaper operating point: with λ = {p['lambda']} it escalates "
                f"{esc(p['escalation_rate'])} of articles and reaches {p['accuracy'] * 100:.2f}% (threshold router: {esc(c['escalation_rate'])}, "
                f"{c['accuracy'] * 100:.2f}%).", ""]
    out += [f"### 3.2 Emotion ({S['splits']['emotion']['test']:,} test tweets)", "", table(em), "", "What this shows:", "",
            f"* {verdict(em)}", *bullets(em).replace("{task}", "emotion").replace("(results/", "(../results/").splitlines(),
            f"* Laya is weak on tweets, and the router learned it: it calls Laya on {esc(ce['escalation_rate'])} of inputs.", "",
            "### 3.3 Which parts matter (ablations, seed 0)", "",
            "| Variant | AG News Δ accuracy (95% CI) | Emotion Δ accuracy (95% CI) |", "|---|---|---|"]
    for name, k in (("without the experts (Laya only)", "ablation: laya only"), ("without Laya (experts only)", "ablation: experts only"),
                    ("without attention", "ablation: no attention")):
        cells = []
        for t in (ag, em):
            g = gate(t, k)
            cells.append("–" if not g else f"{-g[1]:+.2f} ({-g[3]:+.2f} to {-g[2]:+.2f}){' **significant**' if g[2] > 0 else ''}")
        out.append(f"| {name} | {cells[0]} | {cells[1]} |")
    out += ["", "### 3.4 What the head learned about the architectures (AG News)", "",
            "| Expert | Average routing weight α | Real accuracy | AUC of “is this expert right?” |", "|---|---:|---:|---:|"]
    names = {"textcnn": "TextCNN", "rnn": "Bi-RNN", "lstm": "Bi-LSTM", "gru": "Bi-GRU"}
    rt = ag.get("routing") or {}
    for k, v in sorted(rt.items(), key=lambda kv: -kv[1]["mean_alpha"]):
        out.append(f"| {names[k]} | {v['mean_alpha']:.3f} | {v['actual_accuracy'] * 100:.1f}% | {v['noul_auc']:.3f} |")
    if rt:
        low = min(rt, key=lambda k: rt[k]["mean_alpha"])
        out += ["", f"Without being told, the head gives {names[low]} the lowest weight"
                + (" (the plain RNN, the architecture the syllabus flags for vanishing gradients)." if low == "rnn" else ".")]
    return "\n".join(out)


def main_limits(S):
    ag, em = S["tasks"]["ag_news"], S["tasks"]["emotion"]
    lim = ["Laya is **frozen**; ADAR trains only the head.", "Only text classification is covered so far."]
    g = gate(em, "best syllabus deep model")
    if g and not g[0]:
        lim.append(f"On Emotion, ADAR **ties** the best syllabus network ({g[1]:+.2f} points, CI {g[2]:+.2f} to {g[3]:+.2f}).")
    g = gate(ag, "ablation: no attention")
    if g and g[2] <= 0:
        lim.append("Attention's accuracy gain is **not significant**; it mainly helps the probabilities.")
    c = ag["router"]["chosen"]
    lim.append(f"The cascade threshold, chosen on validation, gives {c['accuracy'] * 100:.2f}% on test against "
               f"{c['always_tier1_accuracy'] * 100:.2f}% when always escalating.")
    lim.append("For typed-decisions, ADAR-Cal **re-calibrates** Laya; it does not change Laya's answers or accuracy.")
    return "## 4. Limits, stated plainly\n\n" + "\n".join("* " + x for x in lim)


def why(S):
    ag, em = S["tasks"]["ag_news"], S["tasks"]["emotion"]
    lz_em, be_em, lz_ag, cnn = row(em, "Laya zero-shot"), best_expert(em), row(ag, "Laya zero-shot"), row(ag, "TextCNN")
    c = ag["router"]["chosen"]
    td = S["typed_decisions"]["convaiinnovations/laya-typed-decisions"]["shipped"]
    return "\n".join([
        "| Problem | Evidence (from our Kaggle run) |", "|---|---|",
        f"| **The biggest model is not always right.** | Laya gets {lz_em['accuracy'] * 100:.1f}% on Emotion tweets zero-shot; a {be_em['params_M']:.1f}M-parameter {be_em['model']} gets {be_em['accuracy'] * 100:.1f}%. |",
        f"| **Choosing per input beats choosing per dataset.** | On AG News a perfect per-input router would reach {c['oracle_accuracy'] * 100:.2f}% calling Laya on {esc(c['oracle_escalation_rate'])} of inputs. |",
        f"| **Strong models are slow.** | On a CPU, Laya takes {lz_ag['cpu_ms']:.0f} ms per news article; a TextCNN {cnn['cpu_ms']:.1f} ms. |",
        f"| **Even System-One models drift.** | Laya's fine-tuned checkpoint has ECE {td['ece']:.3f} on the public typed-decisions benchmark. |"])


def demo(log):
    if not os.path.exists(log):
        return ""
    t = open(log, errors="replace").read().replace('\\"', '"')
    ms = re.findall(r"\{'label': '[^']+', 'probabilities': \{[^}]*\}, 'p_head_correct': [\d.]+, 'escalate': (?:True|False)", t)
    if not ms:
        return ""
    return ("Output of the demo cell in the Kaggle run ([`results/kaggle_run/kernel.log`](results/kaggle_run/kernel.log)):\n\n```text\n"
            + "\n".join(m + ", ...}" for m in ms[:2]) + "\n```")


def fill(path, name, body):
    txt = open(path).read()
    a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if a not in txt:
        sys.exit(f"{path} has no {name} markers")
    open(path, "w").write(txt[:txt.index(a) + len(a)] + "\n" + body + "\n" + txt[txt.index(b):])


def main():
    summary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "kaggle_run", "summary.json")
    readme = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "README.md")
    main_md = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "docs", "MAIN.md")
    S = json.load(open(summary))
    fill(readme, "results", section(S))
    fill(readme, "why", why(S))
    fill(readme, "demo", demo(os.path.join(os.path.dirname(os.path.abspath(summary)), "kernel.log")))
    fill(main_md, "main-problem", main_problem(S))
    fill(main_md, "main-results", main_results(S) + "\n\n" + main_limits(S))
    print("README.md and docs/MAIN.md updated from", summary)


if __name__ == "__main__":
    main()
