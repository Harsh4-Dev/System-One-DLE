"""Fills the number-bearing blocks of docs/ADAR_explained.html (between <!-- gen:x --> markers and the
/* gen:data */ script block) from results/kaggle_run/summary.json.

    python tools/report/update_explainer.py [summary.json] [ADAR_explained.html]
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readme_results import best_expert, esc, gate, row, verdict  # noqa: E402

SHORT = {"Laya zero-shot (typed choice)": "Laya zero-shot", "Laya encoder + Logistic Regression": "Laya + linear probe",
         "Stacking LogReg (Laya emb + experts)": "Stacking (Laya + experts)", "sklearn MLPClassifier (TF-IDF)": "MLP (TF-IDF)",
         "Average of syllabus experts": "Average of 4 experts"}
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fill(txt, name, body):
    a, b = f"<!-- gen:{name} -->", f"<!-- /gen:{name} -->"
    i, j = txt.index(a) + len(a), txt.index(b)
    return txt[:i] + body + txt[j:]


def blocks(S):
    ag, em = S["tasks"]["ag_news"], S["tasks"]["emotion"]
    a, cas = row(ag, "ADAR (ours"), row(ag, "ADAR cascade")
    lz_em, be_em, lz_ag, cnn = row(em, "Laya zero-shot"), best_expert(em), row(ag, "Laya zero-shot"), row(ag, "TextCNN")
    c, ce = ag["router"]["chosen"], em["router"]["chosen"]
    td = S["typed_decisions"]["convaiinnovations/laya-typed-decisions"]["shipped"]
    base = [r for r in ag["rows"] if not r["model"].startswith("ADAR")]
    bn = min(base, key=lambda r: r["nll"])
    g_lz = gate(ag, "best Laya zero-shot")
    out = {}
    prob = lambda h, big, p: f'      <div class="problem">\n        <h3>{h}</h3>\n        <span class="big">{big}</span>\n        <p>{p}</p>\n      </div>\n'
    out["problems"] = ('    <div class="problems">\n'
        + prob("The biggest model is not always right", f"{lz_em['accuracy'] * 100:.1f}% vs {be_em['accuracy'] * 100:.1f}%",
               f"Laya zero-shot on Emotion tweets, versus a {be_em['params_M']:.1f}M-parameter {be_em['model']} from our syllabus.")
        + prob("Picking one route for every input wastes accuracy", f"{c['tier0_accuracy'] * 100:.1f}% → {c['oracle_accuracy'] * 100:.1f}%",
               f"AG News: the cheap experts alone, versus a perfect per-input choice that calls Laya on only {esc(c['oracle_escalation_rate'])} of inputs.")
        + prob("Strong models are expensive", f"{cnn['cpu_ms']:.1f} ms vs {lz_ag['cpu_ms']:.0f} ms",
               "A TextCNN versus Laya, per news article on a CPU.")
        + prob("Even System-One models drift", f"ECE {td['ece']:.3f}",
               f"Laya's fine-tuned checkpoint on the public typed-decisions benchmark: its confidence is off by about {td['ece'] * 100:.0f} points on average.")
        + "    </div>\n")
    out["cost0"] = f", {ag['router']['cost0_ms']:.0f} ms together"
    out["params"] = f"{a['params_M']:.1f}M"
    out["branches"] = (f'      <div class="branch"><b>Yes</b> ({esc(1 - c["escalation_rate"])} of news articles, {esc(1 - ce["escalation_rate"])} of tweets): answer immediately from the experts.</div>\n'
                       f'      <div class="branch"><b>No</b> ({esc(c["escalation_rate"])} of articles, {esc(ce["escalation_rate"])} of tweets): escalate ↓</div>\n')
    out["results-top"] = (f'    <div class="note col"><strong>Where these numbers come from.</strong> One full run of our Kaggle notebook (GPU T4 ×2, {S["total_minutes"]:.0f} minutes) '
        f'with the real Laya encoder, on the full AG News data ({S["splits"]["ag_news"]["test"]:,} test articles). Every number is read from <code>results/kaggle_run/summary.json</code>. '
        'Every model is temperature-scaled on validation data, and a claim counts only when its 95% bootstrap confidence interval excludes zero.</div>\n\n'
        '    <div class="stats">\n'
        f'      <div class="stat"><div class="v">{a["accuracy"] * 100:.2f}%</div><div class="l">ADAR accuracy, {g_lz[1]:+.2f} points over Laya zero-shot (CI {g_lz[2]:+.2f} to {g_lz[3]:+.2f}). {verdict(ag)}</div></div>\n'
        f'      <div class="stat"><div class="v">−{(1 - a["nll"] / bn["nll"]) * 100:.0f}%</div><div class="l">log-loss (NLL) versus the best baseline: more honest probabilities</div></div>\n'
        f'      <div class="stat"><div class="v">{esc(c["escalation_rate"])}</div><div class="l">of inputs needed Laya in the cascade: {cas["accuracy"] * 100:.2f}% accuracy at {a["cpu_ms"] / cas["cpu_ms"]:.1f}× less CPU time</div></div>\n'
        '    </div>\n\n')
    rows = sorted((r for r in ag["rows"] if not r["model"].startswith("ADAR ablation")), key=lambda r: r["accuracy"])
    lo = math.floor(min(r["accuracy"] for r in rows) * 100 - 1) / 100
    hi = math.ceil(max(r["accuracy"] for r in rows) * 100 + 0.5) / 100
    out["acc-caption"] = f'      <figcaption>Axis starts at {lo * 100:.0f}% so the differences are visible. Hover a dot for exact values.</figcaption>\n'
    out["cost-caption"] = ('      <figcaption>Up and to the left is better. The line is the cascade with every escalation threshold; '
                           f'the router picked {esc(c["escalation_rate"])} escalation on validation data.</figcaption>\n')
    ab = []
    for name, k in (("Without the experts (Laya only)", "ablation: laya only"), ("Without Laya (experts only)", "ablation: experts only"),
                    ("Without attention", "ablation: no attention")):
        r, g = row(ag, "ADAR " + k), gate(ag, k)
        if r and g:
            tell = f"{-g[1]:+.2f} points (CI {-g[3]:+.2f} to {-g[2]:+.2f}), " + ("significant" if g[2] > 0 else "not significant")
            ab.append(f'          <tr><td>{name}</td><td class="n">{r["accuracy"] * 100:.2f}%</td><td class="n">{r["nll"]:.3f}</td><td>{tell}</td></tr>\n')
    names = {"textcnn": "TextCNN", "rnn": "Bi-RNN", "lstm": "Bi-LSTM", "gru": "Bi-GRU"}
    rt = ag.get("routing") or {}
    rr = "".join(f'          <tr><td>{names[k]}</td><td class="n">{v["mean_alpha"]:.3f}</td><td class="n">{v["actual_accuracy"] * 100:.1f}%</td><td class="n">{v["noul_auc"]:.3f}</td></tr>\n'
                 for k, v in sorted(rt.items(), key=lambda kv: -kv[1]["mean_alpha"]))
    low = min(rt, key=lambda k: rt[k]["mean_alpha"]) if rt else None
    aucs = [v["noul_auc"] for v in rt.values()]
    out["ablation-routing"] = ('    <h3>Which parts matter</h3>\n    <div class="table-wrap">\n      <table>\n'
        '        <thead><tr><th>Variant (seed 0)</th><th class="n">Accuracy</th><th class="n">NLL</th><th>Change when removed</th></tr></thead>\n        <tbody>\n'
        f'          <tr class="ours"><td>Full ADAR (mean of seeds)</td><td class="n">{a["accuracy"] * 100:.2f}%</td><td class="n">{a["nll"]:.3f}</td><td></td></tr>\n'
        + "".join(ab) + '        </tbody>\n      </table>\n    </div>\n\n'
        '    <h3>What the head learned about the architectures</h3>\n'
        '    <div class="col"><p>Nobody told the head which expert is better. It worked out a routing weight for each one, per input:</p></div>\n'
        '    <div class="table-wrap">\n      <table>\n'
        '        <thead><tr><th>Expert</th><th class="n">Avg routing weight</th><th class="n">Real accuracy</th><th class="n">AUC of “is it right?”</th></tr></thead>\n        <tbody>\n'
        + rr + '        </tbody>\n      </table>\n    </div>\n'
        + (f'    <div class="col"><p>It gives {names[low]} the lowest weight'
           + (", the plain RNN, which the syllabus flags for vanishing gradients" if low == "rnn" else "")
           + f'. Its per-input “is this expert right?” predictions separate right from wrong answers with AUCs between {min(aucs):.2f} and {max(aucs):.2f}.</p></div>\n' if rt else ""))
    lim = ["Laya is frozen; ADAR trains only the head.", "Only text classification is covered so far."]
    g = gate(em, "best syllabus deep model")
    if g and not g[0]:
        lim.append(f"On Emotion tweets, ADAR ties the best syllabus network ({g[1]:+.2f} points, CI {g[2]:+.2f} to {g[3]:+.2f}).")
    g = gate(ag, "ablation: no attention")
    if g and g[2] <= 0:
        lim.append("Attention's accuracy gain is not significant; it mainly helps the probabilities.")
    lim.append(f"The cascade threshold chosen on validation gives {c['accuracy'] * 100:.2f}% on test against {c['always_tier1_accuracy'] * 100:.2f}% when always escalating.")
    lim.append("On typed-decisions, ADAR-Cal re-calibrates Laya's probabilities; it does not change Laya's answers.")
    out["limits"] = '    <ul class="clean col">\n' + "".join(f"      <li>{x}</li>\n" for x in lim) + "    </ul>\n"
    # chart data
    data = []
    for r in rows:
        name = r["model"]
        if name.startswith("ADAR cascade"):
            name = f"ADAR cascade ({esc(c['escalation_rate'])} escalated)"
        elif name.startswith("ADAR (ours"):
            name = re.sub(r"ADAR \(ours, mean of (\d+) seeds\)", r"ADAR (mean of \1 seeds)", name)
        name = SHORT.get(name, name)
        data.append(f'    ["{name}", {r["accuracy"]:.4f}, {r["nll"]:.3f}, {r["brier"]:.3f}, {r["ece_ts"]:.3f}, {r["cpu_ms"]:.3g}, {1 if r["model"].startswith("ADAR") else 0}]')
    curve = sorted(ag["router"]["curve"], key=lambda p: p["mean_cost_ms"])
    cmin = min([r["cpu_ms"] for r in rows] + [p["mean_cost_ms"] for p in curve])
    cmax = max([r["cpu_ms"] for r in rows] + [p["mean_cost_ms"] for p in curve])
    c0, c1 = 10 ** math.floor(math.log10(cmin)), 10 ** math.ceil(math.log10(cmax))
    cticks = [v for d in range(int(math.log10(c0)), int(math.log10(c1)) + 1) for v in (10 ** d, 3 * 10 ** d) if c0 <= v <= c1]
    y0 = math.floor(min(r["accuracy"] for r in rows) * 100 - 0.5) / 100
    y1 = math.ceil(max(r["accuracy"] for r in rows) * 100 + 0.3) / 100
    yt = [round(v / 100, 2) for v in range(int(round(y0 * 100)), int(round(y1 * 100)) + 1, 1 if y1 - y0 <= 0.08 else 2)]
    lab = {r: 1 for r in ("TextCNN", "Bi-GRU", "Linear SVM (TF-IDF)", "Laya zero-shot", "Random Forest (TF-IDF)", "MLP (TF-IDF)",
                          re.sub(r"ADAR \(ours, mean of (\d+) seeds\)", r"ADAR (mean of \1 seeds)", a["model"]),
                          f"ADAR cascade ({esc(c['escalation_rate'])} escalated)")}
    step = 0.01 if hi - lo <= 0.1 else 0.02
    out["data"] = ("  var rows = [\n" + ",\n".join(data) + "\n  ];\n"
                   f"  // cascade threshold sweep on the test set (results/kaggle_run/summary.json): [mean ms, accuracy]\n"
                   f"  var curve = {json.dumps([[round(p['mean_cost_ms'], 2), round(p['accuracy'], 4)] for p in curve])};\n"
                   f"  var AX = {json.dumps({'x0': lo, 'x1': hi, 'step': step, 'c0': c0, 'c1': c1, 'cticks': cticks, 'y0': y0, 'y1': y1, 'yticks': yt})};\n"
                   f"  var LABELLED = {json.dumps(lab)};\n")
    return out


def main():
    summary = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "kaggle_run", "summary.json")
    page = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "ADAR_explained.html")
    S = json.load(open(summary))
    txt = open(page).read()
    out = blocks(S)
    for k, v in out.items():
        if k == "data":
            i, j = txt.index("/* gen:data */") + len("/* gen:data */"), txt.index("  /* /gen:data */")
            txt = txt[:i] + "\n" + v + txt[j:]
        else:
            txt = fill(txt, k, ("\n" + v) if v.startswith(" ") else v)
    open(page, "w").write(txt)
    print("explainer updated from", summary)


if __name__ == "__main__":
    main()
