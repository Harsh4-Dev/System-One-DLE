"""Builds the demo deck (Slides artifact format, 1920x1080 HTML sections) from one run's summary.json.

    python tools/report/build_slides.py results/kaggle_run/summary.json <deck>/project [--log kernel.log]

Every number on a slide is read from summary.json (or, for the demo output, from the run's log);
nothing is typed in by hand. Writes <deck>/project/slides/*.html and updates order/sections in deck.json.
"""
import argparse
import html
import json
import math
import os
import re

# ---------- palette: clean, one accent ----------
BG, ALT, CARD, LINE = "#F7F3EA", "#EFE8DA", "#FCFAF5", "#DDD3C0"
INK, BODY, MUTED = "#1F2233", "#3F4250", "#6B6E78"
ACC, ACCT, HL, ACCL = "#C0503A", "#A8432F", "#F3DED5", "#DE9C8C"
SLATE, TAUPE, SAND = "#4A5874", "#A79E8C", "#CFC7B6"
HEAD = "font-family:'Bricolage Grotesque', Arial, sans-serif"
MONO = "font-family:'JetBrains Mono', 'Courier New', monospace"

S = None  # summary
DEMO = []  # parsed demo predictions


# ---------- data helpers ----------
def rows(task):
    return S["tasks"][task]["rows"]


def row(task, prefix):
    return next((r for r in rows(task) if r["model"].startswith(prefix)), None)


def adar(task):
    return row(task, "ADAR (ours")


def seeds(task):
    m = re.search(r"mean of (\d+) seeds", adar(task)["model"])
    return int(m.group(1)) if m else 1


def router(task):
    return S["tasks"][task].get("router") or {}


def esc(task):
    return router(task)["chosen"]["escalation_rate"]


def baselines(task):
    return [r for r in rows(task) if not r["model"].startswith("ADAR")]


EXPERTS = ("TextCNN", "Bi-RNN", "Bi-LSTM", "Bi-GRU")


def best_expert(task):
    return max((r for r in rows(task) if r["model"] in EXPERTS), key=lambda r: r["accuracy"])


def gate(task, needle):
    """(passed, delta_pp, lo, hi, text) for the first QC line containing needle."""
    for q in S["tasks"][task]["qc"]:
        if needle in q:
            m = re.search(r"delta=([+-][\d.]+)pp,? (?:95% )?CI \[([+-][\d.]+), ([+-][\d.]+)\]", q)
            d = tuple(float(x) for x in m.groups()) if m else (math.nan,) * 3
            return (q.startswith("PASS"),) + d + (q,)
    return None


def gate_model(task, needle):
    g = gate(task, needle)
    m = re.search(r"\(([^()]*(?:\([^()]*\))?[^()]*)\): delta", g[4]) if g else None
    return m.group(1) if m else ""


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def epct(v):
    """escalation share: whole percent, one decimal below 1%"""
    return f"{v * 100:.1f}%" if 0 < v < 0.01 or 0.99 < v < 1 else f"{v * 100:.0f}%"


def pts(d, nd=2):
    return f"{d:+.{nd}f} pts"


def ms(v):
    return f"{v:.1f} ms" if v < 10 else f"{v:.0f} ms"


def times(a, b):
    r = a / b
    return f"{r:.1f}×" if r < 10 else f"{r:.0f}×"


def e(s):
    return html.escape(str(s), quote=False)


SHORT = {"Logistic Regression (TF-IDF)": "Log. regression · TF-IDF", "Linear SVM (TF-IDF)": "Linear SVM · TF-IDF",
         "Naive Bayes (TF-IDF)": "Naive Bayes · TF-IDF", "Random Forest (TF-IDF)": "Random forest · TF-IDF",
         "sklearn MLPClassifier (TF-IDF)": "MLP · TF-IDF", "Average of syllabus experts": "Average of 4 experts",
         "Laya zero-shot (typed choice)": "Laya zero-shot", "Laya encoder + Logistic Regression": "Laya + linear probe",
         "Stacking LogReg (Laya emb + experts)": "Stacking (Laya + experts)"}


def short(name):
    if name.startswith("ADAR (ours"):
        return "ADAR (ours)"
    if name.startswith("ADAR cascade"):
        return "ADAR cascade"
    return SHORT.get(name, name)


def colour(name):
    if name.startswith("ADAR cascade"):
        return ACCL
    if name.startswith("ADAR"):
        return ACC
    if name.startswith("Laya"):
        return SLATE
    if name in EXPERTS or name.startswith("Average"):
        return TAUPE
    return SAND


# ---------- layout helpers ----------
def section(sid, inner, notes, transition="fade", bg=BG, layout="flow"):
    lay = "display:flex;flex-direction:column;justify-content:space-between;gap:32px" if layout == "flow" else \
        "display:flex;flex-direction:column;gap:32px"
    return (f'<section id="{sid}" data-transition="{transition}" style="background:{bg};color:{INK};'
            f"font-family:'DM Sans', Arial, sans-serif;padding:128px;{lay}\">\n{inner}\n<aside>{e(notes)}</aside>\n</section>\n")


def header(kicker, title):
    return (f'<div style="display:flex;flex-direction:column;gap:12px">\n'
            f'<p style="font-size:24px;letter-spacing:3px;text-transform:uppercase;color:{ACCT};font-weight:700">{e(kicker)}</p>\n'
            f'<h2 style="{HEAD};font-size:72px;font-weight:700;line-height:1.1;letter-spacing:-1px">{e(title)}</h2>\n</div>')


def takeaway(text, label="What this shows"):
    return (f'<p style="font-size:28px;line-height:1.4;color:{BODY};border-top:1px solid {LINE};padding:20px 0 0 0">'
            f'<b style="color:{INK}">{label}:</b> {e(text)}</p>')


def takeaway_abs(text, top=944, label="What this shows"):
    return (f'<p style="position:absolute;left:128px;top:{top}px;width:1664px;font-size:26px;line-height:1.35;color:{BODY};'
            f'border-top:1px solid {LINE};padding:16px 0 0 0"><b style="color:{INK}">{label}:</b> {e(text)}</p>')


def card(inner, extra="", border=LINE, w=None):
    width = f"width:{w}px;" if w else "flex:1;"
    return (f'<div style="{width}display:flex;flex-direction:column;gap:12px;background:{CARD};border:1px solid {border};'
            f'border-radius:16px;padding:32px;{extra}">\n{inner}\n</div>')


def label(text, color=MUTED):
    return f'<p style="font-size:22px;letter-spacing:2px;text-transform:uppercase;color:{color};font-weight:700">{e(text)}</p>'


def para(text, size=26, color=BODY, extra=""):
    return f'<p style="font-size:{size}px;line-height:1.4;color:{color};{extra}">{text}</p>'


def big(text, color=INK, size=96):
    return f'<p style="{HEAD};font-size:{size}px;font-weight:800;line-height:1;color:{color}">{e(text)}</p>'


def bars(items, lo, hi, top, x_label=128, w_label=330, x_bar=478, w_max=700, step=44, h=30, order0=1, fmt=None):
    """Horizontal bar chart of (name, value) with values in [0,1]; returns html and bottom y."""
    out, y = [], top
    fmt = fmt or (lambda v: pct(v))
    for i, (name, v) in enumerate(items):
        w = max(4, round((v * 100 - lo) / (hi - lo) * w_max))
        bold = "font-weight:700;" if name.startswith("ADAR") else ""
        o = order0 + i
        out.append(f'<p style="position:absolute;left:{x_label}px;top:{y}px;width:{w_label}px;height:{h}px;line-height:{h}px;'
                   f'font-size:22px;text-align:right;white-space:nowrap;{bold}" data-build-in="fade order {o} auto">{e(short(name))}</p>')
        out.append(f'<div style="position:absolute;left:{x_bar}px;top:{y}px;width:{w}px;height:{h}px;background:{colour(name)};'
                   f'border-radius:0 6px 6px 0" data-build-in="left order {o} auto"></div>')
        out.append(f'<p style="position:absolute;left:{x_bar + w + 12}px;top:{y}px;width:120px;height:{h}px;line-height:{h}px;'
                   f'font-size:22px;{bold}" data-build-in="fade order {o} auto">{fmt(v)}</p>')
        y += step
    return "\n".join(out), y


def legend(top, left=128):
    items = [("ADAR", ACC), ("Laya", SLATE), ("syllabus networks", TAUPE), ("scikit-learn / stacking", SAND)]
    out, x = [], left
    for name, c in items:
        out.append(f'<div style="position:absolute;left:{x}px;top:{top + 5}px;width:18px;height:18px;background:{c};border-radius:4px"></div>')
        out.append(f'<p style="position:absolute;left:{x + 26}px;top:{top}px;width:300px;font-size:20px;color:{MUTED};white-space:nowrap">{e(name)}</p>')
        x += 26 + len(name) * 11 + 40
    return "\n".join(out)


def verdict(task):
    """Honest one-line summary of ADAR vs the three main reference points."""
    refs = [("Laya zero-shot", "best Laya zero-shot"), ("the best syllabus network", "best syllabus deep model"),
            ("the best scikit-learn model", "best scikit-learn model")]
    win, tie, lose = [], [], []
    for name, needle in refs:
        g = gate(task, needle)
        if not g:
            continue
        ok, d, lo, hi, _ = g
        (win if ok else lose if hi < 0 else tie).append(name)
    parts = []
    if win:
        parts.append("significantly beats " + join(win))
    if tie:
        parts.append("ties " + join(tie) + " (difference not significant)")
    if lose:
        parts.append("is significantly behind " + join(lose))
    return "ADAR " + "; ".join(parts) + "."


def join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


# ---------- slides ----------
def s_cover():
    nodes = [(1180, 300, "Experts", "TextCNN · RNN · LSTM · GRU", TAUPE, CARD), (1180, 620, "Laya", "System-One model", SLATE, CARD),
             (1500, 460, "ADAR", "decides per input", ACC, HL)]
    out = [f'<div style="width:960px;display:flex;flex-direction:column;gap:28px">',
           f'<p style="font-size:24px;letter-spacing:4px;text-transform:uppercase;color:{ACCT};font-weight:700">Deep Learning project · System-One models</p>',
           f'<h1 style="{HEAD};font-size:200px;font-weight:800;line-height:1;letter-spacing:-6px;color:{INK}">ADAR</h1>',
           f'<p style="{HEAD};font-size:44px;font-weight:600;line-height:1.2;color:{INK}">Adaptive Deep Architecture Routing</p>',
           f'<p style="font-size:30px;line-height:1.4;color:{BODY}">A small trained head that decides which deep-learning model to trust for each input, and when the big model is worth calling.</p>',
           f'<p style="font-size:24px;color:{MUTED}">[Team names] · [Course code] · [Date]</p>', '</div>']
    for i, (x, y, t, sub, b, fill) in enumerate(nodes):
        out.append(f'<div style="position:absolute;left:{x}px;top:{y}px;width:290px;height:140px;background:{fill};border:2px solid {b};border-radius:16px;'
                   f'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px" data-build-in="fade order {i + 1} auto">'
                   f'<p style="font-size:30px;font-weight:700">{t}</p><p style="font-size:20px;color:{MUTED}">{sub}</p></div>')
    out.append(f'<x-connector x1="1470" y1="370" x2="1500" y2="500" style="color:{MUTED};border-width:2px" data-build-in="fade order 3 auto"></x-connector>')
    out.append(f'<x-connector x1="1470" y1="690" x2="1500" y2="560" style="color:{MUTED};border-width:2px" data-build-in="fade order 3 auto"></x-connector>')
    notes = ("Our project is ADAR, Adaptive Deep Architecture Routing. On the right is the whole idea: small networks from "
             "our syllabus, the experts; a big open System-One model called Laya; and ADAR, a small trained head that decides, "
             "for every input, which of them to trust and whether the big model is needed at all. Every number in this deck "
             f"comes from one full run of our notebook on Kaggle with the real Laya model, {S['total_minutes']:.0f} minutes end to end.")
    return section("cover", "\n".join(out), notes, layout="center").replace("justify-content:space-between", "justify-content:center")


def s_keyterms():
    terms = [("System-One model", "Answers typed questions (choice, score, yes/no) in one forward pass, with a probability for every option. No text generation."),
             ("Expert", f"A small syllabus network trained for the task: TextCNN, RNN, LSTM or GRU. On a CPU it takes {ms(min(row('ag_news', k)['cpu_ms'] for k in EXPERTS))}–{ms(max(row('ag_news', k)['cpu_ms'] for k in EXPERTS))} per input."),
             ("Out-of-fold", "We only use an expert’s opinion on data that expert never trained on, so its confidence is not inflated."),
             ("Calibration (ECE)", "Does “80% sure” mean right 80% of the time? ECE is the average gap. Lower is better; 0 is perfect."),
             ("Log-loss (NLL)", "Punishes confident mistakes hardest. Lower means the probabilities are more honest, not just the answers more often right."),
             ("Cascade", "Cheap experts answer first; only inputs they are unsure about “escalate” to the expensive model.")]
    cards = "\n".join(card(f'<p style="font-size:30px;font-weight:700">{e(t)}</p>\n' + para(e(d), 24), border=ACC if t == "Cascade" else LINE)
                      for t, d in terms)
    inner = (header("Before we start", "Key Terms") + f'\n<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px">\n{cards}\n</div>\n'
             + takeaway("we judge every model on two things, how often it is right and how honest its confidence is.", "What this sets up"))
    notes = ("A few terms we use throughout. A System-One model answers typed questions in a single pass, giving a probability "
             "for every option instead of generating text. Experts are our small syllabus networks. Out-of-fold means we only trust "
             "an expert's opinion on data it never trained on. Calibration asks whether 80 percent sure really means right 80 percent "
             "of the time; the expected calibration error, ECE, is the average gap. Log-loss punishes confident mistakes. A cascade "
             "lets cheap models answer first and escalates only the hard inputs. So every model is judged on two things: how often "
             "it is right, and how honest its confidence is.")
    return section("keyterms", inner, notes)


def s_problem():
    lz_em, be_em = row("emotion", "Laya zero-shot"), best_expert("emotion")
    lz_ag, cnn = row("ag_news", "Laya zero-shot"), row("ag_news", "TextCNN")
    td = S["typed_decisions"]["convaiinnovations/laya-typed-decisions"]["shipped"]
    lp = lz_ag["params_M"]
    c1 = card(label("The big model is not always right") + big(pct(lz_em["accuracy"]), SLATE)
              + para(f"Laya ({lp:.0f}M-parameter encoder) on Emotion tweets, zero-shot. A {be_em['params_M']:.1f}M-parameter {be_em['model']} gets {pct(be_em['accuracy'])}."))
    c2 = card(label("Big models are slow") + big(times(lz_ag["cpu_ms"], cnn["cpu_ms"]), INK)
              + para(f"slower on a CPU: Laya takes {ms(lz_ag['cpu_ms'])} per news article, a TextCNN {ms(cnn['cpu_ms'])}."))
    c3 = card(label("Confidence is not honest") + big(f"{td['ece']:.3f}", ACCT)
              + para(f"calibration error (ECE) of Laya’s fine-tuned checkpoint on the typed-decisions benchmark: its confidence is off by about {td['ece'] * 100:.0f} points."))
    inner = (header("Why this project", "The Problem") + f'\n<div style="display:flex;gap:32px">\n{c1}\n{c2}\n{c3}\n</div>\n'
             + takeaway("no single model wins on accuracy, cost and honesty at once, so the right choice depends on the input. "
                        "The course asks us to pick the right architecture for an application; ADAR makes that choice for every input."))
    notes = (f"Three problems, all measured in our own run. First, the biggest model is not always the best: Laya gets only "
             f"{pct(lz_em['accuracy'])} on emotion tweets, while a {be_em['params_M']:.1f} million parameter {be_em['model']} from our syllabus gets "
             f"{pct(be_em['accuracy'])}. Second, the big model is expensive: on a CPU it is {times(lz_ag['cpu_ms'], cnn['cpu_ms'])} slower than a TextCNN. "
             f"Third, confidence is not honest: Laya's fine-tuned checkpoint has a calibration error of {td['ece']:.3f} on the "
             "typed-decisions benchmark. Our course objective is to identify the right architecture for an application. "
             "We asked whether a model can learn to make that choice per input.")
    return section("problem", inner, notes)


def s_systemone():
    lp = row("ag_news", "Laya zero-shot")["params_M"]
    box = lambda t, b, border=INK: (f'<div style="flex:1;display:flex;flex-direction:column;gap:12px;background:{CARD};border:1px solid {border};border-radius:16px;padding:28px">'
                                    f'{label(t)}{para(b, 26, INK)}</div>')
    arrow = f'<x-shape kind="arrow-right" style="width:56px;height:28px;background:{MUTED}"></x-shape>'
    flow = (f'<div style="display:flex;align-items:center;gap:20px">{box("State", "<i>“We were billed twice. Refund it or we cancel.”</i>")}{arrow}'
            f'{box("Typed questions", "department? <b>choice</b><br>urgency? <b>score</b><br>will they cancel? <b>noul</b>")}{arrow}'
            f'{box("Laya", "one forward pass, no text generated", SLATE)}{arrow}'
            f'{box("Typed answers", "a probability for every option, a level, or P(yes)")}</div>')
    stat = lambda n, t: (f'<div style="flex:1;border-top:2px solid {INK};padding:20px 0 0 0;display:flex;flex-direction:column;gap:8px">'
                         f'<p style="{HEAD};font-size:44px;font-weight:700">{n}</p>{para(t)}</div>')
    stats = (f'<div style="display:flex;gap:32px">{stat(f"{lp:.0f}M", "parameters in Laya’s ModernBERT-large encoder (measured in our run), plus a small decision head")}'
             f'{stat("Open", "Apache-2.0, the open alternative to TypeSafe’s Jev")}{stat("3 types", "choice · score · noul (a yes/no probability)")}</div>')
    inner = header("Background", "System-One Models") + "\n" + flow + "\n" + stats + "\n" + takeaway(
        "a System-One model already gives a probability for every option. ADAR needs exactly that to weigh Laya against the small networks.")
    notes = ("A System-One model is named after fast, intuitive System 1 thinking. Instead of writing text word by word like a "
             "chatbot, it reads a state, here a customer message, plus typed questions, and answers all of them in one forward "
             "pass: a probability for every option, a level on a scale, or a yes/no probability called noul. TypeSafe released "
             f"the first one, Jev. We use Laya, the open-source alternative: a {lp:.0f} million parameter ModernBERT-large encoder "
             "with a decision head, Apache licensed. Because it outputs probabilities, ADAR can weigh it against other models.")
    return section("systemone", inner, notes)


def s_idea():
    inner = (f'<p style="font-size:24px;letter-spacing:4px;text-transform:uppercase;color:{ACCT};font-weight:700">Our idea</p>\n'
             f'<h2 style="{HEAD};font-size:80px;font-weight:700;line-height:1.12;letter-spacing:-2px;width:1500px;border-left:8px solid {ACC};padding:0 0 0 48px">'
             'Let small models answer first. Call the big model only when they are unsure, and let a trained head decide whom to trust.</h2>\n'
             f'<p style="font-size:30px;line-height:1.4;color:{BODY};width:1400px;padding:0 0 0 56px">Instead of choosing one architecture by hand for the whole dataset, ADAR learns the choice for each input.</p>')
    notes = ("That leads to one idea. Let the small, cheap syllabus networks answer first. Only when they are unsure do we call "
             "the big System-One model. And instead of us choosing the architecture by hand, a small trained head decides, for "
             "each input, which model to trust and how much.")
    return section("idea", inner, notes).replace("justify-content:space-between", "justify-content:center")


def s_architecture():
    e_ag = esc("ag_news")
    c0 = router("ag_news")["cost0_ms"]
    lp = row("ag_news", "Laya zero-shot")["params_M"]
    hp = adar("ag_news")["params_M"]
    o = []
    box = lambda x, y, w, h, inner, border, fill=CARD, order=1, anim="rise", extra="": (
        f'<div {extra}style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;background:{fill};border:2px solid {border};border-radius:16px;'
        f'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px" data-build-in="{anim} order {order} auto">{inner}</div>')
    conn = lambda x1, y1, x2, y2, order, extra="": (f'<x-connector x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {extra}style="color:{INK};border-width:2px" '
                                                    f'data-build-in="fade order {order} auto"></x-connector>')
    t = lambda s, size=28, w=700, c=INK: f'<p style="font-size:{size}px;font-weight:{w};color:{c}">{s}</p>'
    o.append(box(128, 330, 240, 120, t("Input text", 30), INK, order=1, anim="fade"))
    o.append(conn(368, 390, 448, 390, 2))
    o.append(box(448, 310, 400, 160, t("Syllabus experts") + t("TextCNN · RNN · LSTM · GRU", 22, 400, BODY) + t(f"{ms(c0)} on a CPU, all four", 22, 400, MUTED), TAUPE, order=2))
    o.append(conn(848, 390, 928, 390, 3))
    o.append(f'<x-shape kind="diamond" style="position:absolute;left:928px;top:320px;width:240px;height:140px;background:{CARD};border:2px solid {INK}" data-build-in="pop order 3 auto"></x-shape>')
    o.append(f'<p style="position:absolute;left:928px;top:373px;width:240px;font-size:28px;line-height:34px;font-weight:700;text-align:center;white-space:nowrap" data-build-in="pop order 3 auto">Sure?</p>')
    o.append(conn(1168, 390, 1352, 390, 4))
    o.append(f'<p style="position:absolute;left:1180px;top:334px;width:170px;font-size:22px;font-weight:700;color:{MUTED}" data-build-in="fade order 4 auto">yes · {epct(1 - e_ag)}</p>')
    o.append(box(1352, 330, 440, 120, t("Answer from the experts") + t("fast path", 22, 400, MUTED), TAUPE, order=4))
    o.append(conn(1048, 460, 1048, 640, 5))
    o.append(f'<p style="position:absolute;left:1064px;top:528px;width:170px;font-size:22px;font-weight:700;color:{MUTED}" data-build-in="fade order 5 auto">no · {epct(e_ag)}</p>')
    o.append(box(898, 640, 300, 150, t("Laya encoder") + t(f"frozen · {lp:.0f}M", 22, 400, MUTED), SLATE, order=5))
    o.append(conn(1198, 715, 1352, 715, 6))
    o.append(box(1352, 620, 440, 190, t("ADAR head", 32, 800) + t(f"answer · P(answer right)", 22, 400, BODY) + t(f"P(each expert right) · {hp:.1f}M params", 22, 400, BODY),
                 ACC, HL, order=6, anim="pop", extra='id="adarhead" '))
    dash = f'style="color:{TAUPE};border-width:2px;border-style:dashed" data-build-in="fade order 7 auto"'
    o.append(f'<x-connector x1="648" y1="470" x2="648" y2="860" head="none" {dash}></x-connector>')
    o.append(f'<x-connector x1="648" y1="860" x2="1572" y2="860" head="none" {dash}></x-connector>')
    o.append(f'<x-connector x1="1572" y1="860" x2="1572" y2="810" {dash}></x-connector>')
    o.append(f'<p style="position:absolute;left:664px;top:872px;width:420px;font-size:22px;font-weight:700;color:{MUTED}" data-build-in="fade order 7 auto">expert votes + confidence</p>')
    o.append(takeaway_abs(f"on AG News, {epct(1 - e_ag)} of inputs never touch the {lp:.0f}M-parameter model; the rest get both views, combined by the head."))
    notes = (f"Here is the full system. Text comes in on the left. Four small networks from our syllabus, TextCNN, RNN, LSTM and "
             f"GRU, answer first; together they take {ms(c0)} on a CPU. A router looks only at those experts and asks: are they sure? "
             f"On AG News, {epct(1 - e_ag)} of the time they are, and we answer immediately. The other {epct(e_ag)} escalate: the frozen "
             "Laya encoder reads the text, and the ADAR head combines Laya's view with the experts' votes, shown by the dashed line, "
             "to give the final answer plus calibrated probabilities. Let's zoom into the head.")
    return section("architecture", header("How ADAR works", "System Architecture") + "\n" + "\n".join(o), notes, transition="magic", layout="abs")


def s_head():
    hp = adar("ag_news")["params_M"]
    o = [f'<div id="adarhead" style="position:absolute;left:128px;top:270px;width:1664px;height:640px;background:#FAEEE8;border:2px solid {ACC};border-radius:24px"></div>']
    lab = lambda x, w, s: f'<p style="position:absolute;left:{x}px;top:294px;width:{w}px;font-size:22px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:{MUTED}" data-build-in="fade order 1 auto">{s}</p>'
    o += [lab(168, 160, "input"), lab(346, 694, "answer options"), lab(1058, 694, "experts")]
    tok = lambda x, s, border, dashed=False: (f'<p style="position:absolute;left:{x}px;top:338px;width:160px;height:68px;line-height:64px;background:{CARD};'
                                             f'border:2px {"dashed" if dashed else "solid"} {border};border-radius:12px;{MONO};font-size:22px;font-weight:600;text-align:center" '
                                             f'data-build-in="pop order 1 auto">{s}</p>')
    o.append(tok(168, "STATE", SLATE))
    for i, s in enumerate(["world", "sports", "business", "sci/tech"]):
        o.append(tok(346 + 178 * i, s, INK, True))
    for i, s in enumerate(["CNN", "RNN", "LSTM", "GRU"]):
        o.append(tok(1058 + 178 * i, s, TAUPE))
    c = lambda x1, y1, x2, y2, order, head="": f'<x-connector x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {head}style="color:{INK};border-width:2px" data-build-in="fade order {order} auto"></x-connector>'
    o.append(c(960, 406, 960, 450, 2))
    o.append(f'<p style="position:absolute;left:460px;top:450px;width:1000px;height:90px;line-height:90px;background:{INK};color:{BG};border-radius:16px;font-size:28px;font-weight:700;text-align:center" data-build-in="rise order 2 auto">2-layer transformer: the tokens attend to each other</p>')
    o.append(c(960, 540, 960, 580, 3, 'head="none" '))
    o.append(c(440, 580, 1480, 580, 3, 'head="none" '))
    for x in (440, 960, 1480):
        o.append(c(x, 580, x, 620, 3))
    outs = [("choice", "a probability for every option: the answer", ACC), ("noul × 4", "P(this expert is right here): its routing weight α", TAUPE),
            ("act", "P(my own answer is right): abstain or escalate", SLATE)]
    for i, (n, d, b) in enumerate(outs):
        o.append(f'<div style="position:absolute;left:{200 + 520 * i}px;top:620px;width:480px;height:250px;background:{CARD};border:1px solid {LINE};border-top:6px solid {b};'
                 f'border-radius:16px;padding:28px;display:flex;flex-direction:column;gap:12px" data-build-in="rise order {4 + i} auto">'
                 f'<p style="{MONO};font-size:30px;font-weight:600">{n}</p>{para(d, 26, INK)}</div>')
    o.append(takeaway_abs(f"the head has only {hp:.1f}M parameters and answers in the same typed format as a System-One model: an answer, plus how far to trust each source."))
    notes = ("Inside the head, every input becomes a short row of tokens. One STATE token is Laya's reading of the text. Each "
             "answer option is a token built from Laya's embedding of the label's written description plus every expert's vote "
             "for it. Each expert gets a token saying how sure it is. A small two-layer transformer lets all these tokens attend "
             "to each other. Out come three typed answers, in the style of a System-One model: choice, the answer; noul for each "
             "expert, the probability that this expert is right on this input, which becomes its routing weight alpha; and act, "
             f"the probability that the head's own answer is right, which we use to abstain or escalate. The head has {hp:.1f} million parameters.")
    return section("head", header("How ADAR works", "Inside the ADAR Head") + "\n" + "\n".join(o), notes, layout="abs")


def s_training():
    cell = lambda s, pred: (f'<p style="flex:1;background:{CARD if pred else ALT};border:{"2px dashed " + INK if pred else "2px solid " + ALT};'
                            f'border-radius:8px;font-size:22px;text-align:center;padding:6px">{s}</p>')
    grid = "".join(f'<div style="display:flex;gap:10px">{"".join(cell("predict" if c == r else "train", c == r) for c in range(3))}</div>' for r in range(3))
    formula = lambda s: f'<p style="{MONO};font-size:28px;font-weight:600;line-height:1.5;background:{ALT};border-radius:12px;padding:20px">{s}</p>'
    cards = [("1 · Honest expert opinions", f'<div style="display:flex;flex-direction:column;gap:10px">{grid}</div>'
              + para("Each expert trains 3 times and only predicts the third of the data it never saw. The head learns from those honest opinions.", 26, INK)),
             ("2 · A loss that rewards honesty", formula("L = −log p_y<br>+ Σ_k (p_k − y_k)²")
              + para("Log score + Brier score. Both are lowest only when the probabilities are honest: proper scoring rules, the same idea Laya is trained with.", 26, INK)),
             ("3 · Calibrate", formula("p = softmax(z / T)")
              + para("One temperature T is fitted on validation data, so “80% sure” means right about 80% of the time.", 26, INK))]
    o = [f'<div style="position:absolute;left:{128 + 568 * i}px;top:280px;width:528px;height:620px;background:{CARD};border:1px solid {LINE};border-radius:16px;padding:36px;'
         f'display:flex;flex-direction:column;gap:22px" data-build-in="rise order {i + 1} auto">{label(t, ACCT)}{b}</div>' for i, (t, b) in enumerate(cards)]
    o.append(takeaway_abs("every training choice targets honest probabilities, not only accuracy; the router and the leaderboard work depend on that."))
    notes = ("Three ideas make the head trustworthy. One: honest expert opinions. Each expert is trained three times, each time "
             "leaving out a third of the data, and we keep only its predictions on the third it never saw; otherwise the head would "
             "learn to trust experts that memorised their training data. Two: the loss adds the log score and the Brier score. Both "
             "are proper scoring rules: the only way to minimise them is to report your true belief, the same principle behind "
             "Laya's own training. Three: after training we fit one temperature on validation data, so that 80 percent confidence "
             "really means right 80 percent of the time.")
    return section("training", header("How ADAR works", "How the Head Learns") + "\n" + "\n".join(o), notes, layout="abs")


def router_sim(p_esc):
    return ("<x-embed style=\"position:absolute;left:128px;top:280px;width:980px;height:600px\"><!doctype html><html><head><meta charset=\"utf-8\"><style>"
            f"html,body{{margin:0;height:100%;background:{CARD};overflow:hidden;border-radius:16px}}canvas{{display:block;width:100%;height:100%}}</style></head>"
            "<body><canvas id=\"c\" width=\"980\" height=\"600\"></canvas><script>\n(function(){\n"
            "var c=document.getElementById('c'),g=c.getContext('2d');\n"
            f"var INK='{INK}',EXP='{TAUPE}',LAYA='{SLATE}',ADR='{ACC}',MUT='{MUTED}',LANE='#E6DECF',BG='{CARD}',LIGHT='{BG}';\n"
            f"var P={p_esc:.4f},TOP=180,BOT=420,GX=290,END=840;\n"
            "var dots=[],t=0,nf=0,ns=0;\n"
            "function rr(x,y,w,h,r,f){g.fillStyle=f;g.beginPath();if(g.roundRect){g.roundRect(x,y,w,h,r);}else{g.rect(x,y,w,h);}g.fill();}\n"
            "function txt(s,x,y,col,size,w,al){g.fillStyle=col;g.font=w+' '+size+'px system-ui,-apple-system,\"Segoe UI\",Roboto,sans-serif';g.textAlign=al;g.textBaseline='middle';g.fillText(s,x,y);}\n"
            "function spawn(){dots.push({x:-12,y:TOP,s:Math.random()<P,v:3+Math.random()*1.5,st:0});}\n"
            "function step(d){if(d.st===0){d.x+=d.v;if(d.x>=GX){if(d.s){d.x=GX;d.st=1;}else{d.st=2;}}}"
            "else if(d.st===1){d.y+=d.v;if(d.y>=BOT){d.y=BOT;d.st=2;}}else{d.x+=d.v;if(d.x>=END){d.done=true;if(d.s){ns++;}else{nf++;}}}}\n"
            "function draw(){g.fillStyle=BG;g.fillRect(0,0,980,600);g.strokeStyle=LANE;g.lineWidth=10;g.lineCap='round';g.lineJoin='round';"
            "g.beginPath();g.moveTo(10,TOP);g.lineTo(END,TOP);g.moveTo(GX,TOP);g.lineTo(GX,BOT);g.lineTo(END,BOT);g.stroke();"
            "rr(190,TOP-50,200,100,14,EXP);txt('Experts',GX,TOP,INK,26,700,'center');"
            "rr(420,BOT-50,180,100,14,LAYA);txt('Laya',510,BOT,LIGHT,26,700,'center');"
            "rr(625,BOT-50,180,100,14,ADR);txt('ADAR',715,BOT,LIGHT,26,700,'center');"
            "for(var i=0;i<dots.length;i++){var d=dots[i];g.beginPath();g.arc(d.x,d.y,9,0,6.2832);g.fillStyle=d.st===0&&d.x<GX-60?INK:(d.s?ADR:EXP);g.fill();g.lineWidth=3;g.strokeStyle=BG;g.stroke();}"
            f"var tot=nf+ns,pf=tot?Math.round(100*nf/tot):{round(100 * (1 - p_esc))},ps=tot?100-pf:{round(100 * p_esc)};"
            "txt('answered by the experts',END,TOP-85,INK,24,600,'right');txt(pf+'%',970,TOP,INK,38,800,'right');"
            f"txt('escalated to Laya + ADAR',END,BOT+85,INK,24,600,'right');txt(ps+'%',970,BOT,'{ACCT}',38,800,'right');"
            "txt('Simulation: dots split at the rate the router chose on AG News',24,572,MUT,22,500,'left');}\n"
            "function loop(){t++;if(t%9===0){spawn();}for(var i=0;i<dots.length;i++){step(dots[i]);}dots=dots.filter(function(d){return !d.done;});draw();requestAnimationFrame(loop);}\n"
            "loop();\n})();\n</script></body></html></x-embed>")


def s_router():
    ra, re_ = router("ag_news")["chosen"], router("emotion")["chosen"]
    fa, fe = times(ra["always_tier1_cost_ms"], ra["mean_cost_ms"]), times(re_["always_tier1_cost_ms"], re_["mean_cost_ms"])
    rcard = lambda inner, border=LINE: f'<div style="display:flex;flex-direction:column;gap:8px;background:{CARD};border:1px solid {border};border-radius:16px;padding:22px 26px">{inner}</div>'
    col = (f'<div style="position:absolute;left:1148px;top:280px;width:644px;display:flex;flex-direction:column;gap:14px" data-build-in="rise order 1 auto">'
           + rcard(label("Threshold router") + para("Escalate when P(experts right) &lt; τ; τ is picked on validation data.", 24, INK))
           + rcard(label("REINFORCE router · Unit IV") + f'<p style="{MONO};font-size:24px;font-weight:600">r = 1[correct] − λ·cost</p>'
                   + para("A small policy learns the same decision by policy gradient.", 24, INK))
           + rcard(para(f"<b>AG News:</b> Laya on {epct(ra['escalation_rate'])} of inputs · {fa} cheaper · {pct(ra['accuracy'])} vs {pct(ra['always_tier1_accuracy'])} always-Laya", 24, INK)
                   + para(f"<b>Emotion:</b> Laya on {epct(re_['escalation_rate'])} · {fe} cheaper · {pct(re_['accuracy'])} vs {pct(re_['always_tier1_accuracy'])}", 24, INK), ACC)
           + para(f"Headroom: a perfect router would reach {pct(ra['oracle_accuracy'])} on AG News escalating only {epct(ra['oracle_escalation_rate'])}.", 22, MUTED)
           + '</div>')
    o = [router_sim(ra["escalation_rate"]), col,
         takeaway_abs(f"the router learns per dataset how often the big model is worth its cost: {epct(ra['escalation_rate'])} of news articles, {epct(re_['escalation_rate'])} of tweets.")]
    notes = (f"This is the cascade as a simulation. Every dot is an input. The experts answer first, and the router decides: "
             f"taupe dots are answered right away, coral dots go to Laya plus ADAR. The split matches our AG News run, where "
             f"{epct(ra['escalation_rate'])} escalate. We trained two routers: a threshold on the predicted probability that the experts "
             "are right, with the cut-off chosen on validation data, and a REINFORCE policy from Unit IV with reward one for a correct "
             f"answer minus lambda times the cost. On AG News the cascade is {fa} cheaper than always running Laya plus ADAR, at "
             f"{pct(ra['accuracy'])} instead of {pct(ra['always_tier1_accuracy'])}. On Emotion it calls Laya for {epct(re_['escalation_rate'])} of inputs and is "
             f"{fe} cheaper. A perfect router would reach {pct(ra['oracle_accuracy'])} on AG News, so there is still room to improve the router.")
    return section("router", header("How ADAR works", "The Cascade Router") + "\n" + "\n".join(o), notes, layout="abs")


def s_setup():
    sp = S["splits"]
    n_base = len(baselines("ag_news"))
    sk = sum(1 for r in baselines("ag_news") if r["family"] == "scikit-learn")
    tr = "".join(f'<tr><td>{n}</td><td>{d}</td><td style="text-align:right">{sp[k]["train"]:,}</td><td style="text-align:right">{sp[k]["val"]:,}</td>'
                 f'<td style="text-align:right">{sp[k]["test"]:,}</td><td style="text-align:right">{sp[k]["classes"]}</td></tr>'
                 for k, n, d in (("ag_news", "AG News", "news headline + first line"), ("emotion", "Emotion", "short English tweets")))
    table = (f'<table style="font-size:28px;color:{INK};background:{CARD};border-radius:12px"><tr><th style="width:20%">Dataset</th><th style="width:32%">Text</th>'
             '<th style="width:12%;text-align:right">Train</th><th style="width:12%;text-align:right">Val</th><th style="width:12%;text-align:right">Test</th>'
             f'<th style="width:12%;text-align:right">Classes</th></tr>{tr}</table>')
    stat = lambda n, t: (f'<div style="flex:1;display:flex;flex-direction:column;gap:8px;border-top:2px solid {INK};padding:20px 0 0 0">'
                         f'<p style="{HEAD};font-size:44px;font-weight:700">{n}</p>{para(t)}</div>')
    n_seeds, mins = seeds("ag_news"), S["total_minutes"]
    stats = ('<div style="display:flex;gap:32px">'
             + stat(f"{n_base} baselines", f"{sk} scikit-learn models, 4 syllabus networks and their average, Laya zero-shot, Laya probe, stacking")
             + stat("Fair tests", f"same test rows, every model calibrated on validation data, ADAR over {n_seeds} seeds, 95% bootstrap confidence intervals")
             + stat(f"{mins:.0f} minutes", "whole pipeline on Kaggle’s free GPU T4 ×2, resumable after every stage") + "</div>")
    inner = header("Results", "Experiment Setup") + "\n" + table + "\n" + stats + "\n" + takeaway(
        "every model sees the same test rows and the same calibration step, and a claim counts only if its confidence interval excludes zero.")
    notes = (f"We use two public datasets. AG News: {sp['ag_news']['train']:,} training articles in four topics, "
             f"{sp['ag_news']['val']:,} for validation and the official {sp['ag_news']['test']:,} for testing. Emotion: "
             f"{sp['emotion']['train']:,} short tweets in six emotions. We compare ADAR against {n_base} baselines: {sk} scikit-learn models on "
             "TF-IDF features, the four syllabus networks and their average, Laya answering zero-shot, a logistic regression on Laya's "
             "embeddings, and a stacking model. Every model sees the same test rows and is calibrated on validation data; ADAR is "
             f"trained with {seeds('ag_news')} seeds, and every claim needs a 95 percent bootstrap confidence interval that excludes zero. "
             f"The whole pipeline ran in {S['total_minutes']:.0f} minutes on Kaggle's free GPUs.")
    return section("setup", inner, notes, bg=ALT)


def results_slide(task, sid, title, n_test_label):
    items = sorted([(r["model"], r["accuracy"]) for r in rows(task) if not r["model"].startswith("ADAR ablation")], key=lambda x: -x[1])
    lo_v = min(v for _, v in items) * 100
    lo = max(0, math.floor((lo_v - 2) / 5) * 5)
    hi = 100
    step = 44 if len(items) <= 12 else 38
    chart, y = bars(items, lo, hi, 280, step=step, h=30 if step == 44 else 28)
    a = adar(task)
    refs = [("vs Laya zero-shot", "best Laya zero-shot"), ("vs best syllabus network", "best syllabus deep model"),
            ("vs best scikit-learn", "best scikit-learn model"), ("vs stacking", "best stacking baseline")]
    lines = []
    for lbl, needle in refs:
        g = gate(task, needle)
        if g:
            ok, d, l, h, _ = g
            mark = "significant" if ok else ("not significant" if h >= 0 else "significantly behind")
            lines.append(f'<p style="font-size:22px;line-height:1.35;color:{INK}"><b>{lbl}</b><br>{pts(d)} · CI [{l:+.2f}, {h:+.2f}] · <span style="color:{ACCT if ok else MUTED}">{mark}</span></p>')
    panel = (f'<div style="position:absolute;left:1340px;top:280px;width:452px;display:flex;flex-direction:column;gap:16px;background:{CARD};border:1px solid {LINE};border-radius:16px;padding:28px" data-build-in="rise order {len(items) + 1} auto">'
             + label("ADAR (seed 0) minus …") + "".join(lines)
             + f'<p style="font-size:22px;color:{MUTED};border-top:1px solid {LINE};padding:12px 0 0 0">{S["tasks"][task]["qc_pass"]} of {S["tasks"][task]["qc_total"]} quality checks passed</p></div>')
    how = (f'<p style="position:absolute;left:128px;top:{y + 4}px;width:1180px;font-size:20px;color:{MUTED};white-space:nowrap">How to read: test accuracy on {n_test_label}, axis from {lo}%; '
           f'ADAR = mean of {seeds(task)} seeds (± {a["accuracy_std"] * 100:.2f} pts).</p>')
    o = [chart, legend(y + 36), how, panel, takeaway_abs(verdict(task))]
    return o, items


def s_results_ag():
    o, items = results_slide("ag_news", "results-ag", "AG News Results", f"{S['splits']['ag_news']['test']:,} news articles")
    a, lz = adar("ag_news"), row("ag_news", "Laya zero-shot")
    g = gate("ag_news", "best Laya zero-shot")
    rank = [n for n, _ in items].index(a["model"]) + 1
    notes = (f"These are the AG News results on {S['splits']['ag_news']['test']:,} test articles, sorted by accuracy; the axis starts high "
             f"so the differences are visible. ADAR, averaged over {seeds('ag_news')} seeds, reaches {pct(a['accuracy'], 2)} and ranks number {rank} "
             f"of {len(items)}. Laya on its own, answering zero-shot, gets {pct(lz['accuracy'], 2)}. The panel on the right gives each "
             f"difference with its 95 percent confidence interval: against Laya zero-shot the difference is {pts(g[1])}, interval {g[2]:+.2f} to {g[3]:+.2f}. "
             + verdict("ag_news") + f" {S['tasks']['ag_news']['qc_pass']} of our {S['tasks']['ag_news']['qc_total']} quality checks passed on this dataset.")
    return section("results-ag", header("Results · test accuracy", "AG News Results") + "\n" + "\n".join(o), notes, layout="abs")


def s_honesty():
    a, cas, lz = adar("ag_news"), row("ag_news", "ADAR cascade"), row("ag_news", "Laya zero-shot")
    best = min(baselines("ag_news"), key=lambda r: r["nll"])
    best_ece = min(baselines("ag_news"), key=lambda r: r["ece_ts"])
    red = 1 - a["nll"] / best["nll"]
    mx_n = max(a["nll"], best["nll"], lz["nll"])
    mx_c = max(a["cpu_ms"], cas["cpu_ms"], lz["cpu_ms"])
    o = [f'<div style="position:absolute;left:{x}px;top:280px;width:800px;height:620px;background:{CARD};border:1px solid {LINE};border-radius:16px"></div>' for x in (128, 992)]

    def hbar(x, y, name, v, mx, col, txt, bold, order):
        w = round(v / mx * 300)
        fw = "font-weight:700;" if bold else ""
        return (f'<p style="position:absolute;left:{x}px;top:{y}px;width:300px;height:44px;line-height:44px;font-size:22px;{fw}">{e(name)}</p>'
                f'<div style="position:absolute;left:{x + 310}px;top:{y}px;width:{w}px;height:44px;background:{col};border-radius:0 6px 6px 0" data-build-in="left order {order} auto"></div>'
                f'<p style="position:absolute;left:{x + 322 + w}px;top:{y}px;width:120px;height:44px;line-height:44px;font-size:22px;{fw}" data-build-in="fade order {order} auto">{txt}</p>')
    o.append(f'<p style="position:absolute;left:168px;top:316px;width:720px;font-size:22px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:{MUTED}">Honest probabilities · log-loss (lower is better)</p>')
    o.append(f'<p style="position:absolute;left:168px;top:360px;width:720px;{HEAD};font-size:110px;font-weight:800;line-height:1;color:{ACCT}" data-build-in="pop order 1 auto">{abs(red) * 100:.0f}%</p>')
    o.append(f'<p style="position:absolute;left:168px;top:490px;width:720px;font-size:26px;line-height:1.4">{"lower" if red > 0 else "higher"} log-loss than the best baseline ({e(short(best["model"]))})</p>')
    trio = [("ADAR", a, ACC, True), (short(best["model"]), best, colour(best["model"]), False)]
    if best is not lz:
        trio.append(("Laya zero-shot", lz, SLATE, False))
    for i, (n, r, c, b) in enumerate(trio):
        o.append(hbar(168, 610 + 76 * i, n, r["nll"], mx_n, c, f"{r['nll']:.3f}", b, 2))
    o.append(f'<p style="position:absolute;left:168px;top:842px;width:720px;font-size:20px;color:{MUTED}">ECE after calibration: ADAR {a["ece_ts"]:.4f} · best baseline {best_ece["ece_ts"]:.4f} ({e(short(best_ece["model"]))})</p>')
    o.append(f'<p style="position:absolute;left:1032px;top:316px;width:720px;font-size:22px;letter-spacing:2px;text-transform:uppercase;font-weight:700;color:{MUTED}">Cost · CPU time per input</p>')
    o.append(f'<p style="position:absolute;left:1032px;top:360px;width:720px;{HEAD};font-size:110px;font-weight:800;line-height:1;color:{SLATE}" data-build-in="pop order 3 auto">{times(a["cpu_ms"], cas["cpu_ms"])}</p>')
    o.append(f'<p style="position:absolute;left:1032px;top:490px;width:720px;font-size:26px;line-height:1.4">cheaper than always running Laya + ADAR, at {pct(cas["accuracy"], 2)} vs {pct(a["accuracy"], 2)} accuracy</p>')
    for i, (n, r, c, b) in enumerate((("Laya zero-shot", lz, SLATE, False), ("Always Laya + ADAR", a, ACC, False), ("ADAR cascade", cas, ACCL, True))):
        o.append(hbar(1032, 610 + 76 * i, n, r["cpu_ms"], mx_c, c, ms(r["cpu_ms"]), b, 4))
    drop = (a["accuracy"] - cas["accuracy"]) * 100
    hon = ("ADAR has the lowest log-loss of every model, so its probabilities are the most honest" if red > 0 else
           f"ADAR’s log-loss is {-red * 100:.0f}% higher than the best baseline’s")
    trade = (f"the cascade matches its accuracy ({pct(cas['accuracy'], 2)} vs {pct(a['accuracy'], 2)}) with {times(a['cpu_ms'], cas['cpu_ms'])} less compute" if drop <= 0.05 else
             f"the cascade trades {drop:.2f} pts of accuracy for {times(a['cpu_ms'], cas['cpu_ms'])} less compute")
    o.append(takeaway_abs(f"{hon}; {trade}."))
    notes = (f"Accuracy is not the whole story. On the left is log-loss, which punishes confident mistakes; lower is better. ADAR's is "
             f"{a['nll']:.3f}, {abs(red) * 100:.0f} percent {'lower' if red > 0 else 'higher'} than the best baseline, {short(best['model'])}, at {best['nll']:.3f}, and Laya zero-shot is at "
             f"{lz['nll']:.3f}. On the right is cost, CPU time per input. Laya on its own takes {ms(lz['cpu_ms'])}. Running Laya plus ADAR on everything "
             f"takes {ms(a['cpu_ms'])}. The cascade escalates only the hard inputs and averages {ms(cas['cpu_ms'])}, {times(a['cpu_ms'], cas['cpu_ms'])} cheaper, "
             f"at {pct(cas['accuracy'])} accuracy instead of {pct(a['accuracy'])}.")
    return section("honesty-cost", header("Results · AG News", "Honesty and Cost") + "\n" + "\n".join(o), notes, layout="abs")


def s_results_em():
    o, items = results_slide("emotion", "results-emotion", "Emotion Results", f"{S['splits']['emotion']['test']:,} tweets")
    a, lz, lp, be = adar("emotion"), row("emotion", "Laya zero-shot"), row("emotion", "Laya encoder"), best_expert("emotion")
    re_ = router("emotion")["chosen"]
    notes = (f"The second dataset tells a different story. On emotion tweets Laya is weak: {pct(lz['accuracy'])} zero-shot and "
             f"{pct(lp['accuracy'])} with a trained probe on its embeddings. Our small syllabus networks do far better; the best, "
             f"{be['model']}, reaches {pct(be['accuracy'])}. ADAR reaches {pct(a['accuracy'])}. " + verdict("emotion")
             + f" {S['tasks']['emotion']['qc_pass']} of {S['tasks']['emotion']['qc_total']} checks passed. The router learned on its own that Laya rarely helps here: "
             f"it calls Laya for {epct(re_['escalation_rate'])} of tweets, giving {pct(re_['accuracy'])} at {ms(re_['mean_cost_ms'])} per input instead of {ms(re_['always_tier1_cost_ms'])}.")
    o[-1] = takeaway_abs(verdict("emotion") + f" Laya alone gets {pct(lz['accuracy'])}; the router calls it for only {epct(re_['escalation_rate'])} of tweets.")
    return section("results-emotion", header("Results · test accuracy", "Emotion Results") + "\n" + "\n".join(o), notes, layout="abs")


def s_ablation():
    variants = [("Without the experts (Laya only)", "ablation: laya only"), ("Without Laya (experts only)", "ablation: experts only"),
                ("Without attention", "ablation: no attention")]

    def cell(task, needle):
        g = gate(task, needle)
        r = row(task, "ADAR " + needle)
        if not g or not r:
            return "<td>–</td><td>–</td>"
        _, d, l, h, _ = g
        sig = "yes" if l > 0 else "no"
        return (f'<td style="text-align:right">{-d:+.2f} pts <span style="color:{MUTED}">({sig})</span></td>'
                f'<td style="text-align:right">{r["nll"]:.3f}</td>')
    trs = "".join(f"<tr><td>{n}</td>{cell('ag_news', k)}{cell('emotion', k)}</tr>" for n, k in variants)
    full = (f'<tr style="background:{HL}"><td><b>Full ADAR</b> (seed 0 for the changes)</td><td style="text-align:right"><b>{pct(adar("ag_news")["accuracy"])}</b></td>'
            f'<td style="text-align:right"><b>{adar("ag_news")["nll"]:.3f}</b></td><td style="text-align:right"><b>{pct(adar("emotion")["accuracy"])}</b></td>'
            f'<td style="text-align:right"><b>{adar("emotion")["nll"]:.3f}</b></td></tr>')
    table = (f'<table style="font-size:28px;color:{INK};background:{CARD};border-radius:12px"><tr><th style="width:36%">Variant</th>'
             '<th style="width:18%;text-align:right">AG News Δ acc</th><th style="width:14%;text-align:right">NLL</th>'
             f'<th style="width:18%;text-align:right">Emotion Δ acc</th><th style="width:14%;text-align:right">NLL</th></tr>{full}{trs}</table>')
    how = para("How to read: Δ = accuracy change when the part is removed (paired bootstrap; “yes” = the 95% CI excludes zero). NLL = log-loss, lower is better.", 22, MUTED)
    ag = {k: gate("ag_news", k) for _, k in variants}
    em = {k: gate("emotion", k) for _, k in variants}
    need = [n for n, k in variants[:2] if ag[k] and ag[k][2] > 0]
    msg = []
    if len(need) == 2:
        msg.append("on AG News both sources are needed: removing either the experts or Laya costs significant accuracy")
    elif need:
        msg.append(f"on AG News, removing {join([n.split('(')[0].strip().lower().replace('without ', '') for n in need])} costs significant accuracy")
    k = "ablation: laya only"
    if em[k] and em[k][2] > 0:
        msg.append(f"on Emotion the experts carry the result ({-em[k][1]:+.1f} pts without them)")
    k = "ablation: no attention"
    if ag[k]:
        r_na = row("ag_news", "ADAR ablation: no attention")
        if ag[k][2] > 0:
            msg.append("attention adds significant accuracy")
        elif r_na and r_na["nll"] > adar("ag_news")["nll"]:
            msg.append(f"attention mostly sharpens the probabilities (log-loss {r_na['nll']:.3f} without it, {adar('ag_news')['nll']:.3f} with it)")
        else:
            msg.append("attention adds no significant accuracy on AG News")
    inner = header("Results · ablations", "What Each Part Adds") + "\n" + table + "\n" + how + "\n" + takeaway("; ".join(msg) + ".")
    notes = ("To check that every part earns its place, we removed one part at a time and retrained. "
             + " ".join(f"{n}: on AG News accuracy changes by {-ag[k][1]:+.2f} points, on Emotion by {-em[k][1]:+.2f} points."
                        for n, k in variants if ag[k] and em[k])
             + " A change counts only when its 95 percent interval excludes zero. " + "; ".join(msg) + ".")
    return section("ablation", inner, notes)


def s_leaderboard():
    td = S["typed_decisions"]
    ft, base = td["convaiinnovations/laya-typed-decisions"], td["convaiinnovations/laya"]
    sel = lambda m: m["adar_cal_" + m["adar_cal_selected"]]
    PUB = {"accuracy": 0.766}
    t1 = (f'<table style="font-size:26px;color:{INK}"><tr><th style="width:40%">Metric</th><th style="width:30%;text-align:right">Board</th><th style="width:30%;text-align:right">Our run</th></tr>'
          f'<tr><td>Accuracy</td><td style="text-align:right">{PUB["accuracy"]:.3f}</td><td style="text-align:right"><b>{ft["shipped"]["accuracy"]:.3f}</b></td></tr>'
          f'<tr><td>Brier</td><td style="text-align:right">not listed</td><td style="text-align:right"><b>{ft["shipped"]["brier_soft"]:.3f}</b></td></tr>'
          f'<tr><td>KL</td><td style="text-align:right">not listed</td><td style="text-align:right"><b>{ft["shipped"]["kl"]:.3f}</b></td></tr>'
          f'<tr><td>ECE</td><td style="text-align:right">not listed</td><td style="text-align:right"><b>{ft["shipped"]["ece"]:.3f}</b></td></tr></table>')
    rows_ = [("Laya (base)", base["shipped"]), (f"+ ADAR-Cal ({base['adar_cal_selected']})", sel(base)),
             ("Laya typed-decisions", ft["shipped"]), (f"+ ADAR-Cal ({ft['adar_cal_selected']})", sel(ft))]
    t2 = (f'<table style="font-size:24px;color:{INK}"><tr><th style="width:40%">Model</th><th style="width:15%;text-align:right">Acc ↑</th><th style="width:15%;text-align:right">ECE ↓</th>'
          f'<th style="width:15%;text-align:right">Brier ↓</th><th style="width:15%;text-align:right">KL ↓</th></tr>'
          + "".join(f'<tr><td>{n}</td><td style="text-align:right">{m["accuracy"]:.3f}</td><td style="text-align:right">{m["ece"]:.3f}</td>'
                    f'<td style="text-align:right">{m["brier_soft"]:.3f}</td><td style="text-align:right">{m["kl"]:.3f}</td></tr>' for n, m in rows_) + "</table>")
    c1 = card(label("1 · Our scorer reproduces the board") + para(f"Laya typed-decisions · {ft['shipped']['decisions']:,} decisions", 24, MUTED) + t1)
    c2 = card(label("2 · Calibration study (fitted on train)", ACCT) + t2
              + para(f"Base Laya: ECE {base['shipped']['ece']:.3f} → {sel(base)['ece']:.3f}. Calibration never changes the chosen answer, so accuracy stays the same.", 22, BODY), border=ACC)
    inner = (header("Public benchmark · System-One models", "Typed-Decisions Benchmark") + f'\n<div style="display:flex;gap:32px">\n{c1}\n{c2}\n</div>\n'
             + takeaway(f"our scorer reproduces the board’s accuracy ({ft['shipped']['accuracy']:.3f} vs {PUB['accuracy']:.3f}), so the calibration metrics we add are comparable to it."
                        if abs(ft["shipped"]["accuracy"] - PUB["accuracy"]) < 0.005 else
                        f"our accuracy for Laya typed-decisions ({ft['shipped']['accuracy']:.3f}) differs from the board’s {PUB['accuracy']:.3f}; check the scorer before posting."))
    notes = (f"We also tested on a public System-One benchmark, typed-decisions: 400 cases and {ft['shipped']['decisions']:,} typed decisions. "
             f"First, a sanity check: for Laya's fine-tuned checkpoint our scorer gives accuracy {ft['shipped']['accuracy']:.3f}; the board lists "
             f"{PUB['accuracy']:.3f}. The board does not list its calibration, so we add it: Brier {ft['shipped']['brier_soft']:.3f}, KL "
             f"{ft['shipped']['kl']:.3f} and ECE {ft['shipped']['ece']:.3f}. Second, we applied ADAR's calibration step, per question-type "
             f"temperatures fitted only on the train split. On base Laya, calibration error goes from {base['shipped']['ece']:.3f} to "
             f"{sel(base)['ece']:.3f} and KL from {base['shipped']['kl']:.3f} to {sel(base)['kl']:.3f}, with accuracy unchanged at {base['shipped']['accuracy']:.3f}. "
             f"On the fine-tuned checkpoint the effect is mixed: ECE {ft['shipped']['ece']:.3f} to {sel(ft)['ece']:.3f}, Brier {ft['shipped']['brier_soft']:.3f} to "
             f"{sel(ft)['brier_soft']:.3f}, partly because that checkpoint was trained on the same train split.")
    return section("leaderboard", inner, notes, bg=ALT)


def s_decision():
    base = S["typed_decisions"]["convaiinnovations/laya"]["shipped"]
    hf = S.get("hf_uploads") or []
    rows_ = [("typed-decisions Discussion", "Fill in the missing Brier, KL and ECE for Laya typed-decisions, plus our calibration study",
              "Read by the System-One community (Jev, Laya and others on the board); a contribution, not a rank claim", "Submit now"),
             ("Hugging Face model pages", "Both trained heads, with results in a model-index block",
              "Anyone can load and check the heads; results show on the model page", "Publish now"),
             ("RouterArena (ICLR 2026)", "ADAR routing between LLMs instead of small networks",
              "A ranked, live leaderboard for exactly what ADAR does: the result most likely to get attention", "Next phase"),
             ("“Laya + ADAR-Cal” as a new row", f"accuracy {base['accuracy']:.3f}, below the board’s prior baseline (0.470)",
              "Would read as a weak model, although only the calibration changed", "Do not submit")]
    tr = "".join(f'<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td>'
                 f'<td style="color:{ACCT if d.endswith("now") else MUTED};font-weight:700">{d}</td></tr>' for a, b, c, d in rows_)
    table = (f'<table style="font-size:24px;color:{INK};background:{CARD};border-radius:12px"><tr><th style="width:20%">Where</th><th style="width:32%">What we submit</th>'
             f'<th style="width:34%">Why it matters</th><th style="width:14%">Decision</th></tr>{tr}</table>')
    inner = (header("Leaderboards", "Where We Submit, and Why") + "\n" + table + "\n"
             + para("The 0.470 prior is the board’s own floor (always predicting the label frequencies); the post and procedure are in submissions/.", 22, MUTED) + "\n"
             + takeaway("submit where our result is a real contribution now; RouterArena is where ADAR can earn a rank."))
    notes = ("Where should this go? We chose by asking where our real result adds value. First, the typed-decisions benchmark: "
             "the board lists Laya's fine-tuned checkpoint with accuracy only, so we post the missing calibration metrics and our "
             "calibration study as a Discussion, the way entries are added. Second, the two trained heads go on Hugging Face with "
             "their results, so anyone can check them. The head-turner is RouterArena, a ranked leaderboard for routers between "
             "large language models, which is exactly ADAR's idea at a larger scale; that is our next phase because it needs other "
             f"training data and an API budget. We deliberately do not add a 'Laya plus ADAR-Cal' row: its accuracy, {base['accuracy']:.3f}, "
             "is below the board's own prior baseline, so it would look like a weak model even though only the calibration changed.")
    return section("decision", inner, notes)


def s_syllabus():
    col = lambda u, items, b: (f'<div style="flex:1;display:flex;flex-direction:column;gap:16px;background:{CARD};border:1px solid {LINE};border-top:4px solid {b};border-radius:16px;padding:32px">'
                               f'<p style="{HEAD};font-size:40px;font-weight:800">{u}</p><ul style="font-size:26px;line-height:1.45">{"".join(f"<li>{i}</li>" for i in items)}</ul></div>')
    cols = (col("Unit I", ["TextCNN", "Backpropagation", "BatchNorm", "Gradient clipping", "AdamW + cosine schedule"], TAUPE)
            + col("Unit II", ["CNN classifiers", "Training a new head on a frozen pretrained network"], TAUPE)
            + col("Unit III", ["RNN, LSTM, GRU", "Embeddings", "Attention", "Transformer head", "BERT-family encoder"], SLATE)
            + col("Unit IV", ["Deep RL framework", "REINFORCE policy gradient", "Reward = correct − λ·cost"], ACC))
    inner = (header("Deep Learning course", "Syllabus Coverage") + f'\n<div style="display:flex;gap:24px">{cols}</div>\n'
             + takeaway("one project uses all four units together, and its goal is the course objective itself: identify the appropriate deep architecture for each application, here per input."))
    notes = ("Our project touches all four units. Unit one: the TextCNN expert, backpropagation, batch normalisation, gradient "
             "clipping, AdamW with a cosine schedule. Unit two: CNN classifiers and training a new head on a frozen pretrained "
             "network. Unit three is the core: RNN, LSTM and GRU experts, embeddings, attention, a transformer head and a "
             "BERT-family encoder. Unit four: the REINFORCE router, a policy-gradient method with reward correct minus lambda "
             "times cost. And the course objective, identifying the right architecture for an application, is what ADAR automates.")
    return section("syllabus", inner, notes)


def s_alignment():
    tr = "".join(f"<tr><td>{u}</td><td>{p}</td></tr>" for u, p in (
        ("II · CNNs in vision", "Neural Style Transfer, Fast NST, R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN"),
        ("III · RNNs, transformers", "Image captioning, GPT-2 summariser, time-series forecasting, speech recognition, machine translation, BERT classifier, RNN vs LSTM vs GRU, echo-state networks"),
        ("IV · GANs, deep RL", "GAN, DCGAN, Q-learning grid world, DQN, REINFORCE on CartPole"), ("Other", "Text-to-music generation")))
    table = (f'<table style="flex:1;font-size:24px;color:{INK};background:{CARD};border-radius:12px"><tr><th style="width:22%">Unit</th><th style="width:78%">Class projects on this unit</th></tr>{tr}'
             f'<tr style="background:{HL}"><td><b>I + II + III + IV</b></td><td><b>ADAR</b>: TextCNN and BatchNorm · new head on a frozen pretrained network · RNN, LSTM, GRU, attention, BERT-family encoder · REINFORCE router</td></tr></table>')
    side = (f'<div style="width:560px;display:flex;flex-direction:column;gap:16px">{label("Closest overlaps")}'
            + para("<b>BERT classifier:</b> ADAR uses a BERT-family encoder as one input, and reports a probe on it as a baseline.", 26, INK)
            + para("<b>RNN vs LSTM vs GRU:</b> ADAR uses all three as experts and learns per input which to trust.", 26, INK)
            + para("<b>REINFORCE:</b> ADAR applies it to a real decision, accuracy against compute cost.", 26, INK) + "</div>")
    inner = (header("Context · this course", "Where ADAR Fits") + f'\n<div style="display:flex;gap:40px">{table}{side}</div>\n'
             + takeaway("most projects implement one architecture from one unit; ADAR combines architectures from all four units and decides between them."))
    notes = ("For context, here is how the class projects map onto the syllabus. Most implement one architecture from one unit: "
             "style transfer and the R-CNN family in Unit II; captioning, translation, the BERT classifier and the RNN comparison in "
             "Unit III; GANs and the RL projects in Unit IV. ADAR draws on all four units at once. It overlaps with three projects: it "
             "uses a BERT-family encoder, but as one input among several; it uses RNN, LSTM and GRU, but as experts it chooses between "
             "per input; and it uses REINFORCE, but for a real cost-versus-accuracy decision.")
    return section("alignment", inner, notes)


def s_limits():
    em_tie = not gate("emotion", "best syllabus deep model")[0]
    noatt = gate("ag_news", "ablation: no attention")
    lim = ["Text classification only so far", f"Laya stays frozen; only the {adar('ag_news')['params_M']:.1f}M-parameter head trains"]
    if em_tie:
        lim.append(f"On Emotion, ADAR does not beat the best syllabus network ({best_expert('emotion')['model']}) significantly")
    if noatt and noatt[2] <= 0:
        lim.append("Attention’s accuracy gain is not significant")
    lim += ["ADAR-Cal fixes calibration, not accuracy", "One train/val/test protocol per dataset"]
    nxt = ["RouterArena: route between LLMs instead of small networks", "An image task (CIFAR-10) with CNN experts",
           "An echo-state network as a fifth expert", "A short arXiv write-up"]
    ul = lambda xs, c=INK: f'<ul style="font-size:26px;line-height:1.5;color:{c}">{"".join(f"<li>{e(x)}</li>" for x in xs)}</ul>'
    inner = (header("Wrap-up", "Limits and Next Steps") + f'\n<div style="display:flex;gap:32px">'
             + card(f'<p style="{HEAD};font-size:40px;font-weight:800">Limits</p>' + ul(lim))
             + card(f'<p style="{HEAD};font-size:40px;font-weight:800">Next steps</p>' + ul(nxt), border=ACC) + "</div>\n"
             + takeaway("the gains are real but bounded; the clearest next win is the same head routing between large language models."))
    notes = "We want to be clear about the limits. " + ". ".join(lim) + ". Next: " + "; ".join(nxt) + "."
    return section("limits", inner, notes)


def s_demo():
    step = lambda n, t, b: (f'<div style="flex:1;display:flex;flex-direction:column;gap:12px;border-top:2px solid {b};padding:20px 0 0 0">'
                            f'<p style="{HEAD};font-size:40px;font-weight:800">{n}</p>{para(t)}</div>')
    steps = (step("1 · Run", f"Kaggle notebook on GPU T4 ×2: trains, tests and writes the report in {S['total_minutes']:.0f} minutes", INK)
             + step("2 · Publish", ("Both heads on Hugging Face: " + " and ".join(S["hf_uploads"])) if S.get("hf_uploads") else
                    "Both trained heads, with model cards generated from this run, ready for DEV-HV/adar-head-ag-news and -emotion", INK)
             + step("3 · Use", "Load the head and classify any sentence, with honest confidence", ACC))
    code = ('from adar.inference import ADARClassifier<br>clf = ADARClassifier.from_pretrained("adar-head-ag-news")<br>'
            'clf.predict(["NASA\'s new telescope captured images of a distant galaxy cluster.",<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"Real Madrid beat Barcelona 3-1 in the Champions League semi-final on Tuesday."])')
    out = ""
    if DEMO:
        out = "".join(f'<p style="{MONO};font-size:24px;line-height:1.6;color:{ACCL}">→ {e(d["label"])} {d["p"]:.3f} · P(answer right) {d["act"]:.3f} · '
                      f'escalate: {"yes" if d["escalate"] else "no"}</p>' for d in DEMO[:2])
    box = f'<div style="display:flex;flex-direction:column;gap:12px;background:{INK};border-radius:16px;padding:36px"><p style="{MONO};font-size:24px;line-height:1.6;color:{BG}">{code}</p>{out}</div>'
    inner = header("Wrap-up", "Live Demo") + f'\n<div style="display:flex;gap:32px">{steps}</div>\n' + box
    notes = (f"Now the demo, in three steps. One: the Kaggle notebook runs the whole pipeline on free GPUs and writes the report; "
             f"it took {S['total_minutes']:.0f} minutes. Two: both trained heads are published on Hugging Face. Three: anyone can load a head "
             "and classify a sentence in three lines." + ("" if S.get("hf_uploads") else " The heads are packaged with model cards from this run, ready to upload."))
    if DEMO:
        d = DEMO[0]
        notes += (f" This is the real output from our run: the NASA sentence is classified as {d['label']} with {d['p'] * 100:.1f} percent "
                  f"probability, and the head estimates a {d['act'] * 100:.0f} percent chance that its answer is right, so it "
                  f"{'escalates' if d['escalate'] else 'does not escalate'}. Let me show it live.")
    return section("demo", inner, notes)


def s_closing():
    inner = (f'<div style="width:1100px;display:flex;flex-direction:column;gap:32px">'
             f'<h2 style="{HEAD};font-size:150px;font-weight:800;line-height:1;letter-spacing:-4px;color:{INK}">Thank You</h2>'
             f'<p style="{HEAD};font-size:44px;font-weight:600;color:{ACCT}">Questions?</p>'
             f'<p style="{MONO};font-size:24px;line-height:1.7;color:{BODY}">github.com/Harsh4-Dev/System-One-DLE' + "".join(f"<br>huggingface.co/{r}" for r in S.get("hf_uploads") or []) + '</p></div>')
    notes = "Thank you. The code, the notebook and every result from this run are on GitHub. We are happy to take questions."
    return section("closing", inner, notes).replace("justify-content:space-between", "justify-content:center")


ORDER = ["cover", "keyterms", "problem", "systemone", "idea", "architecture", "head", "training", "router", "setup",
         "results-ag", "honesty-cost", "results-emotion", "ablation", "leaderboard", "decision", "syllabus", "alignment",
         "limits", "demo", "closing"]
BUILD = {"cover": s_cover, "keyterms": s_keyterms, "problem": s_problem, "systemone": s_systemone, "idea": s_idea,
         "architecture": s_architecture, "head": s_head, "training": s_training, "router": s_router, "setup": s_setup,
         "results-ag": s_results_ag, "honesty-cost": s_honesty, "results-emotion": s_results_em, "ablation": s_ablation,
         "leaderboard": s_leaderboard, "decision": s_decision, "syllabus": s_syllabus, "alignment": s_alignment,
         "limits": s_limits, "demo": s_demo, "closing": s_closing}
SECTIONS = {"s1": {"description": "The problem and the key terms", "start": "cover"},
            "s2": {"description": "How ADAR works", "start": "systemone"},
            "s3": {"description": "Results from the full Kaggle run", "start": "setup"},
            "s4": {"description": "Leaderboards", "start": "leaderboard"},
            "s5": {"description": "Course fit, limits and the live demo", "start": "syllabus"}}


def parse_demo(path):
    if not path or not os.path.exists(path):
        return []
    txt = open(path, errors="replace").read().replace('\\"', '"').replace("\\n", "\n")
    out = []
    for m in re.finditer(r"'label': '([^']+)', 'probabilities': \{([^}]*)\}, 'p_head_correct': ([\d.]+), 'escalate': (True|False)", txt):
        probs = dict((k, float(v)) for k, v in re.findall(r"'([^']+)': ([\d.]+)", m.group(2)))
        out.append({"label": m.group(1), "p": probs.get(m.group(1), max(probs.values())), "act": float(m.group(3)),
                    "escalate": m.group(4) == "True"})
    return out


def main():
    global S, DEMO
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("project", help="the deck's project/ folder (holds deck.json and slides/)")
    ap.add_argument("--log", help="the run's notebook log, for the demo output")
    a = ap.parse_args()
    S = json.load(open(a.summary))
    DEMO = parse_demo(a.log)
    os.makedirs(os.path.join(a.project, "slides"), exist_ok=True)
    for sid in ORDER:
        with open(os.path.join(a.project, "slides", f"{sid}.html"), "w") as f:
            f.write(BUILD[sid]())
    dj = os.path.join(a.project, "deck.json")
    deck = json.load(open(dj)) if os.path.exists(dj) else {"v": 4, "lists": "css", "title": "ADAR Project Demo", "cover": "cover"}
    deck.update({"order": ORDER, "sections": SECTIONS})
    with open(dj, "w") as f:
        json.dump(deck, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(ORDER)} slides to {a.project}/slides (demo output: {'yes' if DEMO else 'no'})")


if __name__ == "__main__":
    main()
