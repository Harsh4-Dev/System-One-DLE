import json, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
S = json.load(open(sys.argv[1])); OUT = sys.argv[2]


def find(task, prefix):
    return next(r for r in S["tasks"][task]["rows"] if r["model"].startswith(prefix))


def esc_rate(task):
    rj = S["tasks"][task].get("router") or {}
    if rj.get("chosen"):
        return rj["chosen"]["escalation_rate"]
    m = re.search(r"escalates (\d+)%", find(task, "ADAR cascade")["model"])
    return int(m.group(1)) / 100


ESC = esc_rate("ag_news")
CREAM, CARD, INK, MUT = "#F5EEDC", "#FBF7EC", "#1F2233", "#5E5648"
MUS, COB, COR, GRY, CORL = "#E9A72B", "#2B59C3", "#E4572E", "#B9AE97", "#F2A285"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUT, "ytick.color": INK})

def box(ax, x, y, w, h, fc, txt, tc=INK, ec=None, fs=11, bold=True, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(ax, x1, y1, x2, y2, c=INK, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, ls=ls, mutation_scale=16))

# Figure 1: system architecture
fig, ax = plt.subplots(figsize=(11, 4.6)); fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
ax.set_xlim(0, 22.4); ax.set_ylim(0, 9.2); ax.axis("off")
box(ax, 0.2, 5.8, 2.4, 1.6, CARD, "Input\ntext", ec=INK)
box(ax, 3.4, 5.4, 5.6, 2.4, MUS, "Syllabus experts\nTextCNN · RNN\nLSTM · GRU\n≈16 ms on CPU", fs=10)
ax.add_patch(Polygon([[9.8, 6.6], [11.6, 7.7], [13.4, 6.6], [11.6, 5.5]], closed=True, fc=CARD, ec=INK, lw=1.6))
ax.text(11.6, 6.6, "Experts\nsure?", ha="center", va="center", fontweight="bold", fontsize=10)
box(ax, 15.8, 5.8, 6.2, 1.6, CARD, "Answer from the experts\n(fast path)", ec=MUS, fs=10)
box(ax, 10.0, 1.9, 3.2, 1.9, COB, "Laya encoder\nfrozen · 421M", tc=CREAM, fs=10)
box(ax, 15.8, 1.5, 6.2, 2.6, COR, "ADAR head (2.3M)\nanswer · P(answer right)\nP(each expert right)", fs=10)
arrow(ax, 2.6, 6.6, 3.4, 6.6); arrow(ax, 9.0, 6.6, 9.8, 6.6); arrow(ax, 13.4, 6.6, 15.8, 6.6)
arrow(ax, 11.6, 5.5, 11.6, 3.8); arrow(ax, 13.2, 2.85, 15.8, 2.85)
ax.text(13.7, 7.0, f"yes · {round(100 * (1 - ESC))}%", color=MUT, fontweight="bold", fontsize=9.5); ax.text(11.8, 4.6, f"no · {round(100 * ESC)}%", color=MUT, fontweight="bold", fontsize=9.5)
ax.plot([6.2, 6.2, 18.9], [5.4, 0.6, 0.6], color="#9A6B00", lw=1.8, ls="--"); arrow(ax, 18.9, 0.6, 18.9, 1.5, c="#9A6B00", ls="--")
ax.text(6.4, 0.85, "expert votes + confidence", color="#7A5500", fontsize=9.5, fontweight="bold")
ax.text(0.2, 8.7, "Split shown for AG News (from the Kaggle run)", color=MUT, fontsize=9.5)
fig.savefig(f"{OUT}/fig1_architecture.png", dpi=200, bbox_inches="tight", facecolor=CREAM); plt.close(fig)

# Figure 2: ADAR head
fig, ax = plt.subplots(figsize=(11, 5.2)); fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
ax.set_xlim(0, 22); ax.set_ylim(0, 10.4); ax.axis("off")
toks = [("STATE", COB, CREAM)] + [(t, CARD, INK) for t in ("world", "sports", "business", "sci/tech")] + [(t, MUS, INK) for t in ("CNN", "RNN", "LSTM", "GRU")]
for i, (t, fc, tc) in enumerate(toks):
    box(ax, 0.4 + i * 2.4, 7.6, 2.1, 1.2, fc, t, tc=tc, ec=INK if fc == CARD else None, ls="--" if fc == CARD else "-", fs=10)
ax.text(1.45, 9.2, "input", color=MUT, fontsize=10, fontweight="bold", ha="center")
ax.text(6.85, 9.2, "answer options", color=MUT, fontsize=10, fontweight="bold", ha="center")
ax.text(16.45, 9.2, "experts", color=MUT, fontsize=10, fontweight="bold", ha="center")
box(ax, 4.0, 4.9, 14.0, 1.5, INK, "2-layer transformer (self-attention over all tokens)", tc=CREAM, fs=10.5)
arrow(ax, 11.0, 7.6, 11.0, 6.4)
outs = [("choice\nP(label) for\nevery option", COR, INK), ("noul × 4\nP(expert e is right)\n→ routing weight α", MUS, INK),
        ("act\nP(own answer right)\n→ abstain / escalate", COB, CREAM)]
for i, (t, fc, tc) in enumerate(outs):
    box(ax, 0.6 + i * 7.2, 0.6, 6.6, 2.8, fc, t, tc=tc, fs=10); arrow(ax, 11.0, 4.9, 3.9 + i * 7.2, 3.4)
fig.savefig(f"{OUT}/fig2_head.png", dpi=200, bbox_inches="tight", facecolor=CREAM); plt.close(fig)

def color_for(m):
    if m.startswith("ADAR cascade"): return CORL
    if m.startswith("ADAR"): return COR
    if m.startswith("Laya"): return COB
    if m.startswith(("Bi-", "TextCNN", "Average")): return MUS
    return GRY

def bars(task, names, lo, hi, fname, title):
    sel = [(n, find(task, k)["accuracy"] * 100) for n, k in names]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(sel) + 1.2)); fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    y = range(len(sel))[::-1]
    for yi, (n, v) in zip(y, sel):
        ax.barh(yi, v - lo, left=lo, color=color_for(n), height=0.62)
        ax.text(v + (hi - lo) * 0.01, yi, f"{v:.1f}%", va="center", fontsize=10.5, fontweight="bold" if n.startswith("ADAR") else "normal")
    ax.set_yticks(list(y)); ax.set_yticklabels([n for n, _ in sel])
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("ADAR"): lbl.set_fontweight("bold")
    ax.set_xlim(lo, hi); ax.set_xlabel(f"test accuracy (%) — axis starts at {lo}%")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUT); ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#E3D6BA"); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    fig.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight", facecolor=CREAM); plt.close(fig)

bars("ag_news", [("ADAR (mean of 3 seeds)", "ADAR (ours"), ("ADAR cascade", "ADAR cascade"),
                 ("Laya zero-shot", "Laya zero-shot"), ("Logistic Regression (TF-IDF)", "Logistic Regression (TF-IDF)"),
                 ("Linear SVM (TF-IDF)", "Linear SVM (TF-IDF)"), ("Bi-GRU", "Bi-GRU"), ("Bi-LSTM", "Bi-LSTM"), ("TextCNN", "TextCNN"),
                 ("Bi-RNN", "Bi-RNN"), ("sklearn MLP (TF-IDF)", "sklearn MLP"),
                 ("Laya + LogReg probe", "Laya encoder"), ("Random Forest (TF-IDF)", "Random Forest (TF-IDF)")],
     85, 95, "fig3_agnews.png", "AG News (7,600 test articles)")
bars("emotion", [("Bi-LSTM", "Bi-LSTM"), ("Bi-GRU", "Bi-GRU"), ("ADAR cascade", "ADAR cascade"),
                 ("ADAR (mean of 3 seeds)", "ADAR (ours"), ("TextCNN", "TextCNN"), ("Linear SVM (TF-IDF)", "Linear SVM (TF-IDF)"),
                 ("Logistic Regression (TF-IDF)", "Logistic Regression (TF-IDF)"), ("Laya + LogReg probe", "Laya encoder"),
                 ("Laya zero-shot", "Laya zero-shot")], 50, 97, "fig4_emotion.png", "Emotion (2,000 test tweets)")

# Figure 5: honesty and cost (AG News)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.3)); fig.patch.set_facecolor(CREAM)
d1 = [("ADAR", find("ag_news", "ADAR (ours")["nll"], COR), ("Stacking (best baseline)", find("ag_news", "Stacking")["nll"], GRY),
      ("Laya zero-shot", find("ag_news", "Laya zero-shot")["nll"], COB)]
d2 = [("ADAR cascade", find("ag_news", "ADAR cascade")["cpu_ms"], CORL), ("Always Laya + ADAR", find("ag_news", "ADAR (ours")["cpu_ms"], COR),
      ("Laya zero-shot", find("ag_news", "Laya zero-shot")["cpu_ms"], COB)]
for ax, d, title, fmt in ((axs[0], d1, "Log-loss, lower = more honest", "{:.3f}"), (axs[1], d2, "CPU time per input (ms), lower = cheaper", "{:.0f} ms")):
    ax.set_facecolor(CREAM)
    for i, (n, v, c) in enumerate(d[::-1]):
        ax.barh(i, v, color=c, height=0.6); ax.text(v * 1.01, i, fmt.format(v), va="center", fontsize=10.5)
    ax.set_yticks(range(len(d))); ax.set_yticklabels([n for n, _, _ in d[::-1]])
    ax.set_xlim(0, max(v for _, v, _ in d) * 1.25); ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color="#E3D6BA"); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_honesty_cost.png", dpi=200, bbox_inches="tight", facecolor=CREAM); plt.close(fig)
print("ok")
