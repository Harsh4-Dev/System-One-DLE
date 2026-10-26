// Builds the ADAR project report (.docx). Usage: node build.js <summary.json> <figs dir> <out.docx>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, PageBreak, Footer, Header, PageNumber, TableOfContents, LevelFormat,
  Math: OMath, MathRun, MathFraction, MathSubScript, MathSuperScript, MathSubSuperScript, MathSum, MathRadical,
} = require("docx");

const [summaryPath, figDir, outPath] = process.argv.slice(2);
const S = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
const INK = "1F2233", CORAL = "B8401B", COBALT = "2B59C3", MUTED = "5E5648", CREAM = "F5EEDC", CARD = "FBF7EC";
const FONT = "Calibri";

// ---------- text helpers ----------
function runs(parts) {
  // parts: string | {t, b, i, c, code}
  return parts.map((p) => typeof p === "string" ? new TextRun({ text: p, font: FONT })
    : p.m !== undefined ? new OMath({ children: mx(p.m) })
    : new TextRun({ text: p.t, bold: p.b, italics: p.i, color: p.c, font: p.code ? "Consolas" : FONT, size: p.code ? 20 : undefined }));
}
const P = (...parts) => new Paragraph({ children: runs(parts), spacing: { after: 120, line: 300 } });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })], spacing: { before: 360, after: 160 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })], spacing: { before: 240, after: 120 } });
const Bullet = (...parts) => new Paragraph({ children: runs(parts), numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 288 } });
const Num = (...parts) => new Paragraph({ children: runs(parts), numbering: { reference: "steps", level: 0 }, spacing: { after: 80, line: 288 } });
const Caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: t, italics: true, size: 19, color: MUTED, font: FONT })], spacing: { after: 240 } });
const Note = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, color: MUTED, size: 20, font: FONT })], spacing: { after: 160 } });
const Code = (t) => new Paragraph({ children: [new TextRun({ text: t, font: "Consolas", size: 19 })], shading: { type: ShadingType.CLEAR, fill: "F3EFE6", color: "auto" }, spacing: { after: 0 } });

// ---------- math helpers: mx("f_t = σ(W_f[h_{t−1}, x_t] + b_f)") ----------
const R = (t) => new MathRun(t);
function readGroup(s, i) {
  if (s[i] === "{") { let d = 1, j = i + 1; while (j < s.length && d) { if (s[j] === "{") d++; else if (s[j] === "}") d--; j++; } return [s.slice(i + 1, j - 1), j]; }
  return [s[i], i + 1];
}
function mx(s) {
  const out = []; let buf = ""; let i = 0;
  const flush = () => { if (buf) { out.push(R(buf)); buf = ""; } };
  while (i < s.length) {
    const ch = s[i];
    if (ch === "_" || ch === "^") {
      // the base is the last character plus any combining marks after it (e.g. "p̂" = p + U+0302)
      let cut = buf.length - 1;
      while (cut > 0 && /[\u0300-\u036f]/.test(buf[cut])) cut -= 1;
      const base = buf.slice(cut); buf = buf.slice(0, cut); flush();
      let sub = null, sup = null;
      while (i < s.length && (s[i] === "_" || s[i] === "^")) {
        const kind = s[i]; const [g, j] = readGroup(s, i + 1); i = j;
        if (kind === "_") sub = g; else sup = g;
      }
      const b = [R(base)];
      if (sub !== null && sup !== null) out.push(new MathSubSuperScript({ children: b, subScript: mx(sub), superScript: mx(sup) }));
      else if (sub !== null) out.push(new MathSubScript({ children: b, subScript: mx(sub) }));
      else out.push(new MathSuperScript({ children: b, superScript: mx(sup) }));
      continue;
    }
    buf += ch; i++;
  }
  flush(); return out;
}
const flat = (x) => (typeof x === "string" ? mx(x) : x).flat(Infinity);
const Frac = (n, d) => new MathFraction({ numerator: flat(n), denominator: flat(d) });
// A full n-ary sum needs both limits in OMML; with only a lower index we write Σ with a subscript.
const Sum = (sub, sup, body) => {
  const b = flat(body);
  if (sub && sup) return new MathSum({ children: b, subScript: mx(sub), superScript: mx(sup) });
  return [sub ? new MathSubScript({ children: [R("Σ")], subScript: mx(sub) }) : R("Σ"), R(" "), ...b];
};
const Sqrt = (b) => new MathRadical({ children: mx(b) });
let eqNo = 0;
function Eq(...pieces) {
  eqNo += 1;
  const kids = pieces.flatMap((p) => typeof p === "string" ? mx(p) : Array.isArray(p) ? p : [p]);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
    children: [new OMath({ children: kids }), new TextRun({ text: `\t(${eqNo})`, font: FONT, color: MUTED })],
    tabStops: [{ type: "right", position: 9600 }] });
}

// ---------- tables ----------
const border = { style: BorderStyle.SINGLE, size: 4, color: "D9CCB0" };
const borders = { top: border, bottom: border, left: border, right: border };
function tableOf(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, head, highlight, right) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, color: "auto", fill: head ? INK : highlight ? "FBE2D6" : "FFFFFF" },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: right ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), bold: head || highlight, color: head ? CREAM : INK, size: 19, font: FONT })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true, false, i > 0 && opts.rightCols)) }),
      ...rows.map((r) => new TableRow({ children: r.cells.map((c, i) => cell(c, widths[i], false, r.hl, i > 0 && opts.rightCols)) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- figures ----------
function Fig(file, wPx, caption) {
  const buf = fs.readFileSync(path.join(figDir, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wPx, height: Math.round(wPx * h / w) },
      altText: { title: caption, description: caption, name: file } })] }), Caption(caption)];
}

// ---------- numbers from the run ----------
const T = S.tasks;
const row = (task, prefix) => T[task].rows.find((r) => r.model.startsWith(prefix));
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const AG = (p) => row("ag_news", p), EM = (p) => row("emotion", p);
const adarAG = AG("ADAR (ours"), adarEM = EM("ADAR (ours");
const td = S.typed_decisions;
const tdBase = td["convaiinnovations/laya"], tdFt = td["convaiinnovations/laya-typed-decisions"];
const calKey = tdBase.adar_cal_selected ? "adar_cal_" + tdBase.adar_cal_selected : Object.keys(tdBase).find((k) => /^adar_cal_(soft|hard)$/.test(k));
const tdCal = tdBase[calKey];
const ftCal = tdFt[tdFt.adar_cal_selected ? "adar_cal_" + tdFt.adar_cal_selected : Object.keys(tdFt).find((k) => /^adar_cal_(soft|hard)$/.test(k))];
const escRate = (task) => { const rj = T[task].router; if (rj && rj.chosen) return rj.chosen.escalation_rate;
  const m = /escalates (\d+)%/.exec(row(task, "ADAR cascade").model); return m ? Number(m[1]) / 100 : NaN; };
const escAG = Math.round(escRate("ag_news") * 100), escEM = Math.round(escRate("emotion") * 100);
const auc = (task, k) => (T[task].routing && T[task].routing[k] && T[task].routing[k].noul_auc) || null;
const qcAG = T.ag_news.qc, qcEM = T.emotion.qc;
const gate = (qc, needle) => qc.find((q) => q.includes(needle)) || "";

// ---------- content ----------
const children = [];
// Title page
children.push(
  new Paragraph({ spacing: { before: 2200, after: 200 }, children: [new TextRun({ text: "DEEP LEARNING PROJECT REPORT", bold: true, color: CORAL, size: 22, characterSpacing: 40, font: FONT })] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "ADAR: Adaptive Deep Architecture Routing", bold: true, size: 56, color: INK, font: FONT })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "A System-One decision head that learns which deep architecture to trust, and when to call the big model", size: 30, color: MUTED, font: FONT })] }),
  P({ t: "Team: ", b: true }, "[Team member names and register numbers]"),
  P({ t: "Course: ", b: true }, "[Course code] Deep Learning"),
  P({ t: "Guide: ", b: true }, "[Faculty name]"),
  P({ t: "Institution: ", b: true }, "[Department, College]"),
  P({ t: "Date: ", b: true }, "[Submission date]"),
  new Paragraph({ spacing: { before: 600 }, children: [new TextRun({ text: "Code: github.com/Harsh4-Dev/System-One-DLE  ·  Models: huggingface.co/DEV-HV/adar-head-ag-news", color: MUTED, size: 20, font: FONT })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 32, font: FONT })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  Note("If the contents list is empty, right-click it in Word and choose Update Field."),
  new Paragraph({ children: [new PageBreak()] }),
);

// Abstract
children.push(H1("Abstract"),
  P("Picking the right deep-learning architecture for an application is usually done by hand: we train a CNN, an LSTM and a GRU, keep the best one and discard the rest. Large ",
    { t: "System-One models", i: true }, " such as TypeSafe AI’s Jev and its open-source counterpart Laya answer typed questions in a single forward pass, but they are expensive, they are not always the most accurate, and their probabilities are not always honest."),
  P("We present ", { t: "ADAR (Adaptive Deep Architecture Routing)", b: true },
    ", a 2.3M-parameter decision head trained on top of the frozen Laya encoder and four networks from our syllabus (TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU). For every input, ADAR returns a calibrated answer, the probability that each expert is right, and the probability that its own answer is right. A cascade router lets the cheap experts answer first and calls Laya only when they are unsure."),
  P("On AG News (7,600 test articles), ADAR reaches ", { t: `${pct(adarAG.accuracy, 2)} accuracy`, b: true },
    `, above every baseline including Laya itself (${pct(AG("Laya zero-shot").accuracy, 2)}), with ${Math.round((1 - adarAG.nll / AG("Stacking").nll) * 100)}% lower log-loss than the best baseline; the cascade keeps ${pct(AG("ADAR cascade").accuracy)} accuracy at ${(adarAG.cpu_ms / AG("ADAR cascade").cpu_ms).toFixed(1)}× lower CPU cost. On Emotion tweets, where Laya is weak (${pct(EM("Laya zero-shot").accuracy)}), ADAR ties the best syllabus network (${pct(adarEM.accuracy)}) and the router learns to call Laya for only ${escEM}% of inputs. On the public typed-decisions benchmark, our calibration (ADAR-Cal) lowers Laya’s expected calibration error from ${tdBase.shipped.ece.toFixed(3)} to ${tdCal.ece.toFixed(3)}. All numbers come from one full run of our notebook on Kaggle (GPU T4 ×2, ${Math.round(S.total_minutes)} minutes).`),
);

// 1 Introduction
children.push(H1("1. Introduction"), H2("1.1 The problem"),
  P("Three measurements from our own run motivate the project:"),
  Bullet({ t: "The biggest model is not always right. ", b: true }, `Laya (421M parameters) answers Emotion tweets with ${pct(EM("Laya zero-shot").accuracy)} accuracy, while a 1.2M-parameter Bi-LSTM reaches ${pct(EM("Bi-LSTM").accuracy)}.`),
  Bullet({ t: "Big models are slow. ", b: true }, `On a CPU, Laya needs ${AG("Laya zero-shot").cpu_ms.toFixed(0)} ms per input; a TextCNN needs ${AG("TextCNN").cpu_ms.toFixed(1)} ms, about ${Math.round(AG("Laya zero-shot").cpu_ms / AG("TextCNN").cpu_ms)} times less.`),
  Bullet({ t: "Confidence is not always honest. ", b: true }, `Laya’s fine-tuned checkpoint has an expected calibration error (ECE) of ${tdFt.shipped.ece.toFixed(3)} on its own leaderboard: its probabilities are off by about 20 points.`),
  P("Our course asks us to ", { t: "“analyse and identify the appropriate deep architecture for a given application.”", i: true }, " ADAR turns that manual choice into a learned, per-input decision."),
  H2("1.2 Contributions"),
  Bullet("A ", { t: "System-One decision head", b: true }, " that fuses a frozen System-One encoder with syllabus networks through attention, and outputs typed, calibrated decisions (choice, per-expert noul, act)."),
  Bullet("A ", { t: "cascade router", b: true }, " (threshold and REINFORCE policy gradient) that spends the expensive model only on hard inputs."),
  Bullet("A ", { t: "fair evaluation", b: true }, " against 12 baselines on two datasets with temperature-scaled probabilities, three seeds and paired-bootstrap confidence intervals."),
  Bullet({ t: "ADAR-Cal", b: true }, ", a train-split calibration of Laya evaluated on the public typed-decisions benchmark, with a scorer that reproduces the published leaderboard numbers."),
);

// 2 Background
children.push(H1("2. Background"), H2("2.1 System-One models"),
  P("A System-One model (named after Kahneman’s fast, intuitive System 1) reads a ", { t: "state", i: true }, " (text or JSON) and a set of ", { t: "typed questions", i: true },
    ", and answers all of them in one forward pass without generating text. The three question types are ", { t: "choice", b: true }, " (pick one option), ", { t: "score", b: true }, " (a level on an ordered scale) and ",
    { t: "noul", b: true }, " (a yes/no probability). TypeSafe AI introduced the first such model, Jev; Laya by Convai Innovations is the open-source alternative (Apache-2.0): a ModernBERT-large encoder (395M) plus a two-layer decision head, 421M parameters in total."),
  P("Laya places a mask marker before every option, encodes question, options and state together, and scores each option from the encoder’s hidden state at its marker. With a per-type temperature T:"),
  Eq("p(k | x,q) = ", Frac("exp(s_k / T)", [Sum("j", null, "exp(s_j / T)")])),
  P("Laya is trained with reinforcement learning against strictly proper scoring rules (RLCD), so that the reported probabilities are honest."),
  H2("2.2 The syllabus experts"),
  P({ t: "TextCNN (Unit I–II).", b: true }, " Word embeddings ", { m: "x_1 … x_n" }, " are convolved with filters of width ", { m: "h ∈ {3, 4, 5}" }, ", batch-normalised and max-pooled over time:"),
  Eq("c_i = ReLU(BN(W ∗ x_{i:i+h−1} + b)),   ĉ = max_i c_i"),
  P({ t: "Recurrent experts (Unit III).", b: true }, " A vanilla RNN updates ", { m: "h_t = tanh(W x_t + U h_{t−1} + b)" }, " and suffers from vanishing gradients on long sequences. The LSTM adds gates and a memory cell:"),
  Eq("f_t = σ(W_f[h_{t−1}, x_t] + b_f),  i_t = σ(W_i[h_{t−1}, x_t] + b_i),  o_t = σ(W_o[h_{t−1}, x_t] + b_o)"),
  Eq("c_t = f_t ⊙ c_{t−1} + i_t ⊙ tanh(W_c[h_{t−1}, x_t] + b_c),   h_t = o_t ⊙ tanh(c_t)"),
  P("The GRU merges the gates into an update gate ", { m: "z_t" }, " and a reset gate ", { m: "r_t" }, ":"),
  Eq("h_t = (1 − z_t) ⊙ h_{t−1} + z_t ⊙ tanh(W[r_t ⊙ h_{t−1}, x_t])"),
  P("All three recurrent experts are bidirectional; their output is the concatenation of mean- and max-pooling over valid time steps."),
  H2("2.3 Calibration and proper scoring rules"),
  P("A classifier is ", { t: "calibrated", i: true }, " if, among inputs where it reports confidence c, it is right a fraction c of the time. For N test examples with predicted distribution ", { m: "p_i" }, " and one-hot label ", { m: "y_i" }, " we report:"),
  Eq("NLL = −", Frac("1", "N"), Sum("i", null, "log p_{i,y_i}"), ",    Brier = ", Frac("1", "N"), Sum("i", null, [Sum("k", null, "(p_{ik} − y_{ik})^2")])),
  Eq("ECE = ", Sum("b=1", "15", [Frac("|B_b|", "N"), ...mx(" |acc(B_b) − conf(B_b)|")])),
  P("where ", { m: "B_b" }, " are 15 equal-width confidence bins. NLL (log score) and the Brier score are ", { t: "strictly proper scoring rules", b: true },
    ": their expected value is minimised only by reporting the true probabilities, so they reward honesty as well as accuracy. Temperature scaling fits one scalar T on validation data and replaces p = softmax(z) by softmax(z / T); it changes confidence but never the predicted class."),
);

// 3 Method
children.push(new Paragraph({ children: [new PageBreak()] }), H1("3. Method: ADAR"), H2("3.1 Overview"),
  P("Figure 1 shows the pipeline. Four syllabus experts answer first (≈16 ms on a CPU in total). A router reads only the experts’ signals and predicts whether they are right. If so, the experts’ answer is returned; otherwise the input escalates: the frozen Laya encoder embeds the text and the ADAR head combines that embedding with the experts’ votes."),
  ...Fig("fig1_architecture.png", 620, "Figure 1. ADAR system architecture. Percentages are the AG News routing split from the run."),
  H2("3.2 Honest expert opinions (out-of-fold training)"),
  P("If the head learned from an expert’s predictions on that expert’s own training data, it would learn to trust over-fitted experts. We therefore split the training pool into K = 3 stratified folds and train every expert three times. Example i in fold k(i) only receives predictions from the model that did not see it:"),
  Eq("p̂_e(x_i) = f_e^{(−k(i))}(x_i)"),
  P("At validation and test time the three fold models are averaged. This is stacked generalisation (Wolpert, 1992) applied to deep networks."),
  H2("3.3 The ADAR head"),
  P("Every input becomes a short sequence of 1 + K + E tokens (K answer options, E = 4 experts), projected to d = 256 dimensions. Let z(x) be Laya’s pooled embedding (mean of the last hidden layer concatenated with the first token, 2,048 values), ", { m: "e_k" }, " Laya’s embedding of option k’s written description, and ", { m: "p̂_{ek}" }, " expert e’s probability for option k:"),
  Eq("t_{state} = W_s z(x),    t_k = W_o e_k + W_v[log p̂_{1k}, …, log p̂_{Ek}, p̂_{1k}, …, p̂_{Ek}]"),
  Eq("t_e = W_f φ(p̂_e) + u_e,    φ(p̂_e) = [max_k p̂_{ek}, margin, entropy, agreement]"),
  P("Type embeddings mark the three token kinds. A two-layer pre-norm transformer encoder (4 heads) mixes the tokens: ", { m: "H = Transformer([t_{state}; t_1 … t_K; t_1 … t_E])" }, ". The choice logits combine three paths: a scorer on each option token, a bilinear match between the input and the option, and a per-input weighted vote of the experts:"),
  Eq("ℓ_k = g(h_k) + ", Frac("(U t_{state})^⊤(V t_k)", [Sqrt("d")]), " + ", Sum("e=1", "E", "α_e(x) log p̂_{ek}"), ",    α(x) = softplus(W_α h_{state})"),
  P("The routing weights ", { m: "α_e(x)" }, " are the “adaptive routing” in ADAR: the head decides, for each input, how much each architecture’s vote counts. Three typed outputs follow, in the style of a System-One model:"),
  Eq("p = softmax(ℓ / T),    q_e = σ(w_n^⊤ h_e),    a = σ(MLP([h_{state}, top-1(p), margin(p), entropy(p)]))"),
  P("Here p is the ", { t: "choice", b: true }, " answer, ", { m: "q_e" }, " the ", { t: "noul", b: true }, " probability that expert e is right on this input, and a the ", { t: "act", b: true }, " probability that the head’s own answer is right. Figure 2 shows the structure. During training each expert’s evidence is dropped with probability 0.15 so the head stays robust when an expert is missing."),
  ...Fig("fig2_head.png", 600, "Figure 2. Inside the ADAR head: tokens, a two-layer transformer and three typed outputs."),
  H2("3.4 Training objective"),
  P("The head minimises two proper scoring rules for the answer plus binary cross-entropies for the auxiliary outputs, where ", { m: "ŷ_e" }, " is expert e’s predicted class and ŷ the head’s:"),
  Eq("L = −log p_y + ", Sum("k", null, "(p_k − y_k)^2"), " + 0.2", Sum("e", null, "BCE(q_e, 1[ŷ_e = y])"), " + 0.1 BCE(a, 1[ŷ = y])"),
  P("After early stopping on validation NLL, one temperature is fitted on the validation set:"),
  Eq("T^* = arg min_T NLL_{val}(softmax(ℓ / T))"),
  H2("3.5 The cascade router"),
  P("Tier 0 is the average of the four experts (temperature-scaled); tier 1 is Laya + ADAR. From the experts’ features f(x) (top probability, margin, entropy, agreement, each expert’s top probability) a logistic regression predicts r(x) = P(tier 0 is correct). The ", { t: "threshold router", b: true }, " escalates when r(x) < τ, with τ the smallest threshold whose validation accuracy is within 0.2 points of always escalating."),
  P("The ", { t: "REINFORCE router", b: true }, " (Unit IV) treats escalation as an action ", { m: "a ∈ {0, 1}" }, " of a stochastic policy ", { m: "π_θ(a | f)" }, " and maximises the expected reward, with c the cost share of the expensive path:"),
  Eq("R = 1[final answer correct] − λ c a,    ∇_θ J ≈ (R − b) ∇_θ log π_θ(a | f(x))"),
  P("b is a moving-average baseline that reduces the gradient variance. Sweeping λ traces an accuracy–cost curve."),
  H2("3.6 ADAR-Cal for the typed-decisions benchmark"),
  P("For typed decisions we keep Laya’s answers and only recalibrate: one temperature per (question type t, number of options K), fitted on the benchmark’s train split. The soft objective matches the gold probability distribution ", { m: "g_d" }, "; the hard objective uses its top label. The variant is chosen on the train split only:"),
  Eq("T^*_{t,K} = arg min_T ", Sum("d ∈ train(t,K)", null, [...mx("−"), Sum("k", null, "g_{dk} log softmax(z_d / T)_k")])),
);

// 4 Setup
const sp = S.splits;
children.push(new Paragraph({ children: [new PageBreak()] }), H1("4. Experimental setup"), H2("4.1 Datasets"),
  tableOf(["Dataset", "Text", "Train pool", "Validation", "Test", "Classes"], [
    { cells: ["AG News", "news headline + first sentence", sp.ag_news.train.toLocaleString("en-US"), sp.ag_news.val.toLocaleString("en-US"), sp.ag_news.test.toLocaleString("en-US"), sp.ag_news.classes] },
    { cells: ["Emotion (dair-ai)", "short English tweets", sp.emotion.train.toLocaleString("en-US"), sp.emotion.val.toLocaleString("en-US"), sp.emotion.test.toLocaleString("en-US"), sp.emotion.classes] },
    { cells: ["typed-decisions", "tickets, invoices, incidents", "1,200 cases", "–", "400 cases (2,000 decisions)", "mixed"] },
  ], [1800, 2900, 1300, 1200, 1500, 930], { rightCols: true }),
  gap(),
  P("AG News has no official validation split, so 6,000 articles are held out from its training set. The official test sets are used exactly once, for the final tables."),
  H2("4.2 Baselines"),
  Bullet({ t: "scikit-learn on TF-IDF (word 1–2-grams, 200k features): ", b: true }, "Multinomial Naive Bayes, Logistic Regression, Linear SVM (Platt-calibrated), Random Forest (200 trees, 30k rows), MLPClassifier (256 hidden, chi² top-20k features, 40k rows)."),
  Bullet({ t: "Syllabus networks: ", b: true }, "TextCNN, Bi-RNN, Bi-LSTM, Bi-GRU, and their average."),
  Bullet({ t: "Laya-based: ", b: true }, "Laya zero-shot (a typed choice question with the label descriptions), logistic regression on Laya embeddings (probe), and stacking (logistic regression on Laya embeddings plus the experts’ out-of-fold log-probabilities)."),
  H2("4.3 Implementation details"),
  tableOf(["Component", "Setting"], [
    { cells: ["Experts", "embedding 128, 128 filters (3/4/5) or 128 hidden units per direction; max length 64 (AG News) / 48 (Emotion); vocabulary 30k"] },
    { cells: ["Expert training", "AdamW, lr 2e-3, weight decay 1e-4, batch 256, ≤10 epochs, early stopping (patience 2) on validation NLL, gradient clipping at 1.0"] },
    { cells: ["Laya encoder", "convaiinnovations/laya, frozen, fp16 on GPU, max 128 (AG News) / 64 (Emotion) tokens, sharded cache on both GPUs"] },
    { cells: ["ADAR head", "d = 256, 2 layers, 4 heads, feed-forward 512, dropout 0.1, 2.3M parameters"] },
    { cells: ["Head training", "AdamW, lr 1e-3, weight decay 0.01, batch 512, ≤30 epochs, 5% warm-up + cosine, early stopping (patience 5), expert dropout 0.15, seeds 0/1/2"] },
    { cells: ["Hardware", `Kaggle GPU T4 ×2; whole pipeline ${Math.round(S.total_minutes)} minutes; CPU latencies are batch-1 medians on the Kaggle CPU`] },
  ], [2200, 7430]),
  gap(),
  H2("4.4 Evaluation protocol"),
  P("Every model’s probabilities are temperature-scaled on the validation split and evaluated on the same test rows. We report accuracy, macro-F1, NLL, Brier, ECE (15 bins) and CPU latency. ADAR is trained with three seeds and reported as mean ± standard deviation. A claim “A beats B” counts only if the 95% confidence interval of the accuracy difference, from 2,000 paired bootstrap resamples of the test set, excludes zero:"),
  Eq("Δ = acc(A) − acc(B),    CI_{95} = [Δ_{2.5%}, Δ_{97.5%}] over paired resamples"),
);

// 5 Results
const agRows = [
  ["ADAR (ours, 3 seeds)", adarAG, true], [`ADAR cascade (${escAG}% to Laya)`, AG("ADAR cascade"), true], ["Laya zero-shot", AG("Laya zero-shot"), false],
  ["Logistic Regression (TF-IDF)", AG("Logistic Regression"), false], ["Linear SVM (TF-IDF)", AG("Linear SVM"), false], ["Bi-GRU", AG("Bi-GRU"), false],
  ["Bi-LSTM", AG("Bi-LSTM"), false], ["Stacking (Laya + experts)", AG("Stacking"), false], ["TextCNN", AG("TextCNN"), false], ["Random Forest (TF-IDF)", AG("Random Forest"), false]];
const fmtRow = ([n, r, hl]) => ({ hl, cells: [n, pct(r.accuracy, 2) + (r.accuracy_std ? ` ± ${(r.accuracy_std * 100).toFixed(2)}` : ""), r.nll.toFixed(3), r.brier.toFixed(3), r.ece_ts.toFixed(3), r.cpu_ms.toFixed(1)] });
children.push(new Paragraph({ children: [new PageBreak()] }), H1("5. Results"), H2("5.1 AG News"),
  ...Fig("fig3_agnews.png", 590, "Figure 3. AG News test accuracy (7,600 articles). Axis starts at 85%."),
  tableOf(["Model", "Accuracy", "NLL ↓", "Brier ↓", "ECE ↓", "CPU ms"], agRows.map(fmtRow), [3300, 1500, 1100, 1100, 1100, 1530], { rightCols: true }),
  gap(),
  P(`ADAR is the most accurate model and passes all ${T.ag_news.qc_pass} of ${T.ag_news.qc_total} quality checks. Selected confidence intervals (seed 0):`),
  Bullet(gate(qcAG, "Laya zero-shot").replace("PASS — ", "")),
  Bullet(gate(qcAG, "best scikit-learn").replace("PASS — ", "")),
  Bullet(gate(qcAG, "best syllabus").replace("PASS — ", "")),
  H2("5.2 Honesty and cost"),
  ...Fig("fig5_honesty_cost.png", 620, "Figure 4. AG News: log-loss (left) and CPU time per input (right)."),
  P(`ADAR’s log-loss (${adarAG.nll.toFixed(3)}) is ${Math.round((1 - adarAG.nll / AG("Stacking").nll) * 100)}% lower than the best baseline’s, so its probabilities are the most honest of all models. The cascade calls Laya for ${escAG}% of inputs and averages ${AG("ADAR cascade").cpu_ms.toFixed(0)} ms per input instead of ${adarAG.cpu_ms.toFixed(0)} ms (${(adarAG.cpu_ms / AG("ADAR cascade").cpu_ms).toFixed(1)}× cheaper) while keeping ${pct(AG("ADAR cascade").accuracy)} accuracy.`),
  H2("5.3 Emotion"),
  ...Fig("fig4_emotion.png", 560, "Figure 5. Emotion test accuracy (2,000 tweets). Axis starts at 50%."),
  P(`Laya is weak on emotion tweets (${pct(EM("Laya zero-shot").accuracy)} zero-shot, ${pct(EM("Laya encoder").accuracy)} with a probe), consistent with the 57.3% reported on Laya’s own model card. ADAR (${pct(adarEM.accuracy)} ± ${(adarEM.accuracy_std * 100).toFixed(2)}) beats every scikit-learn model (best: Linear SVM, ${pct(EM("Linear SVM").accuracy)}) and Laya by 34 points, but only ties the best syllabus network (Bi-LSTM, ${pct(EM("Bi-LSTM").accuracy)}); the difference is not significant, and ${T.emotion.qc_total - T.emotion.qc_pass} of ${T.emotion.qc_total} checks fail for that reason. The router learned that Laya does not help here: it escalates only ${escEM}% of tweets, giving ${pct(EM("ADAR cascade").accuracy)} accuracy at ${EM("ADAR cascade").cpu_ms.toFixed(1)} ms instead of ${adarEM.cpu_ms.toFixed(0)} ms per input.`),
  H2("5.4 Ablations (AG News)"),
  tableOf(["Variant", "Accuracy", "Change", "Significant?"], [
    { hl: true, cells: ["Full ADAR (mean of 3 seeds)", pct(adarAG.accuracy, 2), "–", "–"] },
    { cells: ["Without the experts (Laya only)", pct(AG("ADAR ablation: laya only").accuracy, 2), `${((AG("ADAR ablation: laya only").accuracy - adarAG.accuracy) * 100).toFixed(1)} pts`, "yes"] },
    { cells: ["Without Laya (experts only)", pct(AG("ADAR ablation: experts only").accuracy, 2), `${((AG("ADAR ablation: experts only").accuracy - adarAG.accuracy) * 100).toFixed(1)} pts`, "yes"] },
    { cells: ["Without attention (no transformer)", pct(AG("ADAR ablation: no attention").accuracy, 2), `${((AG("ADAR ablation: no attention").accuracy - adarAG.accuracy) * 100).toFixed(1)} pts`, "no"] },
  ], [4100, 1700, 1700, 2130], { rightCols: true }),
  gap(),
  P(`Both information sources are needed. Attention adds little accuracy but improves the probabilities (log-loss ${AG("ADAR ablation: no attention").nll.toFixed(3)} without it, ${adarAG.nll.toFixed(3)} with it).`),
  H2("5.5 typed-decisions benchmark"),
  P("First we checked that our scorer matches the leaderboard by evaluating the published fine-tuned checkpoint:"),
  tableOf(["laya-typed-decisions", "Published", "Our run"], [
    { cells: ["Accuracy", "0.766", tdFt.shipped.accuracy.toFixed(4)] }, { cells: ["Brier (vs gold distribution)", "0.062", tdFt.shipped.brier_soft.toFixed(4)] },
    { cells: ["ECE (10 bins)", "0.213", tdFt.shipped.ece.toFixed(4)] }], [4600, 2500, 2530], { rightCols: true }),
  gap(),
  P("Our entry, base Laya + ADAR-Cal (temperatures fitted on the train split):"),
  tableOf(["Metric (2,000 test decisions)", "Laya", "Laya + ADAR-Cal", "Change"], [
    { cells: ["ECE ↓", tdBase.shipped.ece.toFixed(3), tdCal.ece.toFixed(3), `${Math.round((tdCal.ece / tdBase.shipped.ece - 1) * 100)}%`] },
    { cells: ["Brier ↓", tdBase.shipped.brier_soft.toFixed(3), tdCal.brier_soft.toFixed(3), `${Math.round((tdCal.brier_soft / tdBase.shipped.brier_soft - 1) * 100)}%`] },
    { cells: ["KL ↓", tdBase.shipped.kl.toFixed(3), tdCal.kl.toFixed(3), `${Math.round((tdCal.kl / tdBase.shipped.kl - 1) * 100)}%`] },
    { cells: ["Accuracy ↑", tdBase.shipped.accuracy.toFixed(3), tdCal.accuracy.toFixed(3), "unchanged"] }], [3600, 1900, 2300, 1830], { rightCols: true }),
  gap(),
  P(`Calibration never changes the chosen answer, so accuracy is unchanged while the probability metrics move. Applying the same procedure to the fine-tuned checkpoint changed its ECE from ${tdFt.shipped.ece.toFixed(3)} to ${ftCal.ece.toFixed(3)} and its Brier score from ${tdFt.shipped.brier_soft.toFixed(3)} to ${ftCal.brier_soft.toFixed(3)}: on soft gold labels, sharpening the top answer (lower ECE) can move the distribution away from the gold (higher Brier). That checkpoint was also trained on the same train split, so its train-fitted temperatures are biased.`),
);

// 6 Discussion
children.push(H1("6. Discussion"),
  Bullet({ t: "When does Laya help? ", b: true }, `On news topics Laya’s encoder carries information the experts miss (${((adarAG.accuracy - AG("ADAR ablation: experts only").accuracy) * 100).toFixed(1)} points over the experts-only head). On tweets it does not; the router detects this and stops paying for it. The same head therefore adapts its cost to the dataset.`),
  Bullet({ t: "Why attention adds little. ", b: true }, "With 9–11 tokens, the bilinear path and the expert vote already capture most interactions; attention mainly sharpens the probabilities. Larger option sets or more experts may change this."),
  Bullet({ t: "Calibration is not accuracy. ", b: true }, "On typed-decisions the gold answers are soft distributions. Making the top answer more confident lowers ECE but can move the probabilities away from the gold distribution (higher Brier). Reporting both metrics avoids over-claiming."),
  Bullet({ t: "Honest per-expert reliability. ", b: true }, T.ag_news.routing
    ? `On AG News the noul outputs separate each expert’s right and wrong answers with AUCs of ${["textcnn", "rnn", "lstm", "gru"].map((k) => auc("ag_news", k)).filter((v) => v).map((v) => v.toFixed(3)).join(", ")} (TextCNN, RNN, LSTM, GRU); this per-input reliability is what makes routing possible.`
    : "The noul outputs separate each expert’s right and wrong answers well above chance (see the run report); this per-input reliability is what makes routing possible."),
);

// 7 Syllabus
children.push(H1("7. Mapping to the Deep Learning syllabus"),
  tableOf(["Unit", "Topics used in ADAR"], [
    { cells: ["I", "Neural network training, backpropagation, initialisation (gate initialised to equal expert weights), vanishing/exploding gradients (gradient clipping), gradient-descent strategies (AdamW, warm-up + cosine schedule), batch normalisation, CNN structure and training (TextCNN)"] },
    { cells: ["II", "CNN classifiers; training a new head on a frozen pretrained network (fine-tuning)"] },
    { cells: ["III", "RNN architecture and its challenges, LSTM, GRU, embeddings, attention, transformers, a BERT-family encoder (ModernBERT inside Laya)"] },
    { cells: ["IV", "Deep reinforcement learning framework (state, action, reward), policy-gradient methods (REINFORCE with a baseline); routing as a one-step contextual decision"] },
  ], [900, 8730]),
  gap(),
);

// 8 Limitations
children.push(H1("8. Limitations and future work"),
  Bullet("Only text classification (AG News, Emotion) is covered; vision tasks (Unit II) are future work."),
  Bullet("Laya is frozen; only the 2.3M-parameter head is trained."),
  Bullet("On Emotion, ADAR ties rather than beats the best expert; attention’s accuracy gain is not significant."),
  Bullet("ADAR-Cal improves calibration on typed-decisions but not accuracy."),
  Bullet("Future work: RouterArena (routing between large language models with the same per-expert noul idea), an image task with CNN experts, an echo-state-network expert, and a short arXiv paper."),
);

// 9 Reproducibility
children.push(H1("9. Reproducibility"),
  Num("Import notebooks/adar_kaggle.ipynb into Kaggle; set Accelerator to GPU T4 ×2 and Internet on."),
  Num("Save Version → Save & Run All. Every stage checkpoints; a re-run skips finished work."),
  Num("Download adar_outputs.zip: results/ (tables, figures, quality checks), publish/ (trained heads), typed_decisions/ (benchmark files)."),
  Num("Regenerate the leaderboard entry and model cards: python tools/make_submissions.py."),
  P("Using a published head:"),
  Code("from adar.inference import ADARClassifier"),
  Code("clf = ADARClassifier.from_pretrained(\"adar-head-ag-news\")"),
  Code("clf.predict([\"NASA's new telescope captured images of a distant galaxy cluster.\"])"),
  Code("# -> sci_tech 0.948, P(answer right) 0.973, escalate: False   (real output from the run)"),
  gap(),
);

// References
children.push(H1("References"),
  ...[
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1).",
    "Cho, K. et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. EMNLP.",
    "Convai Innovations (2026). Laya: open-source System-One decision model. huggingface.co/convaiinnovations/laya.",
    "Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.",
    "Efron, B., Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.",
    "Gneiting, T., Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477).",
    "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.",
    "Hochreiter, S., Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).",
    "Ioffe, S., Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML.",
    "Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.",
    "Kim, Y. (2014). Convolutional neural networks for sentence classification. EMNLP.",
    "LocalLLaMA (2026). typed-decisions benchmark. huggingface.co/datasets/LocalLLaMA/typed-decisions.",
    "Loshchilov, I., Hutter, F. (2019). Decoupled weight decay regularization. ICLR.",
    "Saravia, E. et al. (2018). CARER: Contextualized affect representations for emotion recognition. EMNLP.",
    "TypeSafe AI (2026). Introducing System One models and Jev. typesafe.ai/blog.",
    "Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.",
    "Warner, B. et al. (2024). Smarter, better, faster, longer: A modern bidirectional encoder (ModernBERT). arXiv:2412.13663.",
    "Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8.",
    "Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2).",
    "Zhang, X., Zhao, J., LeCun, Y. (2015). Character-level convolutional networks for text classification. NeurIPS.",
  ].map((r) => new Paragraph({ children: [new TextRun({ text: r, size: 20, font: FONT })], spacing: { after: 80 }, indent: { left: 360, hanging: 360 } })),
);

// Appendix: full tables
const fullTable = (task) => tableOf(["Model", "Acc.", "Macro-F1", "NLL", "Brier", "ECE", "CPU ms"],
  T[task].rows.map((r) => ({ hl: r.model.startsWith("ADAR (ours") || r.model.startsWith("ADAR cascade"),
    cells: [r.model, pct(r.accuracy, 2), r.macro_f1.toFixed(3), r.nll.toFixed(3), r.brier.toFixed(3), r.ece_ts.toFixed(3), r.cpu_ms.toFixed(1)] })),
  [3430, 1000, 1050, 1000, 1000, 1000, 1150], { rightCols: true });
children.push(new Paragraph({ children: [new PageBreak()] }), H1("Appendix A. Full result tables"),
  Note("All values from the full Kaggle run (results/kaggle_run/summary.json). Probabilities temperature-scaled on validation data; ECE after scaling."),
  H2("A.1 AG News"), fullTable("ag_news"), gap(), H2("A.2 Emotion"), fullTable("emotion"), gap(),
  H2("A.3 Quality checks"),
  ...qcAG.map((q) => Bullet("AG News: " + q)), ...qcEM.map((q) => Bullet("Emotion: " + q)),
);

const doc = new Document({
  creator: "ADAR project team", title: "ADAR: Adaptive Deep Architecture Routing", features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: INK, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: CORAL, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "ADAR — Deep Learning project report", color: MUTED, size: 18, font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], color: MUTED, size: 18, font: FONT })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes"); });
