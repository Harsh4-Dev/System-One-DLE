// Builds an editable PowerPoint deck (native shapes, charts and tables, speaker notes) from one run.
// Usage: node build_pptx.js <summary.json> <deck slides dir (for notes/takeaways)> <kernel.log> <out.pptx>
// Every number is read from summary.json; notes and "What this shows" lines come from the HTML deck,
// which tools/report/build_slides.py generates from the same summary.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const [summaryPath, slidesDir, logPath, outPath] = process.argv.slice(2);
const S = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
const T = S.tasks;

// ---------- palette (clean: cream, ink, one accent) ----------
const THEME = {
  name: "ADAR", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2233", lt1: "FCFAF5", dk2: "3F4250", lt2: "F7F3EA", accent1: "C0503A", accent2: "4A5874",
    accent3: "A79E8C", accent4: "CFC7B6", accent5: "DE9C8C", accent6: "6B6E78", hlink: "A8432F", folHlink: "6B6E78" },
};
const K = { bg: "F7F3EA", alt: "EFE8DA", card: "FCFAF5", line: "DDD3C0", ink: "1F2233", body: "3F4250", muted: "6B6E78",
  acc: "C0503A", acct: "A8432F", hl: "F3DED5", accl: "DE9C8C", slate: "4A5874", taupe: "A79E8C", sand: "CFC7B6" };

// ---------- data helpers ----------
const rows = (t) => T[t].rows;
const row = (t, p) => rows(t).find((r) => r.model.startsWith(p));
const adar = (t) => row(t, "ADAR (ours");
const nSeeds = (t) => +(/mean of (\d+) seeds/.exec(adar(t).model) || [0, 1])[1];
const EXP = ["TextCNN", "Bi-RNN", "Bi-LSTM", "Bi-GRU"];
const bestExpert = (t) => rows(t).filter((r) => EXP.includes(r.model)).sort((a, b) => b.accuracy - a.accuracy)[0];
const baselines = (t) => rows(t).filter((r) => !r.model.startsWith("ADAR"));
const router = (t) => T[t].router.chosen;
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const esc = (v) => (v > 0 && v < 0.01) || (v > 0.99 && v < 1) ? (v * 100).toFixed(1) + "%" : Math.round(v * 100) + "%";
const times = (a, b) => { const r = a / b; return (r < 10 ? r.toFixed(1) : r.toFixed(0)) + "×"; };
const msf = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0)) + " ms";
const gate = (t, needle) => { const q = T[t].qc.find((x) => x.includes(needle)); if (!q) return null;
  const m = /delta=([+-][\d.]+)pp,? (?:95% )?CI \[([+-][\d.]+), ([+-][\d.]+)\]/.exec(q);
  return m ? { ok: q.startsWith("PASS"), d: +m[1], lo: +m[2], hi: +m[3] } : null; };
const SHORT = { "Logistic Regression (TF-IDF)": "Log. regression · TF-IDF", "Linear SVM (TF-IDF)": "Linear SVM · TF-IDF",
  "Naive Bayes (TF-IDF)": "Naive Bayes · TF-IDF", "Random Forest (TF-IDF)": "Random forest · TF-IDF", "sklearn MLPClassifier (TF-IDF)": "MLP · TF-IDF",
  "Average of syllabus experts": "Average of 4 experts", "Laya zero-shot (typed choice)": "Laya zero-shot",
  "Laya encoder + Logistic Regression": "Laya + linear probe", "Stacking LogReg (Laya emb + experts)": "Stacking (Laya + experts)" };
const short = (n) => n.startsWith("ADAR (ours") ? "ADAR (ours)" : n.startsWith("ADAR cascade") ? "ADAR cascade" : SHORT[n] || n;
const colourOf = (n) => n.startsWith("ADAR cascade") ? K.accl : n.startsWith("ADAR") ? K.acc : n.startsWith("Laya") ? K.slate
  : EXP.includes(n) || n.startsWith("Average") ? K.taupe : K.sand;

// notes + takeaways from the HTML deck
const htmlOf = (id) => { const f = path.join(slidesDir, id + ".html"); return fs.existsSync(f) ? fs.readFileSync(f, "utf8") : ""; };
const unhtml = (s) => s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&nbsp;/g, " ").trim();
const notesOf = (id) => { const m = /<aside>([\s\S]*?)<\/aside>/.exec(htmlOf(id)); return m ? unhtml(m[1]) : ""; };
const takeOf = (id) => { const m = /<b style="color:#1F2233">([^<]+):<\/b>\s*([^<]+)<\/p>/.exec(htmlOf(id)); return m ? [unhtml(m[1]), unhtml(m[2])] : null; };
const demo = (() => { if (!logPath || !fs.existsSync(logPath)) return [];
  const t = fs.readFileSync(logPath, "utf8").replace(/\\"/g, '"');
  return [...t.matchAll(/'label': '([^']+)', 'probabilities': \{([^}]*)\}, 'p_head_correct': ([\d.]+), 'escalate': (True|False)/g)].slice(0, 2)
    .map((m) => { const pr = Object.fromEntries([...m[2].matchAll(/'([^']+)': ([\d.]+)/g)].map((x) => [x[1], +x[2]]));
      return { label: m[1], p: pr[m[1]], act: +m[3], esc: m[4] === "True" }; }); })();

// ---------- deck ----------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "ADAR: Adaptive Deep Architecture Routing";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, MX = 0.6, CW = W - 2 * MX;

pres.defineSlideMaster({ title: "ADAR_CONTENT", background: { color: K.bg }, margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.35, w: CW, h: 0.35, fontSize: 12, bold: true, color: K.acct, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.68, w: CW, h: 0.75, fontSize: 34, bold: true, color: K.ink, fontFace: "Cambria", margin: 0, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: W - 1.0, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: K.muted, align: "right" } });
pres.defineSlideMaster({ title: "ADAR_DARK", background: { color: K.ink },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: MX, y: 2.2, w: 8.5, h: 1.6, fontSize: 72, bold: true, color: K.card, fontFace: "Cambria", margin: 0 }, text: "" } }] });

let lastSec = null;
const sec = (t) => { if (t !== lastSec) { pres.addSection({ title: t }); lastSec = t; } return t; };
const T_ = { isTextBox: true, margin: 0 };
const txt = (s, text, o) => s.addText(text, Object.assign({}, T_, { fontFace: "Calibri", color: K.body, fontSize: 14, valign: "top" }, o));
const rect = (s, o) => s.addShape(pres.ShapeType.roundRect, Object.assign({ rectRadius: 0.08, fill: { color: K.card }, line: { color: K.line, width: 0.75 } }, o));
const label = (s, text, o) => txt(s, text.toUpperCase(), Object.assign({ fontSize: 11, bold: true, color: K.muted, charSpacing: 1.5 }, o));
const takeaway = (s, id, fallback) => { const t = takeOf(id) || ["What this shows", fallback || ""];
  s.addShape(pres.ShapeType.line, { x: MX, y: 6.55, w: CW, h: 0, line: { color: K.line, width: 0.75 } });
  txt(s, [{ text: t[0] + ": ", options: { bold: true, color: K.ink } }, { text: t[1] }], { x: MX, y: 6.65, w: CW - 0.6, h: 0.5, fontSize: 13 }); };
const content = (id, section, kicker, title) => { const s = pres.addSlide({ masterName: "ADAR_CONTENT", sectionTitle: sec(section) });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" }); s.addText(title, { placeholder: "title" });
  const n = notesOf(id); if (n) s.addNotes(n); return s; };
const card = (s, x, y, w, h, head, body, o = {}) => { rect(s, { x, y, w, h, line: { color: o.border || K.line, width: o.border ? 1.25 : 0.75 } });
  txt(s, head, { x: x + 0.2, y: y + 0.18, w: w - 0.4, h: 0.4, fontSize: o.headSize || 16, bold: true, color: K.ink });
  txt(s, body, { x: x + 0.2, y: y + 0.62, w: w - 0.4, h: h - 0.75, fontSize: o.bodySize || 13 }); };
const table = (s, header, body, o) => {
  const hdr = header.map((h, i) => ({ text: h, options: { bold: true, color: K.muted, fill: { color: K.card }, align: o.right && i >= (o.rightFrom || 1) ? "right" : "left" } }));
  const bd = body.map((r) => r.cells.map((c, i) => ({ text: String(c), options: { bold: !!r.bold, fill: { color: r.hl ? K.hl : K.card }, align: o.right && i >= (o.rightFrom || 1) ? "right" : "left" } })));
  s.addTable([hdr, ...bd], Object.assign({ fontFace: "Calibri", fontSize: 13, color: K.ink, border: { type: "solid", pt: 0.5, color: K.line }, margin: 0.06, valign: "middle" }, o)); };

// 1 cover
{ const s = pres.addSlide({ masterName: "ADAR_DARK", sectionTitle: sec("The problem") });
  s.addText("ADAR", { placeholder: "title" });
  txt(s, "DEEP LEARNING PROJECT · SYSTEM-ONE MODELS", { x: MX, y: 1.7, w: 8, h: 0.35, fontSize: 13, bold: true, color: K.accl, charSpacing: 2 });
  txt(s, "Adaptive Deep Architecture Routing", { x: MX, y: 3.8, w: 8, h: 0.6, fontSize: 26, bold: true, color: K.card, fontFace: "Cambria" });
  txt(s, "A small trained head that decides which deep-learning model to trust for each input, and when the big model is worth calling.", { x: MX, y: 4.5, w: 7.6, h: 0.9, fontSize: 16, color: "CFC9BC" });
  txt(s, "[Team names] · [Course code] · [Date]", { x: MX, y: 5.8, w: 7, h: 0.35, fontSize: 13, color: "A7A196" });
  const node = (x, y, t, sub, fill, line, tc) => { s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.4, h: 1.05, rectRadius: 0.1, fill: { color: fill }, line: { color: line, width: 1.25 } });
    txt(s, [{ text: t, options: { bold: true, fontSize: 18, color: tc, breakLine: true } }, { text: sub, options: { fontSize: 11, color: tc } }], { x, y: y + 0.2, w: 2.4, h: 0.8, align: "center" }); };
  node(9.3, 1.6, "Experts", "TextCNN · RNN · LSTM · GRU", "2A2E42", K.taupe, K.card);
  node(9.3, 4.4, "Laya", "System-One model", "2A2E42", "8C9AB8", K.card);
  node(10.3, 3.0, "ADAR", "decides per input", K.hl, K.acc, K.ink);
  s.addShape(pres.ShapeType.line, { x: 10.5, y: 2.65, w: 0.5, h: 0.35, line: { color: "8A8F9C", width: 1.25, endArrowType: "triangle" } });
  s.addShape(pres.ShapeType.line, { x: 10.5, y: 4.4, w: 0.5, h: 0.35, flipV: true, line: { color: "8A8F9C", width: 1.25, endArrowType: "triangle" } });
  s.addNotes(notesOf("cover")); }

// 2 key terms
{ const s = content("keyterms", "The problem", "Before we start", "Key Terms");
  const lo = Math.min(...EXP.map((k) => row("ag_news", k).cpu_ms)), hi = Math.max(...EXP.map((k) => row("ag_news", k).cpu_ms));
  const terms = [["System-One model", "Answers typed questions (choice, score, yes/no) in one forward pass, with a probability for every option. No text generation."],
    ["Expert", `A small syllabus network trained for the task: TextCNN, RNN, LSTM or GRU. ${msf(lo)}–${msf(hi)} per input on a CPU.`],
    ["Out-of-fold", "We only use an expert’s opinion on data that expert never trained on, so its confidence is not inflated."],
    ["Calibration (ECE)", "Does “80% sure” mean right 80% of the time? ECE is the average gap. Lower is better; 0 is perfect."],
    ["Log-loss (NLL)", "Punishes confident mistakes hardest. Lower means more honest probabilities, not just more right answers."],
    ["Cascade", "Cheap experts answer first; only inputs they are unsure about escalate to the expensive model."]];
  terms.forEach(([h, b], i) => card(s, MX + (i % 3) * 4.1, 1.75 + Math.floor(i / 3) * 2.3, 3.9, 2.1, h, b, { border: h === "Cascade" ? K.acc : null }));
  takeaway(s, "keyterms"); }

// 3 problem
{ const s = content("problem", "The problem", "Why this project", "The Problem");
  const lzE = row("emotion", "Laya zero-shot"), beE = bestExpert("emotion"), lzA = row("ag_news", "Laya zero-shot"), cnn = row("ag_news", "TextCNN");
  const td = S.typed_decisions["convaiinnovations/laya-typed-decisions"].shipped;
  const stat = (x, lab, big, col, body) => { rect(s, { x, y: 1.75, w: 3.9, h: 4.4 }); label(s, lab, { x: x + 0.25, y: 1.95, w: 3.4, h: 0.5 });
    txt(s, big, { x: x + 0.25, y: 2.5, w: 3.4, h: 1.1, fontSize: 54, bold: true, color: col, fontFace: "Cambria" });
    txt(s, body, { x: x + 0.25, y: 3.75, w: 3.4, h: 2.2, fontSize: 15 }); };
  stat(MX, "The big model is not always right", pct(lzE.accuracy), K.slate, `Laya (${lzA.params_M.toFixed(0)}M-parameter encoder) on Emotion tweets, zero-shot. A ${beE.params_M.toFixed(1)}M-parameter ${beE.model} gets ${pct(beE.accuracy)}.`);
  stat(MX + 4.1, "Big models are slow", times(lzA.cpu_ms, cnn.cpu_ms), K.ink, `slower on a CPU: Laya takes ${msf(lzA.cpu_ms)} per news article, a TextCNN ${msf(cnn.cpu_ms)}.`);
  stat(MX + 8.2, "Confidence is not honest", td.ece.toFixed(3), K.acct, `calibration error (ECE) of Laya’s fine-tuned checkpoint on the typed-decisions benchmark: off by about ${Math.round(td.ece * 100)} points.`);
  takeaway(s, "problem"); }

// 4 system-one
{ const s = content("systemone", "How ADAR works", "Background", "System-One Models");
  const boxes = [["State", "“We were billed twice. Refund it or we cancel.”"], ["Typed questions", "department? choice\nurgency? score\nwill they cancel? noul"],
    ["Laya", "one forward pass, no text generated"], ["Typed answers", "a probability for every option, a level, or P(yes)"]];
  boxes.forEach(([h, b], i) => { const x = MX + i * 3.1; card(s, x, 1.8, 2.6, 1.9, h, b, { border: h === "Laya" ? K.slate : K.ink, headSize: 14, bodySize: 14 });
    if (i < 3) s.addShape(pres.ShapeType.rightArrow, { x: x + 2.67, y: 2.6, w: 0.36, h: 0.3, fill: { color: K.muted }, line: { color: K.muted } }); });
  const lp = row("ag_news", "Laya zero-shot").params_M;
  [[`${lp.toFixed(0)}M`, "parameters in Laya’s ModernBERT-large encoder (measured in our run), plus a small decision head"],
   ["Open", "Apache-2.0, the open alternative to TypeSafe’s Jev"], ["3 types", "choice · score · noul (a yes/no probability)"]].forEach(([n, t], i) => {
    const x = MX + i * 4.1; s.addShape(pres.ShapeType.line, { x, y: 4.2, w: 3.8, h: 0, line: { color: K.ink, width: 1 } });
    txt(s, n, { x, y: 4.35, w: 3.8, h: 0.6, fontSize: 28, bold: true, color: K.ink, fontFace: "Cambria" }); txt(s, t, { x, y: 5.0, w: 3.8, h: 1.0, fontSize: 14 }); });
  takeaway(s, "systemone"); }

// 5 architecture
{ const s = content("architecture", "How ADAR works", "How ADAR works", "System Architecture");
  const e = router("ag_news"), c0 = T.ag_news.router.cost0_ms, lp = row("ag_news", "Laya zero-shot").params_M, hp = adar("ag_news").params_M;
  const node = (x, y, w, h, t, sub, border, fill = K.card) => { s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: border, width: 1.5 } });
    txt(s, [{ text: t, options: { bold: true, fontSize: 15, color: K.ink, breakLine: !!sub } }].concat(sub ? [{ text: sub, options: { fontSize: 11, color: K.muted } }] : []), { x, y, w, h, align: "center", valign: "middle" }); };
  const arrow = (x, y, w, h, o = {}) => s.addShape(pres.ShapeType.line, { x, y, w, h, line: Object.assign({ color: K.ink, width: 1.25, endArrowType: "triangle" }, o) });
  node(MX, 2.0, 1.7, 0.9, "Input text", "", K.ink);
  arrow(MX + 1.7, 2.45, 0.5, 0);
  node(2.8, 1.85, 2.9, 1.2, "Syllabus experts", `TextCNN · RNN · LSTM · GRU\n${msf(c0)} on a CPU, all four`, K.taupe);
  arrow(5.7, 2.45, 0.5, 0);
  s.addShape(pres.ShapeType.diamond, { x: 6.2, y: 1.85, w: 1.8, h: 1.2, fill: { color: K.card }, line: { color: K.ink, width: 1.25 } });
  txt(s, "Sure?", { x: 6.2, y: 1.85, w: 1.8, h: 1.2, align: "center", valign: "middle", bold: true, fontSize: 16, color: K.ink });
  arrow(8.0, 2.45, 1.3, 0); txt(s, `yes · ${esc(1 - e.escalation_rate)}`, { x: 8.1, y: 2.05, w: 1.2, h: 0.3, fontSize: 12, bold: true, color: K.muted });
  node(9.3, 2.0, 3.4, 0.9, "Answer from the experts", "fast path", K.taupe);
  arrow(7.1, 3.05, 0, 1.25); txt(s, `no · ${esc(e.escalation_rate)}`, { x: 7.25, y: 3.45, w: 1.2, h: 0.3, fontSize: 12, bold: true, color: K.muted });
  node(6.0, 4.3, 2.2, 1.1, "Laya encoder", `frozen · ${lp.toFixed(0)}M`, K.slate);
  arrow(8.2, 4.85, 1.1, 0);
  node(9.3, 4.15, 3.4, 1.4, "ADAR head", `answer · P(answer right)\nP(each expert right) · ${hp.toFixed(1)}M params`, K.acc, K.hl);
  s.addShape(pres.ShapeType.line, { x: 4.25, y: 3.05, w: 0, h: 2.95, line: { color: K.taupe, width: 1.25, dashType: "dash" } });
  s.addShape(pres.ShapeType.line, { x: 4.25, y: 6.0, w: 6.75, h: 0, line: { color: K.taupe, width: 1.25, dashType: "dash" } });
  arrow(11.0, 6.0, 0, -0.45, { color: K.taupe, dashType: "dash" });
  txt(s, "expert votes + confidence", { x: 4.4, y: 6.05, w: 3, h: 0.3, fontSize: 12, bold: true, color: K.muted });
  takeaway(s, "architecture"); }

// 6 head
{ const s = content("head", "How ADAR works", "How ADAR works", "Inside the ADAR Head");
  rect(s, { x: MX, y: 1.6, w: CW, h: 4.75, fill: { color: "FAEEE8" }, line: { color: K.acc, width: 1.25 } });
  const toks = [["STATE", K.slate, false], ["world", K.ink, true], ["sports", K.ink, true], ["business", K.ink, true], ["sci/tech", K.ink, true], ["CNN", K.taupe, false], ["RNN", K.taupe, false], ["LSTM", K.taupe, false], ["GRU", K.taupe, false]];
  label(s, "input", { x: 0.85, y: 1.75, w: 1, h: 0.3 }); label(s, "answer options", { x: 2.15, y: 1.75, w: 3, h: 0.3 }); label(s, "experts", { x: 7.15, y: 1.75, w: 3, h: 0.3 });
  toks.forEach(([t, col, dash], i) => { const x = 0.85 + i * 1.29 + (i >= 1 ? 0.0 : 0);
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.1, w: 1.15, h: 0.5, rectRadius: 0.06, fill: { color: K.card }, line: { color: col, width: 1.25, dashType: dash ? "dash" : "solid" } });
    txt(s, t, { x, y: 2.1, w: 1.15, h: 0.5, align: "center", valign: "middle", fontFace: "Courier New", fontSize: 12, bold: true, color: K.ink }); });
  s.addShape(pres.ShapeType.roundRect, { x: 3.4, y: 2.95, w: 6.5, h: 0.6, rectRadius: 0.08, fill: { color: K.ink }, line: { color: K.ink } });
  txt(s, "2-layer transformer: the tokens attend to each other", { x: 3.4, y: 2.95, w: 6.5, h: 0.6, align: "center", valign: "middle", bold: true, fontSize: 15, color: K.card });
  [["choice", "a probability for every option: the answer", K.acc], ["noul × 4", "P(this expert is right here): its routing weight α", K.taupe],
   ["act", "P(my own answer is right): abstain or escalate", K.slate]].forEach(([h, b, col], i) => { const x = 1.0 + i * 3.85;
    rect(s, { x, y: 3.9, w: 3.55, h: 2.1, line: { color: col, width: 1.5 } });
    txt(s, h, { x: x + 0.2, y: 4.05, w: 3.1, h: 0.45, fontFace: "Courier New", fontSize: 18, bold: true, color: K.ink }); txt(s, b, { x: x + 0.2, y: 4.6, w: 3.1, h: 1.2, fontSize: 14 }); });
  takeaway(s, "head"); }

// 7 training
{ const s = content("training", "How ADAR works", "How ADAR works", "How the Head Learns");
  const cols = [["1 · Honest expert opinions", null, "Each expert trains 3 times and only predicts the third of the data it never saw. The head learns from those honest opinions."],
    ["2 · A loss that rewards honesty", "L = −log p_y + Σ_k (p_k − y_k)²", "Log score + Brier score. Both are lowest only when the probabilities are honest: proper scoring rules, the same idea Laya is trained with."],
    ["3 · Calibrate", "p = softmax(z / T)", "One temperature T is fitted on validation data, so “80% sure” means right about 80% of the time."]];
  cols.forEach(([h, f, b], i) => { const x = MX + i * 4.1; rect(s, { x, y: 1.7, w: 3.9, h: 4.6 }); label(s, h, { x: x + 0.25, y: 1.9, w: 3.4, h: 0.3, color: K.acct });
    if (f) { s.addShape(pres.ShapeType.roundRect, { x: x + 0.25, y: 2.4, w: 3.4, h: 0.8, rectRadius: 0.06, fill: { color: K.alt }, line: { color: K.alt } });
      txt(s, f, { x: x + 0.35, y: 2.4, w: 3.2, h: 0.8, valign: "middle", fontFace: "Courier New", fontSize: 13, bold: true, color: K.ink }); }
    else { for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) { const pr = r === c;
      s.addShape(pres.ShapeType.roundRect, { x: x + 0.25 + c * 1.15, y: 2.4 + r * 0.38, w: 1.05, h: 0.3, rectRadius: 0.04, fill: { color: pr ? K.card : K.alt }, line: { color: pr ? K.ink : K.alt, width: 1, dashType: pr ? "dash" : "solid" } });
      txt(s, pr ? "predict" : "train", { x: x + 0.25 + c * 1.15, y: 2.4 + r * 0.38, w: 1.05, h: 0.3, align: "center", valign: "middle", fontSize: 11, color: K.ink }); } }
    txt(s, b, { x: x + 0.25, y: 3.65, w: 3.4, h: 2.4, fontSize: 15, color: K.ink }); });
  takeaway(s, "training"); }

// 8 router
{ const s = content("router", "How ADAR works", "How ADAR works", "The Cascade Router");
  const a = router("ag_news"), em = router("emotion");
  const rf = T.ag_news.router.reinforce.filter((p) => p.escalation_rate > 0 && p.accuracy >= a.accuracy - 0.001).sort((x, y) => x.escalation_rate - y.escalation_rate)[0];
  card(s, MX, 1.7, 5.9, 1.4, "Threshold router", "Escalate when P(experts right) < τ; τ is picked on validation data.");
  card(s, MX, 3.25, 5.9, 1.6, "REINFORCE router · Unit IV", "Reward r = 1[correct] − λ·cost. A small policy learns the same decision by policy gradient."
    + (rf ? ` At λ = ${rf.lambda} it calls Laya on ${esc(rf.escalation_rate)} of articles for ${pct(rf.accuracy, 2)}.` : ""));
  const tb = [["", "AG News", "Emotion"], ["Laya called on", esc(a.escalation_rate), esc(em.escalation_rate)],
    ["Cascade accuracy", pct(a.accuracy, 2), pct(em.accuracy, 2)], ["Always Laya + ADAR", pct(a.always_tier1_accuracy, 2), pct(em.always_tier1_accuracy, 2)],
    ["CPU ms per input", `${msf(a.mean_cost_ms)} vs ${msf(a.always_tier1_cost_ms)}`, `${msf(em.mean_cost_ms)} vs ${msf(em.always_tier1_cost_ms)}`],
    ["Cheaper by", times(a.always_tier1_cost_ms, a.mean_cost_ms), times(em.always_tier1_cost_ms, em.mean_cost_ms)]];
  table(s, tb[0], tb.slice(1).map((r) => ({ cells: r })), { x: 6.8, y: 1.7, w: 5.9, colW: [2.3, 1.8, 1.8], right: true });
  txt(s, `Headroom: a perfect router would reach ${pct(a.oracle_accuracy, 2)} on AG News, calling Laya on only ${esc(a.oracle_escalation_rate)} of inputs.`, { x: 6.8, y: 4.6, w: 5.9, h: 0.6, fontSize: 13, color: K.muted });
  takeaway(s, "router"); }

// 9 setup
{ const s = content("setup", "Results", "Results", "Experiment Setup");
  const sp = S.splits, nb = baselines("ag_news").length, sk = baselines("ag_news").filter((r) => r.family === "scikit-learn").length;
  table(s, ["Dataset", "Text", "Train", "Val", "Test", "Classes"], [
    { cells: ["AG News", "news headline + first line", sp.ag_news.train.toLocaleString("en-US"), sp.ag_news.val.toLocaleString("en-US"), sp.ag_news.test.toLocaleString("en-US"), sp.ag_news.classes] },
    { cells: ["Emotion", "short English tweets", sp.emotion.train.toLocaleString("en-US"), sp.emotion.val.toLocaleString("en-US"), sp.emotion.test.toLocaleString("en-US"), sp.emotion.classes] }],
    { x: MX, y: 1.75, w: CW, colW: [2.0, 4.13, 1.6, 1.4, 1.4, 1.6], right: true, rightFrom: 2, fontSize: 15 });
  [[`${nb} baselines`, `${sk} scikit-learn models, 4 syllabus networks and their average, Laya zero-shot, Laya probe, stacking`],
   ["Fair tests", `same test rows, every model calibrated on validation data, ADAR over ${nSeeds("ag_news")} seeds, 95% bootstrap confidence intervals`],
   [`${Math.round(S.total_minutes)} minutes`, "whole pipeline on Kaggle’s free GPU T4 ×2, resumable after every stage"]].forEach(([n, t], i) => {
    const x = MX + i * 4.1; s.addShape(pres.ShapeType.line, { x, y: 3.7, w: 3.8, h: 0, line: { color: K.ink, width: 1 } });
    txt(s, n, { x, y: 3.85, w: 3.8, h: 0.6, fontSize: 26, bold: true, color: K.ink, fontFace: "Cambria" }); txt(s, t, { x, y: 4.5, w: 3.8, h: 1.4, fontSize: 14 }); });
  takeaway(s, "setup"); }

// 10/12 results charts
const resultsSlide = (task, id, title, nTest) => {
  const s = content(id, "Results", "Results · test accuracy", title);
  const items = rows(task).filter((r) => !r.model.startsWith("ADAR ablation")).sort((a, b) => a.accuracy - b.accuracy);
  const lo = Math.max(0, Math.floor((Math.min(...items.map((r) => r.accuracy)) * 100 - 2) / 5) * 5);
  s.addChart(pres.ChartType.bar, [{ name: "Test accuracy", labels: items.map((r) => short(r.model)), values: items.map((r) => +(r.accuracy * 100).toFixed(2)) }], {
    x: MX, y: 1.55, w: 7.9, h: 4.4, barDir: "bar", chartColors: items.map((r) => colourOf(r.model)), barGapWidthPct: 35,
    valAxisMinVal: lo, valAxisMaxVal: 100, valAxisMajorUnit: lo >= 80 ? 5 : 10, valAxisLabelFormatCode: "0\"%\"",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", dataLabelFontSize: 10, dataLabelColor: K.ink, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: K.body, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt", valAxisLabelColor: K.muted, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E6DECF", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, showTitle: false });
  txt(s, `How to read: test accuracy on ${nTest}, axis from ${lo}%. ADAR = mean of ${nSeeds(task)} seeds (± ${(adar(task).accuracy_std * 100).toFixed(2)} pts).\nColours: ADAR (red) · Laya (slate) · syllabus networks (taupe) · scikit-learn and stacking (sand).`,
    { x: MX, y: 5.98, w: 7.9, h: 0.5, fontSize: 10, color: K.muted });
  rect(s, { x: 8.8, y: 1.6, w: 3.93, h: 4.75 });
  label(s, "ADAR (seed 0) minus …", { x: 9.0, y: 1.75, w: 3.5, h: 0.3 });
  const refs = [["Laya zero-shot", "best Laya zero-shot"], ["best syllabus network", "best syllabus deep model"], ["best scikit-learn", "best scikit-learn model"], ["stacking", "best stacking baseline"]];
  let y = 2.15;
  refs.forEach(([n, k]) => { const g = gate(task, k); if (!g) return;
    const word = g.ok ? "significant" : g.hi < 0 ? "significantly behind" : "not significant";
    txt(s, [{ text: "vs " + n, options: { bold: true, color: K.ink, breakLine: true } },
      { text: `${g.d >= 0 ? "+" : ""}${g.d.toFixed(2)} pts · CI [${g.lo.toFixed(2)}, ${g.hi.toFixed(2)}] · ` }, { text: word, options: { color: g.ok ? K.acct : K.muted, bold: g.ok } }],
      { x: 9.0, y, w: 3.55, h: 0.8, fontSize: 13 }); y += 0.9; });
  txt(s, `${T[task].qc_pass} of ${T[task].qc_total} quality checks passed`, { x: 9.0, y: 5.85, w: 3.5, h: 0.35, fontSize: 12, color: K.muted });
  takeaway(s, id);
};
resultsSlide("ag_news", "results-ag", "AG News Results", `${S.splits.ag_news.test.toLocaleString("en-US")} news articles`);

// 11 honesty & cost
{ const s = content("honesty-cost", "Results", "Results · AG News", "Honesty and Cost");
  const a = adar("ag_news"), cas = row("ag_news", "ADAR cascade"), lz = row("ag_news", "Laya zero-shot");
  const best = baselines("ag_news").slice().sort((x, y) => x.nll - y.nll)[0], red = 1 - a.nll / best.nll;
  const panel = (x, lab, big, bigCol, sub, data, fmt) => { rect(s, { x, y: 1.6, w: 5.95, h: 4.75 }); label(s, lab, { x: x + 0.25, y: 1.75, w: 5.4, h: 0.3 });
    txt(s, big, { x: x + 0.25, y: 2.1, w: 5.4, h: 0.9, fontSize: 48, bold: true, color: bigCol, fontFace: "Cambria" });
    txt(s, sub, { x: x + 0.25, y: 3.0, w: 5.4, h: 0.6, fontSize: 14 });
    s.addChart(pres.ChartType.bar, [{ name: lab, labels: data.map((d) => d[0]), values: data.map((d) => d[1]) }], { x: x + 0.1, y: 3.65, w: 5.7, h: 2.55, barDir: "bar",
      chartColors: data.map((d) => d[2]), barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: fmt, dataLabelFontSize: 11, dataLabelColor: K.ink, dataLabelFontFace: "+mn-lt",
      catAxisLabelColor: K.body, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin" }); };
  const nllData = [["ADAR", +a.nll.toFixed(3), K.acc], [short(best.model), +best.nll.toFixed(3), colourOf(best.model)]];
  if (best !== lz) nllData.push(["Laya zero-shot", +lz.nll.toFixed(3), K.slate]);
  panel(MX, "Honest probabilities · log-loss (lower is better)", `${Math.round(Math.abs(red) * 100)}% ${red > 0 ? "lower" : "higher"}`, K.acct, `log-loss than the best baseline (${short(best.model)})`, nllData, "0.000");
  panel(6.78, "Cost · CPU ms per input", `${times(a.cpu_ms, cas.cpu_ms)} cheaper`, K.slate, `than always running Laya + ADAR, at ${pct(cas.accuracy, 2)} vs ${pct(a.accuracy, 2)} accuracy`,
    [["Laya zero-shot", Math.round(lz.cpu_ms), K.slate], ["Always Laya + ADAR", Math.round(a.cpu_ms), K.acc], ["ADAR cascade", Math.round(cas.cpu_ms), K.accl]], "0\" ms\"");
  takeaway(s, "honesty-cost"); }

resultsSlide("emotion", "results-emotion", "Emotion Results", `${S.splits.emotion.test.toLocaleString("en-US")} tweets`);

// 13 ablation
{ const s = content("ablation", "Results", "Results · ablations", "What Each Part Adds");
  const cell = (t, k) => { const g = gate(t, k), r = row(t, "ADAR " + k); return g && r ? [`${(-g.d).toFixed(2)} pts (${g.lo > 0 ? "yes" : "no"})`, r.nll.toFixed(3)] : ["–", "–"]; };
  const v = [["Without the experts (Laya only)", "ablation: laya only"], ["Without Laya (experts only)", "ablation: experts only"], ["Without attention", "ablation: no attention"]];
  table(s, ["Variant", "AG News Δ acc", "NLL", "Emotion Δ acc", "NLL"], [
    { hl: true, bold: true, cells: ["Full ADAR", pct(adar("ag_news").accuracy), adar("ag_news").nll.toFixed(3), pct(adar("emotion").accuracy), adar("emotion").nll.toFixed(3)] },
    ...v.map(([n, k]) => ({ cells: [n, ...cell("ag_news", k), ...cell("emotion", k)] }))],
    { x: MX, y: 1.75, w: CW, colW: [4.33, 2.4, 1.5, 2.4, 1.5], right: true, fontSize: 15 });
  txt(s, "How to read: Δ = accuracy change when the part is removed and the head retrained (seed 0, paired bootstrap); “yes” = the 95% CI excludes zero. NLL = log-loss, lower is better.",
    { x: MX, y: 3.75, w: CW, h: 0.4, fontSize: 12, color: K.muted });
  [["ag_news", "ablation: laya only", "AG News without the experts"], ["ag_news", "ablation: experts only", "AG News without Laya"], ["emotion", "ablation: laya only", "Emotion without the experts"]].forEach(([t, k, lab], i) => {
    const g = gate(t, k); if (!g) return; const x = MX + i * 4.1;
    rect(s, { x, y: 4.35, w: 3.9, h: 1.95 });
    txt(s, `${(-g.d).toFixed(2)} pts`, { x: x + 0.25, y: 4.5, w: 3.4, h: 0.8, fontSize: 36, bold: true, color: g.lo > 0 ? K.acct : K.muted, fontFace: "Cambria" });
    txt(s, `${lab} (${g.lo > 0 ? "significant" : "not significant"})`, { x: x + 0.25, y: 5.35, w: 3.4, h: 0.8, fontSize: 14 }); });
  takeaway(s, "ablation"); }

// 14 typed-decisions
{ const s = content("leaderboard", "Leaderboards", "Public benchmark · System-One models", "Typed-Decisions Benchmark");
  const td = S.typed_decisions, ft = td["convaiinnovations/laya-typed-decisions"], base = td["convaiinnovations/laya"];
  const sel = (m) => m["adar_cal_" + m.adar_cal_selected];
  rect(s, { x: MX, y: 1.6, w: 5.2, h: 4.75 }); label(s, "1 · Our scorer reproduces the board", { x: MX + 0.25, y: 1.75, w: 4.7, h: 0.3 });
  txt(s, `Laya typed-decisions · ${ft.shipped.decisions.toLocaleString("en-US")} decisions`, { x: MX + 0.25, y: 2.1, w: 4.7, h: 0.3, fontSize: 12, color: K.muted });
  table(s, ["Metric", "Board", "Our run"], [{ cells: ["Accuracy", "0.766", ft.shipped.accuracy.toFixed(3)] }, { cells: ["Brier", "not listed", ft.shipped.brier_soft.toFixed(3)] },
    { cells: ["KL", "not listed", ft.shipped.kl.toFixed(3)] }, { cells: ["ECE", "not listed", ft.shipped.ece.toFixed(3)] }], { x: MX + 0.25, y: 2.55, w: 4.7, colW: [1.7, 1.5, 1.5], right: true, fontSize: 14 });
  rect(s, { x: 6.0, y: 1.6, w: 6.73, h: 4.75, line: { color: K.acc, width: 1.25 } }); label(s, "2 · Calibration study (fitted on train)", { x: 6.25, y: 1.75, w: 6, h: 0.3, color: K.acct });
  table(s, ["Model", "Acc ↑", "ECE ↓", "Brier ↓", "KL ↓"], [["Laya (base)", base.shipped], [`+ ADAR-Cal (${base.adar_cal_selected})`, sel(base)], ["Laya typed-decisions", ft.shipped], [`+ ADAR-Cal (${ft.adar_cal_selected})`, sel(ft)]]
    .map(([n, m]) => ({ cells: [n, m.accuracy.toFixed(3), m.ece.toFixed(3), m.brier_soft.toFixed(3), m.kl.toFixed(3)] })), { x: 6.25, y: 2.2, w: 6.25, colW: [2.45, 0.95, 0.95, 0.95, 0.95], right: true, fontSize: 13 });
  txt(s, `Base Laya: ECE ${base.shipped.ece.toFixed(3)} → ${sel(base).ece.toFixed(3)}. Calibration never changes the chosen answer, so accuracy stays the same.`, { x: 6.25, y: 4.75, w: 6.25, h: 0.7, fontSize: 13 });
  takeaway(s, "leaderboard"); }

// 15 decision
{ const s = content("decision", "Leaderboards", "Leaderboards", "Where We Submit, and Why");
  const base = S.typed_decisions["convaiinnovations/laya"].shipped;
  const rws = [["typed-decisions (discussion #2 + .eval_results PR)", "Independent re-run of Laya typed-decisions with the Brier, KL and ECE the maintainer asked for", "Answers an open request; puts the verified row on the Hub leaderboard", "Submit now"],
    ["Hugging Face model pages", "Both trained heads, with results in a model-index block", "Anyone can load and check the heads", "Publish now"],
    ["RouterArena (ICLR 2026)", "ADAR routing between LLMs instead of small networks", "Ranked, live leaderboard for exactly what ADAR does: the head-turner", "Next phase"],
    ["“Laya + ADAR-Cal” as a new row", `accuracy ${base.accuracy.toFixed(3)}, below the board’s prior baseline (0.470)`, "Would read as a weak model, although only calibration changed", "Do not submit"]];
  s.addTable([["Where", "What we submit", "Why it matters", "Decision"].map((h) => ({ text: h, options: { bold: true, color: K.muted, fill: { color: K.card } } })),
    ...rws.map((r) => r.map((c, i) => ({ text: c, options: { fill: { color: K.card }, bold: i === 0 || i === 3, color: i === 3 ? (c.endsWith("now") ? K.acct : K.muted) : K.ink } })))],
    { x: MX, y: 1.7, w: CW, colW: [2.6, 3.9, 4.1, 1.53], fontFace: "Calibri", fontSize: 15, border: { type: "solid", pt: 0.5, color: K.line }, margin: 0.08, valign: "middle" });
  txt(s, "The 0.470 prior is the board’s own floor (always predicting the label frequencies). Steps and links: docs/LEADERBOARD_SUBMISSION.md.", { x: MX, y: 5.9, w: CW, h: 0.4, fontSize: 12, color: K.muted });
  takeaway(s, "decision"); }

// 16 syllabus
{ const s = content("syllabus", "Course fit", "Deep Learning course", "Syllabus Coverage");
  [["Unit I", ["TextCNN", "Backpropagation", "BatchNorm", "Gradient clipping", "AdamW + cosine schedule"], K.taupe],
   ["Unit II", ["CNN classifiers", "A new head on a frozen pretrained network"], K.taupe],
   ["Unit III", ["RNN, LSTM, GRU", "Embeddings", "Attention", "Transformer head", "BERT-family encoder"], K.slate],
   ["Unit IV", ["Deep RL framework", "REINFORCE policy gradient", "Reward = correct − λ·cost"], K.acc]].forEach(([u, items, col], i) => {
    const x = MX + i * 3.08; rect(s, { x, y: 1.7, w: 2.9, h: 4.5, line: { color: col, width: 1.5 } });
    txt(s, u, { x: x + 0.2, y: 1.85, w: 2.5, h: 0.55, fontSize: 24, bold: true, color: K.ink, fontFace: "Cambria" });
    txt(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })), { x: x + 0.2, y: 2.5, w: 2.55, h: 3.5, fontSize: 15, color: K.ink, paraSpaceAfter: 6 }); });
  takeaway(s, "syllabus"); }

// 17 alignment
{ const s = content("alignment", "Course fit", "Context · this course", "Where ADAR Fits");
  table(s, ["Unit", "Class projects on this unit"], [
    { cells: ["II · CNNs in vision", "Neural Style Transfer, Fast NST, R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN"] },
    { cells: ["III · RNNs, transformers", "Image captioning, GPT-2 summariser, time-series forecasting, speech recognition, machine translation, BERT classifier, RNN vs LSTM vs GRU, echo-state networks"] },
    { cells: ["IV · GANs, deep RL", "GAN, DCGAN, Q-learning grid world, DQN, REINFORCE on CartPole"] }, { cells: ["Other", "Text-to-music generation"] },
    { hl: true, bold: true, cells: ["I + II + III + IV", "ADAR: TextCNN and BatchNorm · new head on a frozen pretrained network · RNN, LSTM, GRU, attention, BERT-family encoder · REINFORCE router"] }],
    { x: MX, y: 1.7, w: 7.9, colW: [2.0, 5.9], fontSize: 12 });
  label(s, "Closest overlaps", { x: 8.8, y: 1.75, w: 3.9, h: 0.3 });
  txt(s, [{ text: "BERT classifier: ", options: { bold: true, color: K.ink } }, { text: "ADAR uses a BERT-family encoder as one input, and reports a probe on it as a baseline.", options: { breakLine: true } },
    { text: "RNN vs LSTM vs GRU: ", options: { bold: true, color: K.ink } }, { text: "ADAR uses all three as experts and learns per input which to trust.", options: { breakLine: true } },
    { text: "REINFORCE: ", options: { bold: true, color: K.ink } }, { text: "ADAR applies it to a real decision, accuracy against compute cost." }],
    { x: 8.8, y: 2.15, w: 3.9, h: 3.8, fontSize: 14, color: K.ink, paraSpaceAfter: 10 });
  takeaway(s, "alignment"); }

// 18 limits
{ const s = content("limits", "Course fit", "Wrap-up", "Limits and Next Steps");
  const lim = ["Text classification only so far", `Laya stays frozen; only the ${adar("ag_news").params_M.toFixed(1)}M-parameter head trains`];
  const g = gate("emotion", "best syllabus deep model"); if (g && !g.ok) lim.push(`On Emotion, ADAR ties the best syllabus network (${bestExpert("emotion").model})`);
  const na = gate("ag_news", "ablation: no attention"); if (na && na.lo <= 0) lim.push("Attention’s accuracy gain is not significant");
  lim.push("ADAR-Cal fixes calibration, not accuracy", "One train/val/test protocol per dataset");
  const nxt = ["RouterArena: route between LLMs instead of small networks", "An image task (CIFAR-10) with CNN experts", "An echo-state network as a fifth expert", "A short arXiv write-up"];
  [["Limits", lim, K.line], ["Next steps", nxt, K.acc]].forEach(([h, items, col], i) => { const x = MX + i * 6.15; rect(s, { x, y: 1.7, w: 5.95, h: 4.5, line: { color: col, width: 1.25 } });
    txt(s, h, { x: x + 0.25, y: 1.85, w: 5.4, h: 0.55, fontSize: 24, bold: true, color: K.ink, fontFace: "Cambria" });
    txt(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })), { x: x + 0.25, y: 2.5, w: 5.45, h: 3.5, fontSize: 15, color: K.ink, paraSpaceAfter: 6 }); });
  takeaway(s, "limits", "the gains are real but bounded; the clearest next win is the same head routing between large language models."); }

// 19 demo
{ const s = content("demo", "Course fit", "Wrap-up", "Live Demo");
  [["1 · Run", `Kaggle notebook on GPU T4 ×2: trains, tests and writes the report in ${Math.round(S.total_minutes)} minutes`],
   ["2 · Publish", (S.hf_uploads || []).length ? "Both heads on Hugging Face: " + S.hf_uploads.join(", ") : "Both trained heads, with model cards from this run, ready for Hugging Face"],
   ["3 · Use", "Load the head and classify any sentence, with honest confidence"]].forEach(([h, b], i) => { const x = MX + i * 4.1;
    s.addShape(pres.ShapeType.line, { x, y: 1.7, w: 3.8, h: 0, line: { color: i === 2 ? K.acc : K.ink, width: 1 } });
    txt(s, h, { x, y: 1.85, w: 3.8, h: 0.55, fontSize: 22, bold: true, color: K.ink, fontFace: "Cambria" }); txt(s, b, { x, y: 2.45, w: 3.8, h: 1.0, fontSize: 14 }); });
  s.addShape(pres.ShapeType.roundRect, { x: MX, y: 3.6, w: CW, h: 2.6, rectRadius: 0.1, fill: { color: K.ink }, line: { color: K.ink } });
  const code = ["from adar.inference import ADARClassifier", 'clf = ADARClassifier.from_pretrained("adar-head-ag-news")',
    "clf.predict([\"NASA's new telescope captured images of a distant galaxy cluster.\",", '             "Real Madrid beat Barcelona 3-1 in the Champions League semi-final on Tuesday."])'];
  const outs = demo.map((d) => `→ ${d.label} ${d.p.toFixed(3)} · P(answer right) ${d.act.toFixed(3)} · escalate: ${d.esc ? "yes" : "no"}`);
  txt(s, code.map((c, j) => ({ text: c, options: { color: K.card, breakLine: true } })).concat(outs.map((o, j) => ({ text: o, options: { color: K.accl, breakLine: j < outs.length - 1 } }))),
    { x: MX + 0.3, y: 3.8, w: CW - 0.6, h: 2.25, fontFace: "Courier New", fontSize: 15, paraSpaceAfter: 4 });
  txt(s, "Live: run this in the notebook (or Python with the head downloaded); the two lines above are the real output from our Kaggle run.", { x: MX, y: 6.3, w: CW, h: 0.4, fontSize: 12, color: K.muted }); }

// 20 closing
{ const s = pres.addSlide({ masterName: "ADAR_DARK", sectionTitle: sec("Course fit") });
  s.addText("Thank You", { placeholder: "title" });
  txt(s, "Questions?", { x: MX, y: 3.9, w: 6, h: 0.6, fontSize: 28, bold: true, color: K.accl, fontFace: "Cambria" });
  txt(s, ["github.com/Harsh4-Dev/System-One-DLE"].concat((S.hf_uploads || []).map((r) => "huggingface.co/" + r)).join("\n"), { x: MX, y: 4.7, w: 8, h: 0.8, fontFace: "Courier New", fontSize: 14, color: "CFC9BC" });
  s.addNotes(notesOf("closing")); }

(async () => {
  await pres.writeFile({ fileName: outPath });
  const { applyTheme } = require(process.env.PPTX_APPLY_THEME);
  await applyTheme(outPath, THEME);
  console.log("wrote", outPath);
})();
