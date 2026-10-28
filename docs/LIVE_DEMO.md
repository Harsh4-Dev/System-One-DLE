# Live demo guide

Three ways to demo, from safest to most impressive. Prepare **A** in any case, and **B or C** the day before.

| | What | Needs | Prep | Risk |
|---|---|---|---|---|
| **A** | Show the real output already in the run | nothing (slide or Kaggle page) | 0 min | none |
| **B** | Run the trained head live on Kaggle | browser, Kaggle login, internet | 10 min | low |
| **C** | Run it on your laptop | Python 3.10+, ~1 GB free, internet once | 20 min | low once tested |

The demo script `tools/demo.py` prints, for each sentence: a probability bar for every label, ADAR's answer,
**P(answer is right)**, whether it would **escalate**, and for each of the four syllabus experts its own vote,
**P(expert right)** and the **routing weight α** the head gave it. That is the whole idea of ADAR on one screen.

---

## A. Zero-risk: show the run's own output

1. Deck slide **Live Demo** shows the real output from our Kaggle run:
   `sci_tech 0.927 · P(answer right) 0.966 · escalate: no` and `sports 0.990 · P(answer right) 0.998 · escalate: no`.
2. Or open https://www.kaggle.com/code/harshavardhan88/adar-system-one, scroll to the last cell ("demo"),
   and show the printed predictions. That page is the executed notebook: proof the numbers came from a real run.

Say: *"This is the head trained in our 63-minute Kaggle run. For a news sentence it gives the answer, a probability
for every topic, and its own probability of being right; when that is low it escalates instead of guessing."*

## B. Live on Kaggle (recommended)

Do steps 1–5 the day before, then just press **Run all** in the talk.

1. Open https://www.kaggle.com/code → **New Notebook**.
2. Right panel → **Session options**: Accelerator **GPU T4 ×2** (or None for CPU, slower), **Internet on**
   (needs a phone-verified account).
3. Right panel → **Add Input** → **Your Work** → **Notebooks** → `adar-system-one` → **Add**.
   Its output (including the trained heads) appears under `/kaggle/input/`.
4. Paste these cells:

```python
# Cell 1: code + Laya (about 1 minute)
!pip install -q "laya>=0.3.23"
!git clone -q https://github.com/Harsh4-Dev/System-One-DLE.git
%cd System-One-DLE
!git checkout -q claude/intelligent-bell-xzh81w   # skip once the branch is merged into main
```

```python
# Cell 2: find the trained heads in the attached run output
import glob, os, zipfile
heads = glob.glob("/kaggle/input/**/publish/adar-head-*", recursive=True)
if not heads:  # only the zip was attached: unzip it
    z = glob.glob("/kaggle/input/**/adar_outputs.zip", recursive=True)[0]
    zipfile.ZipFile(z).extractall("/kaggle/working/run")
    heads = glob.glob("/kaggle/working/run/**/publish/adar-head-*", recursive=True)
AG = next(h for h in heads if h.endswith("ag_news")); EMO = next(h for h in heads if h.endswith("emotion"))
print(AG, EMO, sep="\n")
```

```python
# Cell 3: the demo (built-in example sentences)
!python tools/demo.py --head {AG}
```

```python
# Cell 4: the audience's sentence (edit the text live)
!python tools/demo.py --head {AG} "Type the sentence the audience gives you here."
```

```python
# Cell 5: same idea on tweets (Laya is weak here, the experts are strong)
!python tools/demo.py --head {EMO} "i finally got the job i wanted and i can't stop smiling"
```

5. **Run all once** (first run downloads Laya, about 0.8 GB). Then **Save Version → Quick save** so the outputs stay
   visible even if the Wi-Fi fails during the talk (you can fall back to showing them, as in A).

Tip: `python tools/demo.py --head {AG} -i` asks for sentences one by one (type, Enter; empty line to stop).

## C. On your laptop

```bash
git clone https://github.com/Harsh4-Dev/System-One-DLE.git
cd System-One-DLE
git checkout claude/intelligent-bell-xzh81w          # until merged into main
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch "laya>=0.3.23" safetensors scikit-learn
```

Get the heads (about 50 MB each):

- from Kaggle: notebook page → **Output** → download `adar_outputs.zip`, unzip; the heads are in `publish/`; or
- from Hugging Face once uploaded (`docs/LEADERBOARD_SUBMISSION.md`, section 3):
  `python -c "from huggingface_hub import snapshot_download; print(snapshot_download('DEV-HV/adar-head-ag-news'))"`

Run it once **the day before** (downloads Laya and checks everything works):

```bash
python tools/demo.py --head publish/adar-head-ag_news            # example sentences
python tools/demo.py --head publish/adar-head-ag_news -i         # type sentences live
python tools/demo.py --head publish/adar-head-emotion "i feel so alone since my best friend moved away"
```

On a laptop CPU: about 2 minutes to load (the Laya encoder), then about 1 second per sentence (tested in our
environment with the real Laya model: 860 ms per sentence). Load it **before** you start talking.
After the first run it also works offline: set `HF_HUB_OFFLINE=1`.

---

## What to show, in 2 minutes

1. **An easy sentence** ("Real Madrid beat Barcelona 3-1…"): all experts agree, high P(answer right), no escalation.
   *"Easy inputs: the cheap experts already agree, the head is confident."*
2. **An ambiguous one** ("Apple unveiled a faster chip… its shares rose 3%"): tech or business?
   Point at the experts that **disagree**, the lower P(answer right) and the routing weights.
   *"Here the head sees the disagreement, weighs the experts per input, and its confidence drops."*
3. **A sentence from the audience.** Whatever it is, read out the answer and P(answer right).
4. **One tweet with the emotion head.** *"On tweets Laya alone gets 58%; the experts carry this task, and the router
   learned to call Laya for only 0.1% of tweets."*

If the demo fails: say *"Here is the same thing from our recorded run"* and switch to A. Never debug live.

## Likely questions (with answers from our run)

| Question | Answer |
|---|---|
| Is ADAR better than just using Laya? | On AG News yes: 93.28% vs 92.53%, +0.93 points, 95% CI +0.42 to +1.47 (significant). On tweets by +34 points. |
| Better than the best syllabus model? | On AG News yes (+1.05 points, significant). On Emotion it **ties** Bi-LSTM; we say so. |
| What does "calibrated" mean here? | When ADAR says 80%, it is right about 80% of the time. Its log-loss is 15% lower than the best baseline's. |
| Why not always run Laya? | Cost: the cascade gets 93.32% at 189 ms per input instead of 492 ms (2.6× cheaper). |
| Where is deep learning in this? | All four units: TextCNN/BatchNorm (I), new head on a frozen pretrained network (II), RNN/LSTM/GRU, attention, transformer, BERT-family encoder (III), REINFORCE router (IV). |
| Did you train Laya? | No, Laya is frozen. We trained the 2.3M-parameter head and the four experts. |
| How do you know the numbers are real? | One Kaggle run, executed notebook public, proof bundle and log in the repo; every slide number is generated from `results/kaggle_run/summary.json`. Our scorer reproduces Laya's published 0.766 on typed-decisions. |
| What's the leaderboard result? | We independently re-ran Laya on typed-decisions (0.7665 vs the self-reported 0.766) and added the KL, Brier and ECE the maintainers asked for. We do not claim a rank. |
| What would you do next? | RouterArena: the same head routing between large language models. |
| Limitations? | Text only; Laya frozen; ties on Emotion; attention's accuracy gain not significant; calibration fixes honesty, not accuracy. |
