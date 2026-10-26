# Fresh run with the Kaggle CLI (proof run)

One command pushes the notebook to Kaggle's free **GPU T4 ×2**, waits for it, downloads the proof
bundle and regenerates **every number** the project shows (deck, Word report, model cards,
leaderboard post) from that run.

```
python tools/kaggle_run.py --user <kaggle-username>
```

```
notebooks/adar_kaggle.ipynb ──push──► Kaggle (GPU T4 ×2, ~1–1.5 h) ──► adar_results_small.zip
                                                                        │
       tools/collect_results.py ◄──────────────────────────────────────┘
                 │
                 ▼
     results/kaggle_run/summary.json   ← the single source of truth
                 │
   ┌─────────────┼───────────────────────────────┐
   ▼             ▼                               ▼
 make_submissions.py      tools/report/*        the deck (numbers copied from summary.json)
 (leaderboard post,       (figures + Word
  model cards)             report)
```

## 1. One-time setup (5 minutes)

### 1.1 Kaggle account

1. Sign in at kaggle.com and **verify your phone number** (Settings → Phone verification).
   Without it Kaggle refuses GPUs and internet for notebooks.
2. **Settings → API → Generate New Token.** Copy the token (it starts with `KGAT_` on new accounts).
   Keep it secret; never paste it into a chat or commit it.

### 1.2 Running from a Claude Code cloud session

Open the environment settings (environment menu in the session title bar → **Edit**):

| Setting | Value |
|---|---|
| Network access → Allowed domains | `kaggle.com`, `www.kaggle.com`, `storage.googleapis.com` (output downloads), `huggingface.co` |
| Environment variable / secret `KAGGLE_API_TOKEN` | the token from 1.1 |
| Environment variable `KAGGLE_USERNAME` | your Kaggle username |
| Secret `HF_TOKEN` (optional) | a Hugging Face **write** token, to update the model cards |

A **new session** picks the secrets up. Start it on branch `claude/intelligent-bell-xzh81w` and say:

> Do the fresh Kaggle run in docs/KAGGLE_CLI_SETUP.md, then follow docs/NEXT_STEPS.md.

### 1.3 Running from your own computer instead

```bash
git clone https://github.com/Harsh4-Dev/System-One-DLE && cd System-One-DLE
git checkout claude/intelligent-bell-xzh81w
pip install kaggle torch scikit-learn scipy matplotlib
export KAGGLE_API_TOKEN=...        # or: kaggle auth login   (browser sign-in)
export KAGGLE_USERNAME=...
```

## 2. Run

```bash
python tools/kaggle_run.py --user $KAGGLE_USERNAME --quick   # optional 30-45 min check on subsets
python tools/kaggle_run.py --user $KAGGLE_USERNAME           # the full proof run
```

What it does:

1. Builds `build/kaggle_kernel/` (the notebook + `kernel-metadata.json`: private, GPU, internet on).
2. `kaggle kernels push -p build/kaggle_kernel --accelerator NvidiaTeslaT4` (GPU T4 ×2).
3. Polls `kaggle kernels status` every 2 minutes.
4. `kaggle kernels output … --file-pattern adar_results_small.zip` into `runs/<timestamp>/`.
5. `tools/collect_results.py` → `results/kaggle_run/summary.json` (the old one is archived in `results/previous_runs/`).
6. Regenerates `submissions/` and `docs/ADAR_Project_Report.docx` (the kernel log is kept as
   `results/kaggle_run/kernel.log`; the demo output on the slides and in the report comes from it).
7. With `--deck <deck>/project`, also rebuilds every slide with `tools/report/build_slides.py`.
   Run that script by hand any time: `python tools/report/build_slides.py results/kaggle_run/summary.json <deck>/project --log results/kaggle_run/kernel.log`.

The slide, report and post wording follows the quality checks: "beats" only when the 95% CI
excludes zero, "ties" when it does not, "behind" when ADAR is significantly worse.

If your terminal closes during the run, the Kaggle run continues. Fetch it later with
`python tools/kaggle_run.py --user $KAGGLE_USERNAME --no-push`.

## 3. The proof

| Proof | Where |
|---|---|
| The executed notebook, every cell's output | kaggle.com/code/<user>/adar-system-one (the version created by the push); make it public to share |
| The raw numbers | `runs/<timestamp>/adar_results_small.zip` (tables, QC checks, router curves, typed-decisions metrics, run log) |
| The numbers we quote | `results/kaggle_run/summary.json`, built only from that zip |

Rule for the deck and the report: **a number that is not in `summary.json` does not go on a slide.**

## 4. Troubleshooting

| Message | Fix |
|---|---|
| `401` / `Authentication required` | `KAGGLE_API_TOKEN` missing or wrong; generate a new token |
| `403` on push, or "GPU not available" | phone verification missing, or the weekly GPU quota (30 h) is used up |
| status `error` | open the version on kaggle.com → Logs; or download `.*\.log` with `kaggle kernels output` |
| `no adar_results_small.zip` | the run failed before its last stage; the log names the stage; re-run (finished stages are cached only inside one Kaggle session) |
| `CONNECT tunnel failed, 403` in a cloud session | the domain is not in Allowed domains (1.2) |
