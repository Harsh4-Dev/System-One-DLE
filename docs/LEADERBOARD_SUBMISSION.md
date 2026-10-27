# Leaderboard submission guide

Everything to submit is already generated from the fresh Kaggle run (`results/kaggle_run/summary.json`)
by `python tools/make_submissions.py`. This guide says **where** each piece goes, **why**, and the exact
clicks or commands. Links were checked on 2026-10-27.

| # | Target | What we submit | Value | When | Time |
|---|---|---|---|---|---|
| 0 | GitHub + Kaggle | make the repo and the notebook public | every post links to them | **first** | 10 min |
| 1 | typed-decisions discussion #2 | independent re-run of Laya typed-decisions: accuracy plus the KL / Brier / ECE the maintainer asked for | answers an open maintainer request on the System-One benchmark | **now** | 5 min |
| 2 | laya-typed-decisions model repo | community PR with `.eval_results/typed-decisions.yaml` | the row shows on the Hub leaderboard widget, linked to our run | **now** | 10 min |
| 3 | Hugging Face model pages | our two trained ADAR heads with model cards | anyone can load and check our heads | **now** | 20 min |
| 4 | RouterArena | ADAR as a router between LLMs | ranked, live leaderboard for exactly what ADAR does: the head-turner | **next phase** | days |
| ✗ | typed-decisions, new row "Laya + ADAR-Cal" | — | accuracy 0.363 is below the board's prior baseline (0.470) | **do not submit** | — |

Why not a new typed-decisions row: our calibration step improves Laya's calibration (ECE 0.173 → 0.021)
but never changes its answers, so accuracy stays 0.363. A row would rank last and read as a weak model.
The honest, useful contribution is the reproduction and the missing metrics (1 and 2).

---

## 0. Make the sources public (do this first)

Every post below links to the code and the run, so reviewers must be able to open them.

**GitHub** — https://github.com/Harsh4-Dev/System-One-DLE

1. Merge the work: **Pull requests → New pull request → base `main` ← compare `claude/intelligent-bell-xzh81w` → Create → Merge**.
2. **Settings → General → Danger Zone → Change visibility → Public.**
3. Optional: **Add file → Create new file → `LICENSE` → Choose a license template → Apache 2.0** (same licence as Laya).
4. Optional: **Releases → Draft a new release**, tag `v1.0-kaggle-run`, attach `results/kaggle_run/adar_results_small.zip`.

**Kaggle notebook** — https://www.kaggle.com/code/harshavardhan88/adar-system-one

1. Open it → **Share** (top right) → **Public** → Save. The executed version shows every cell's output: that is the proof of the run.

---

## 1. typed-decisions: reply in discussion #2

- Benchmark: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Thread: https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/2 ("Laya benchmark results")
- Why this thread: the maintainer added Laya typed-decisions at 0.766 as *self-reported, not re-run*, and asked
  for **KL and Brier from the full distributions**. Our run re-ran it independently (0.7665) and has those numbers.

Steps

1. Sign in at https://huggingface.co (free account).
2. Open the thread link above.
3. Open `submissions/typed_decisions/DISCUSSION.md`, copy everything **below the first line** ("Reply to: …").
4. Paste it into the comment box at the bottom of the thread → **Comment**.
5. After step 2 below, edit the comment and replace `<link to the PR>` with the PR link.

What it contains (all from our run): accuracy 0.7665, KL 0.117, Brier 0.0615, ECE 0.2137, p50 latency 148 ms,
and the calibration table (base Laya ECE 0.173 → 0.021).

## 2. typed-decisions: community PR with `.eval_results`

The benchmark collects scores from YAML files in model repos (format confirmed in
[discussion #10](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/10) and the
[Hub docs](https://huggingface.co/docs/hub/eval-results)). A PR shows on the model page and the benchmark's
leaderboard with a **community** badge while it is open; the model's author can merge it.

- Model repo: https://huggingface.co/convaiinnovations/laya-typed-decisions
- File to add: `submissions/typed_decisions/eval_results_pr/typed-decisions.yaml` (four entries: `accuracy`, `kl_from_gold`, `brier`, `ece`)
- PR text: `submissions/typed_decisions/eval_results_pr/PR_DESCRIPTION.md`

**Option A: in the browser**

1. Open the model repo → **Files and versions** → **+ Add file** → **Create a new file**.
2. File name: `.eval_results/typed-decisions.yaml`. Paste the YAML file's content.
3. Commit title: `Add typed-decisions results (independent re-run)`; description: paste `PR_DESCRIPTION.md` (without its first line).
4. Choose **Open as a pull request** → **Commit**.
5. Copy the PR link into your discussion #2 comment (step 1.5).

**Option B: from Python** (needs a write token from https://huggingface.co/settings/tokens)

```python
from huggingface_hub import HfApi, login
login()  # paste the token when asked; never commit it
desc = open("submissions/typed_decisions/eval_results_pr/PR_DESCRIPTION.md").read().split("\n", 2)[2]
pr = HfApi().upload_file(
    path_or_fileobj="submissions/typed_decisions/eval_results_pr/typed-decisions.yaml",
    path_in_repo=".eval_results/typed-decisions.yaml",
    repo_id="convaiinnovations/laya-typed-decisions",
    commit_message="Add typed-decisions results (independent re-run)",
    commit_description=desc, create_pr=True)
print(pr)  # the PR URL
```

Check: the benchmark page's leaderboard should list `convaiinnovations/laya-typed-decisions` (community badge) within minutes.

## 3. Hugging Face model pages for the ADAR heads

The trained heads are in the full Kaggle output, not in Git (they are binaries).

1. **Get the heads.** Open https://www.kaggle.com/code/harshavardhan88/adar-system-one → **Output** → download
   `adar_outputs.zip`. Unzip it: `publish/adar-head-ag_news/` and `publish/adar-head-emotion/` each hold
   `model.safetensors`, `config.json`, `options.safetensors`, `experts/`.
2. **Use our cards.** Replace each folder's `README.md` with the generated card:
   `submissions/huggingface/adar-head-ag-news/README.md` → `publish/adar-head-ag_news/README.md`, and the same for emotion.
   The cards carry a `model-index` block with the run's accuracy and macro-F1 (shown as "Evaluation results" on the page).
3. **Create a write token:** https://huggingface.co/settings/tokens → **Create new token** → **Write**.
4. **Upload** (on your computer):

```python
from huggingface_hub import HfApi, login
login()
api = HfApi()
for task, folder in (("ag-news", "publish/adar-head-ag_news"), ("emotion", "publish/adar-head-emotion")):
    repo = f"DEV-HV/adar-head-{task}"
    api.create_repo(repo, repo_type="model", exist_ok=True)   # private=True to start private
    api.upload_folder(folder_path=folder, repo_id=repo, commit_message="ADAR head from the Kaggle proof run")
    print("https://huggingface.co/" + repo)
```

   Or in the browser: https://huggingface.co/new → name `adar-head-ag-news` → **Create** → **Files → Add file → Upload files** → drag the folder's files → **Commit**.
5. Check each page shows the **Evaluation results** box, then **Settings → Make public** when you want it shared.
6. Using a published head: `huggingface_hub.snapshot_download("DEV-HV/adar-head-ag-news")` returns a folder; pass it to `ADARClassifier.from_pretrained(folder)`.

## 4. RouterArena (next phase, the head-turner)

- Leaderboard: https://routeworks.github.io/leaderboard
- Repository and rules: https://github.com/RouteWorks/RouterArena
- Why: RouterArena ranks routers that choose an LLM per query. ADAR's core idea, per-input routing with a calibrated
  "is this model right?" probability per candidate, is exactly that task, at a scale people notice.

**Rule that matters:** RouterArena is evaluation-only. Do **not** train, fit or tune anything on its data (including the
label files); such entries are rejected or withdrawn. Train on other data, e.g. RouterBench
(https://huggingface.co/datasets/withmartian/routerbench, recorded answers of many LLMs).

Steps (from the RouterArena README)

1. Install `uv`, clone the repo, `uv sync`.
2. Download the dataset from its Hugging Face page (linked in the README), then
   `uv run python ./scripts/process_datasets/prep_datasets.py`.
3. Only if you use their inference pipeline: copy `.env.example` to `.env` and add API keys.
4. Create `router_inference/config/adar.json` with `pipeline_params` (`router_name`, `router_cls_name`, `models`).
   Add each model's price per million tokens to `model_cost/model_cost.json`; each model must exist in
   `universal_model_names.py` (a new model also needs an endpoint in `llm_inference/model_inference.py`).
5. Write the router: a class inheriting `BaseRouter` with `_get_prediction()` (see `router_inference/router/example_router.py`),
   registered in `router_inference/router/__init__.py`. For ADAR: one noul head per candidate LLM, trained on RouterBench,
   then pick the cheapest model whose P(right) clears the threshold.
6. Predictions: `uv run python ./router_inference/generate_prediction_file.py adar sub_10` (809 queries, for testing), then `full` (8,400) and `robustness` (420).
7. Validate: `uv run python ./router_inference/check_config_prediction_files.py adar full`.
8. Inference: `uv run python ./llm_inference/run.py adar` (cached in `./cached_results/`).
9. Optional local score: `uv run python ./llm_evaluation/run.py adar full`.
10. Pull request to `main` adding `router_inference/config/adar.json`, `router_inference/predictions/adar.json`
    (with `generated_result` filled) and `router_inference/predictions/adar-robustness.json`.
11. Comment `/evaluate` on the PR. The workflow scores it and posts the result; the leaderboard updates on approval.

Budget: step 8 calls the chosen LLMs unless RouterArena's cache has them. Start with `sub_10`.

---

## Checklist

- [ ] GitHub repo merged and public; Kaggle notebook public (section 0)
- [ ] Reply posted in typed-decisions discussion #2 (section 1)
- [ ] `.eval_results` PR opened on `convaiinnovations/laya-typed-decisions`; its link added to the reply (section 2)
- [ ] `DEV-HV/adar-head-ag-news` and `DEV-HV/adar-head-emotion` uploaded with the generated cards (section 3)
- [ ] Screenshots of the reply, the PR and the leaderboard widget for the presentation
- [ ] RouterArena plan (section 4) as "next steps" in the talk

What to say in the presentation: *"Our evaluation independently reproduces the published accuracy of Laya's
fine-tuned checkpoint on the typed-decisions benchmark (0.7665 vs 0.766), and we contributed the KL, Brier and ECE
the maintainers asked for. Our trained heads are on Hugging Face. Next, the same head goes to RouterArena."*
Do not claim a rank on typed-decisions: the row belongs to Laya's authors; our contribution is the verification.
