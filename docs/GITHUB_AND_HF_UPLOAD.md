# Publishing: GitHub repository and Hugging Face models

## Part 1 — GitHub

### 1.1 Get the code onto GitHub

The work lives on the branch `claude/intelligent-bell-xzh81w` of `Harsh4-Dev/System-One-DLE`.

If a push from Claude was refused with *"Claude doesn't have GitHub access"*: reconnect GitHub at
`https://claude.ai/connect-github`, and install the **Claude GitHub App** on this repository from the
same page. After that the branch can be pushed.

To push from your own computer instead:

```bash
git clone https://github.com/Harsh4-Dev/System-One-DLE.git
cd System-One-DLE
git checkout claude/intelligent-bell-xzh81w      # once the branch is on GitHub
```

Merge it into `main` with a pull request on GitHub (**Pull requests → New → base: main, compare:
claude/intelligent-bell-xzh81w → Create → Merge**).

### 1.2 Make the repository ready to share

| Step | How | Why |
|---|---|---|
| Add a licence | **Add file → Create new file → `LICENSE` → Choose a licence template → Apache 2.0** | Laya is Apache-2.0; the same licence keeps the head and code compatible. Your choice, but leaderboards and the Hub expect one. |
| Make it public | **Settings → General → Danger Zone → Change visibility** | leaderboard entries and model cards link here |
| Describe it | the gear next to **About**: description, topics `deep-learning`, `system-one`, `laya`, `calibration`, `routing` | people find it |
| Attach results | **Releases → Draft a new release → tag `v0.1`**, attach `adar_outputs.zip` from Kaggle | a permanent, citable copy of the exact numbers |

### 1.3 Keep the notebook in sync

`notebooks/adar_kaggle.ipynb` is generated from the `adar/` package. After changing any file in
`adar/`:

```bash
python -m pytest -q tests          # all tests must pass
python tools/build_notebook.py     # regenerates the notebook
git add -A && git commit -m "..." && git push
```

Then re-import the notebook on Kaggle (**File → Import Notebook**), or in a Kaggle notebook use
**File → Link to GitHub** to pull it from the repository.

---

## Part 2 — Hugging Face

### 2.1 Account and token

1. Create an account at `huggingface.co`.
2. **Settings → Access Tokens → Create new token**, type **Write** (or fine-grained with
   "write access to contents of all repos you can write to"). Copy it once; you cannot view it again.
3. Never put the token in the notebook or the repository. Use the Kaggle secret `HF_TOKEN` or
   `huggingface_hub.login()` on your computer.

### 2.2 What gets uploaded

Each task produces `publish/adar-head-<task>/`:

| File | Content |
|---|---|
| `model.safetensors` | the ADAR head weights (about 2M parameters) |
| `config.json` | architecture, labels, option descriptions, fitted temperature, encoder and expert settings |
| `options.safetensors` | Laya embeddings of the option descriptions |
| `experts/` | TextCNN / RNN / LSTM / GRU weights (one per fold) and their vocabulary |
| `README.md` | the generated **model card**: description, results table, usage, limitations |
| `REPORT.md`, `*.png` | the full comparison and figures for that task |

Laya itself is **not** copied; the model card points to `convaiinnovations/laya` as `base_model`,
and loading the head downloads Laya from its own repository.

### 2.3 Option A — automatic upload from Kaggle

1. Kaggle: **Add-ons → Secrets → `HF_TOKEN`**, tick it for the notebook.
2. In the configuration cell: `HF_REPO_PREFIX = "<your-username>/adar-head"`.
3. Run the notebook. Stage 8 creates `<your-username>/adar-head-ag-news` and
   `<your-username>/adar-head-emotion` as **private** repositories and uploads the folders.

### 2.4 Option B — manual upload from your computer

```bash
pip install -U huggingface_hub
unzip adar_outputs.zip -d adar_outputs
```

```python
from huggingface_hub import HfApi, login

login()  # paste the write token when asked
api = HfApi()
for task in ("ag_news", "emotion"):
    repo = f"<your-username>/adar-head-{task.replace('_', '-')}"
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=f"adar_outputs/publish/adar-head-{task}", repo_id=repo)
```

This is the same API the notebook uses in Option A.

### 2.5 Check the model card, then go public

1. Open the model page on the Hub and read the card. Check that:
   * the results table shows your **Kaggle** numbers (not the local stand-in QC numbers);
   * the `base_model: convaiinnovations/laya` and `license: apache-2.0` fields are present;
   * the limitations section is still true for your run.
2. **Settings → Change model visibility → Public**.
3. Add the model link to the GitHub README and to your leaderboard entries.

### 2.6 How someone uses the published head

```bash
pip install laya torch transformers safetensors huggingface_hub
git clone https://github.com/Harsh4-Dev/System-One-DLE && cd System-One-DLE
```

```python
from huggingface_hub import snapshot_download
from adar.inference import ADARClassifier

path = snapshot_download("<your-username>/adar-head-ag-news")
clf = ADARClassifier.from_pretrained(path)
clf.predict(["The striker scored twice as the home side won the derby."])
# -> label, calibrated probabilities, P(each expert is right), routing weights, escalate flag
```

### 2.7 Optional: a live demo

A Hugging Face **Space** (Gradio) that loads the head and shows the probabilities, the routing
weights and the escalate flag makes the project easy to try. It needs a CPU Space with about 4 GB
of RAM for Laya.
