# Status after the fresh Kaggle run (handoff)

The fresh proof run finished on Kaggle (kernel `adar-system-one`, GPU T4 ×2, 63 minutes, all 19
stages done, real Laya encoder). Checks passed:

- `results/kaggle_run/summary.json` → every stage `done`, `standin_encoder: false`.
- typed-decisions: `laya-typed-decisions` accuracy 0.7665 (the board lists 0.766), so the scorer agrees.
- Laya zero-shot on Emotion is 58.05% (Laya's model card: 57.3%).

## Rebuilt from that run (nothing typed by hand)

| What | Built by |
|---|---|
| `results/kaggle_run/` (summary, per-task reports, figures, kernel log) | `tools/collect_results.py` |
| `README.md` results, "why" table and demo output; `docs/MAIN.md` sections 1, 3, 4 | `tools/report/readme_results.py` |
| `docs/ADAR_explained.html` (artifact https://claude.ai/artifact/Hjvfc4zov1kefnSsrZggYL) | `tools/report/update_explainer.py` |
| `docs/ADAR_Project_Report.docx` + `build/figs/` | `tools/report/make_figures.py`, `tools/report/build_report.js` |
| Deck (artifact https://claude.ai/artifact/Apmff51782q7q3aasjaLHJ, 21 slides) | `tools/report/build_slides.py` |
| `submissions/typed_decisions/DISCUSSION.md`, `entry.json`, `submissions/huggingface/*/README.md` | `tools/make_submissions.py` |

`python tools/kaggle_run.py --user <name> [--deck <deck>/project]` repeats all of it for a new run.

## Still to do by the owner

1. Post `submissions/typed_decisions/DISCUSSION.md` as a new Discussion on
   `huggingface.co/datasets/LocalLLaMA/typed-decisions` (public, so post it yourself or confirm first).
2. Upload both heads with their model cards (`docs/GITHUB_AND_HF_UPLOAD.md`). The run did not upload
   them (`hf_uploads` is empty), so the slides say "ready to publish", not "published".
3. Fill the deck's cover placeholders (team names, course code, date) and the report's title page.
4. Revoke the Kaggle API token used for the run (Kaggle → Settings → API).
