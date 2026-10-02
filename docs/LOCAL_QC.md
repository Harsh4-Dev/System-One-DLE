# Quality checks done before the Kaggle run

The environment where ADAR was built could not download Laya's weights or Hugging Face datasets
(network policy). Everything that could be checked without them was checked; this page lists
what was verified, how, and what only the Kaggle run can confirm.

## Verified

| Check | How | Result |
|---|---|---|
| Unit tests | `python -m pytest -q tests` (12 tests) | all pass |
| Crash and resume | killed a real run mid-stage and restarted it; unit tests for interrupted expert training, half-written embedding shards, failed / blocked stages, restoring a previous Kaggle output | finished stages and models skipped; only missing work re-ran |
| Head structure | test that permuting the answer options permutes the outputs identically | passes (options are treated as a set) |
| Published head = pipeline | loaded `publish/adar-head-ag_news` with `ADARClassifier` on raw text and compared with the pipeline's predictions | 100% same labels, max probability difference 0.00015 |
| Laya code path | built a tiny random checkpoint in Laya's own format with Laya's `build_model`, loaded it with the real `laya` 0.3.23 library, ran the whole pipeline on `backend="laya"` including zero-shot, CPU latency through `system_one`, and the typed-decisions runner (synthetic dataset in the real schema) | every stage completes |
| The notebook itself | executed `notebooks/adar_kaggle.ipynb` cell by cell with `nbclient` (smoke profile, CPU) | 34 cells, 0 errors, all stages done, the final cell classifies the example sentences correctly |
| Method works | mid-size run: AG News, 30,000 training rows, full 7,600-row test set, DistilBERT as a stand-in encoder | **13 / 13 QC gates pass** — see `qc/ag_news_standin/REPORT.md` |

## Problems found and fixed during QC

* The pipeline wrote the experts into a wrong sub-folder (`experts/<task>/x`). Fixed with a
  directory helper.
* The stage runner's `name` argument collided with the stages' own `name` argument. Renamed.
* The first notebook cell called `nvidia-smi` unconditionally and crashed on machines without a GPU. Guarded.
* scikit-learn's MLPClassifier on 200,000 TF-IDF columns ran for 18+ minutes on 30k rows (would be
  hours on Kaggle). Now uses the 20,000 most informative columns (chi²): 102 s, same accuracy range.
* If Laya runs out of GPU memory it silently moves to CPU; the encoder wrapper now follows the
  model's real device instead of sending tensors to the GPU.
* Report labels said "Laya" even when the stand-in encoder was used. Labels now name the real encoder.
* An f-string used nested double quotes, which only Python 3.12+ accepts. Fixed for 3.10/3.11.
* Saved files were readable only by their owner (0600). Now 0644.

## Not verified here (needs the Kaggle run)

* **Real Laya numbers.** The stand-in encoder (DistilBERT, 66M, not trained for decisions) is much
  weaker than Laya; the Kaggle run decides how ADAR does with Laya underneath.
* **The Emotion dataset loader** (Hugging Face `dair-ai/emotion`): column names follow the dataset
  card but could not be downloaded here.
* **typed-decisions numbers.** The scorer follows the leaderboard's stated definitions; whether it
  reproduces the published 0.766 / 0.062 / 0.213 is checked in the run
  (see `LEADERBOARD_SUBMISSION.md`, step 1).
* **GPU speed and both-GPU sharding.** The multi-device code path ran on CPU only.
