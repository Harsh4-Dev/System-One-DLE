# After the fresh Kaggle run: what to rebuild (handoff)

Order of work once `python tools/kaggle_run.py` has finished and
`results/kaggle_run/summary.json` comes from the fresh run (`"source": "fresh Kaggle run …"`).

## 0. Check the run before using it

- `summary.json` → `stage_status`: every stage `done`; `standin_encoder` is `false`.
- typed-decisions: `convaiinnovations/laya-typed-decisions` → `shipped.accuracy` ≈ 0.766 (published).
  If not, stop: the scorer or the download is wrong.
- Laya zero-shot on Emotion should stay near Laya's own published 57.3%.

## 1. Regenerated automatically by `kaggle_run.py`

- `submissions/typed_decisions/DISCUSSION.md`, `entry.json`, `submissions/huggingface/*/README.md`
- `docs/ADAR_Project_Report.docx` (+ figures in `build/figs/`)

Commit them.

## 2. The deck (claude.ai artifact, Slides)

Deck: https://claude.ai/artifact/Apmff51782q7q3aasjaLHJ (read it first; the owner may have edited it).

The owner asked for:

1. **Every number from `results/kaggle_run/summary.json`** of the fresh run. No number from the old log.
2. **Clean and professional:** less colour pop. Cream background, ink text, one accent (ADAR),
   muted tones for everything else; fewer filled colour blocks, thin outlines instead.
3. **Self-explanatory, with insight:** each content slide ends with a one-line
   "What this shows" takeaway; charts get a "how to read" line; speaker notes carry the script.
4. **New slides:**
   - *Key terms* (System-One model, calibration/ECE, log-loss, cascade, out-of-fold) early in the deck.
   - *Where ADAR fits*: the class's other projects grouped by syllabus unit (below), and how ADAR
     differs: it combines architectures from all four units and decides between them per input.
     Factual, no ranking of classmates.
   - *Leaderboard decision*: why typed-decisions (Discussion with the missing Laya metrics) and the
     HF model pages now, and RouterArena next; why not a "Laya + ADAR-Cal" row (accuracy below the prior).
5. Keep: problem → System-One background → idea → architecture → head → training → router → setup →
   results (AG News, honesty/cost, Emotion, ablations) → leaderboard → fit/syllabus → limits → demo.

### Other class projects, by syllabus unit (for the "Where ADAR fits" slide)

| Unit | Projects |
|---|---|
| II (CNNs in vision) | Neural Style Transfer, Fast Neural Style Transfer, R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN |
| III (RNNs, attention, transformers) | Image Captioning with transformers, GPT-2 summariser, Time-series forecasting, Speech recognition, Machine translation, BERT classifier, RNN vs LSTM vs GRU language modelling, Echo-state networks |
| IV (GANs, deep RL) | GAN, DCGAN, Q-learning grid world*, DQN (CartPole/MountainCar), REINFORCE (CartPole) |
| outside the listed topics | Text-to-music generation |
| **I + II + III + IV** | **ADAR** (TextCNN, BatchNorm, clipping · frozen pretrained network + new head · RNN/LSTM/GRU, attention, transformer, BERT-family encoder · REINFORCE router) |

\* tabular Q-learning is not *deep* RL; the syllabus lists deep RL.

Closest overlaps, and how ADAR differs: *BERT as a classifier* (ADAR uses a BERT-family encoder as
one input and is compared against a probe on it); *RNN vs LSTM vs GRU* (ADAR uses all three as experts
and learns per input which to trust); *REINFORCE on CartPole* (ADAR uses REINFORCE for a real
cost-accuracy routing decision).

## 3. Leaderboard posts

- Post `submissions/typed_decisions/DISCUSSION.md` as a new Discussion on
  `huggingface.co/datasets/LocalLLaMA/typed-decisions` (needs the owner's HF login, or `HF_TOKEN`
  with `huggingface_hub.HfApi().create_discussion(repo_id="LocalLLaMA/typed-decisions", repo_type="dataset", title=..., description=...)`;
  ask the owner before posting, since it is public).
- Update the two model cards (`submissions/README.md`, part B).
