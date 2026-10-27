Title: Add typed-decisions results (independent re-run)

This adds `.eval_results/typed-decisions.yaml` so the model's typed-decisions scores show on its page and on the
[benchmark leaderboard](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), following the format asked for in
[discussion #10](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/10).

Scores from our independent re-run on the `test` split (400 cases, 2,000 decisions, full gold distributions):
accuracy 0.7665, KL from gold 0.117, Brier 0.0615, ECE 0.2137.
The accuracy matches the 0.766 you published. Details: [discussion #2](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/2).

Evaluation code and logs: https://github.com/Harsh4-Dev/System-One-DLE
