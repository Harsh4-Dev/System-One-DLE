"""Live demo: classify sentences with a trained ADAR head and show how it decided.

    python tools/demo.py --head path/to/adar-head-ag_news                 # built-in example sentences
    python tools/demo.py --head path/to/adar-head-ag_news "Your sentence"  # your own sentences
    python tools/demo.py --head path/to/adar-head-emotion -i               # type sentences live

For every sentence it prints ADAR's answer with a probability for every label, P(answer right), whether it
would escalate, and each syllabus expert's own vote, its P(right) and the routing weight the head gave it.
Runs on a CPU (about 1 s per sentence with Laya) or a GPU. The first run downloads Laya from Hugging Face.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXAMPLES = {
    "ag_news": ["NASA's new telescope captured images of a distant galaxy cluster.",
                "Real Madrid beat Barcelona 3-1 in the Champions League semi-final on Tuesday.",
                "Oil prices fell as OPEC agreed to raise output next quarter.",
                "Apple unveiled a faster chip for its laptops, and its shares rose 3% after the event."],
    "emotion": ["i finally got the job i wanted and i can't stop smiling",
                "i feel so alone since my best friend moved away",
                "i am furious that they cancelled my flight without any warning",
                "i feel a little nervous about tomorrow's exam"],
}
NAMES = {"textcnn": "TextCNN", "rnn": "Bi-RNN", "lstm": "Bi-LSTM", "gru": "Bi-GRU"}


def bar(p, width=24):
    n = round(p * width)
    return "█" * n + "·" * (width - n)


def show(clf, texts):
    t0 = time.perf_counter()
    res = clf.predict(texts)
    exp = clf.expert_probs(texts).cpu().numpy()  # B, E, K
    ms = (time.perf_counter() - t0) * 1000 / len(texts)
    kinds = clf.cfg["experts"]["kinds"]
    for i, (text, r) in enumerate(zip(texts, res)):
        print("\n" + "─" * 78)
        print(f"  {text}")
        print("─" * 78)
        for lab, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
            mark = "  ← answer" if lab == r["label"] else ""
            print(f"  {lab:>10}  {bar(p)}  {p * 100:5.1f}%{mark}")
        print(f"\n  P(answer is right) = {r['p_head_correct']:.3f}   escalate / ask a human: {'YES' if r['escalate'] else 'no'}")
        if "p_expert_correct" in r:
            print("\n  expert      its vote         P(expert right)   routing weight α")
            for e, k in enumerate(kinds):
                vote = clf.labels[int(exp[i, e].argmax())]
                agree = "" if vote == r["label"] else "  (disagrees)"
                print(f"  {NAMES.get(k, k):<10}  {vote:<10} {exp[i, e].max() * 100:5.1f}%   {r['p_expert_correct'][k]:>10.3f}   {r['routing_weights'][k]:>12.3f}{agree}")
    print(f"\n  {ms:.0f} ms per sentence on {clf.device} (Laya encoder + 4 experts x 3 folds + head)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="folder of a trained head, e.g. publish/adar-head-ag_news")
    ap.add_argument("-i", "--interactive", action="store_true", help="type sentences one by one")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("texts", nargs="*")
    a = ap.parse_args()
    from adar.inference import ADARClassifier
    print("loading the head, the 12 expert models and the Laya encoder ...")
    t0 = time.perf_counter()
    clf = ADARClassifier.from_pretrained(a.head, device="cpu" if a.cpu else None)
    print(f"ready in {time.perf_counter() - t0:.0f} s · labels: {', '.join(clf.labels)}")
    task = "emotion" if "sadness" in clf.labels or "joy" in clf.labels else "ag_news"
    if a.texts:
        show(clf, a.texts)
    elif not a.interactive:
        show(clf, EXAMPLES[task])
    if a.interactive:
        while True:
            try:
                text = input("\nsentence (empty line to quit)> ").strip()
            except EOFError:
                break
            if not text:
                break
            show(clf, [text])


if __name__ == "__main__":
    main()
