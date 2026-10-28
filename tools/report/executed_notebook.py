"""Rebuilds the executed Kaggle notebook (with outputs) from the run's own artifacts.

    python tools/report/executed_notebook.py [kernel.log] [results dir] [out.ipynb]

Text outputs come from the Kaggle kernel log (results/kaggle_run/kernel.log, every stdout/stderr line of the run in
order), split into cells at the stage markers the pipeline prints. Tables and figures the notebook displayed come
from the files the run saved (results/kaggle_run/<task>/REPORT.md and *.png, SUMMARY.md). Nothing is re-computed.
"""
import base64
import copy
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOISE = ("Debugger warning", "frozen modules", "-Xfrozen_modules", "PYDEVD_DISABLE_FILE_VALIDATION",
         "Debugging will proceed", "MissingIDFieldWarning", "_validate(nbdict",
         # Kaggle converts the notebook after the run; its messages are not part of any cell
         "NbConvertApp", "mistune.py", "filter_links.py", "cells[i][c] = re.sub", "text = re.sub(r'_'")


def chunks(log):
    out = []
    for e in json.load(open(log)):
        if e["stream_name"] == "stderr" and any(n in e["data"] for n in NOISE):
            continue
        out.append((e["stream_name"], e["data"]))
    return out


def split_at(text_chunks, markers):
    """Split the ordered chunks into len(markers)+1 segments; segment k starts at the line holding markers[k-1]."""
    segs, cur, k = [], [], 0
    for name, data in text_chunks:
        if k < len(markers) and markers[k] in data:
            i = data.index(markers[k])
            line_start = data.rfind("\n", 0, i) + 1
            if line_start:
                cur.append((name, data[:line_start]))
            segs.append(cur)
            cur, k = [(name, data[line_start:])], k + 1
            while k < len(markers) and markers[k] in data[line_start:]:  # several markers in one chunk
                rest = cur.pop()[1]
                j = rest.index(markers[k])
                ls = rest.rfind("\n", 0, j) + 1
                if ls:
                    cur.append((name, rest[:ls]))
                segs.append(cur)
                cur, k = [(name, rest[ls:])], k + 1
        else:
            cur.append((name, data))
    segs.append(cur)
    while len(segs) < len(markers) + 1:
        segs.append([])
    return segs


def streams(seg):
    outs = []
    for name, data in seg:
        if outs and outs[-1]["name"] == name:
            outs[-1]["text"].append(data)
        else:
            outs.append({"output_type": "stream", "name": name, "text": [data]})
    for o in outs:
        o["text"] = "".join(o["text"]).splitlines(keepends=True)
    return [o for o in outs if "".join(o["text"]).strip()]


def md(text):
    return {"output_type": "display_data", "metadata": {}, "data": {"text/markdown": text.splitlines(keepends=True), "text/plain": ["<IPython.core.display.Markdown object>"]}}


def png(path):
    return {"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(open(path, "rb").read()).decode(), "text/plain": ["<IPython.core.display.Image object>"]}}


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "kaggle_run", "kernel.log")
    res = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "kaggle_run")
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "notebooks", "adar_kaggle_executed.ipynb")
    nb = json.load(open(os.path.join(ROOT, "notebooks", "adar_kaggle.ipynb")))
    cells = nb["cells"]
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    src = lambda i: "".join(cells[i]["source"])
    # one marker per code cell after the first (the line its output starts with); None = the cell prints nothing
    markers, owners = [], [code[0]]
    for i in code[1:]:
        s = src(i)
        m = None
        if s.startswith("%%writefile"):
            m = "Writing " + s.split()[1]
        elif "pipe.stage_experts" in s:
            m = "[run ] experts:"
        elif "pipe.stage_embed" in s:
            m = "[run ] embed:"
        elif "pipe.stage_head" in s:
            m = "[run ] head:"
        elif "pipe.stage_report" in s:
            m = "[run ] report:"
        elif "stage_typed_decisions" in s:
            m = "[run ] typed_decisions:"
        elif "stage_bundle" in s:
            m = "[run ] bundle"
        elif "ADARClassifier" in s:
            m = "{'label':"
        elif "PROFILE" in s:
            m = None  # the config cell: its output follows the last %%writefile, handled below
        if m:
            markers.append(m)
            owners.append(i)
    segs = split_at(chunks(log), markers)
    outputs = {i: streams(seg) for i, seg in zip(owners, segs)}
    # the config cell's own prints sit at the end of the last %%writefile segment, after its "Writing ..." line
    cfg_cell = next(i for i in code if "PROFILE" in src(i))
    last_wf = max(i for i in owners if src(i).startswith("%%writefile"))
    wf_out = outputs.get(last_wf, [])
    if wf_out:
        text = "".join(wf_out[0]["text"])
        first, _, rest = text.partition("\n")
        wf_out[0]["text"] = [first + "\n"]
        outputs[cfg_cell] = ([{"output_type": "stream", "name": "stdout", "text": rest.splitlines(keepends=True)}] if rest.strip() else []) + wf_out[1:]
        outputs[last_wf] = wf_out[:1]
    # displays: the report tables and figures, the run summary (files the run itself wrote)
    for i in code:
        s = src(i)
        if s.startswith("%%writefile"):
            continue
        if "pipe.stage_report" in s:
            for task in ("ag_news", "emotion"):
                rep = os.path.join(res, task, "REPORT.md")
                if os.path.exists(rep):
                    outputs[i] = outputs.get(i, []) + [md(open(rep).read().split("![reliability]")[0])]
                    for f in ("reliability.png", "accuracy_vs_latency.png", "risk_coverage.png"):
                        if os.path.exists(os.path.join(res, task, f)):
                            outputs[i].append(png(os.path.join(res, task, f)))
        if "stage_bundle" in s and os.path.exists(os.path.join(res, "SUMMARY.md")):
            outputs[i] = outputs.get(i, []) + [md(open(os.path.join(res, "SUMMARY.md")).read()[:20000])]
    new = copy.deepcopy(nb)
    n = 0
    for i, c in enumerate(new["cells"]):
        if c["cell_type"] == "code":
            n += 1
            c["execution_count"] = n
            c["outputs"] = outputs.get(i, [])
    note = ("> **Executed run.** Outputs of the full Kaggle run (kernel `harshavardhan88/adar-system-one`, GPU T4 ×2, "
            "63 minutes, 19/19 stages done). Text outputs are the kernel's own log (`results/kaggle_run/kernel.log`); "
            "tables and figures are the files the run wrote (`results/kaggle_run/`). Live version: "
            "https://www.kaggle.com/code/harshavardhan88/adar-system-one\n")
    new["cells"].insert(1, {"cell_type": "markdown", "metadata": {}, "source": note.splitlines(keepends=True)})
    json.dump(new, open(out, "w"), indent=1, ensure_ascii=False)
    print("wrote", out, {k: len(v) for k, v in outputs.items() if v})


if __name__ == "__main__":
    main()
