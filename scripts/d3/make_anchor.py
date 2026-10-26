"""D3 anchor sheet: re-label a random subset of batch-1 and batch-2 items to identify the annotator-standard drift
(scripts/d3/drift.py found batch 2 stricter by ~.06 gain units after adjusting for cells and item covariates;
which standard is right is not identified without items labelled under both).

Design: simple random samples without replacement of n_per_batch labelled items from each batch (each batch's labelled
set is itself a design sample, so the anchor subsample inherits its design weights 1/pi_t), shuffled together into one
blinded sheet with the usual instructions and 5% exact repeats (witness.d3.write_sheet).  Annotators see neither
the batch nor the original label.  Analysis (scripts/d3/anchor.py): per batch, mean of (new gain - original
gain) with the 1/pi_t weights; their difference estimates the drift between the batch-1 and batch-2 standards under
one current standard; with two independent annotators also the inter-annotator agreement and a second standard.
Writes anchor_key.json (row -> item, batch, original label; read only by the analysis) and anchor_sheet.xlsx.
usage: python scripts/d3/make_anchor.py configs/d3_duretrieval.yaml <n_per_batch> <annotated_1.xlsx> <annotated_2.xlsx>
"""
import json
import os
import sys

import numpy as np
from witness import d3 as WD  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
n_per = int(sys.argv[2])
D = WD.load_frame(cfg)
batches = [WD.read_batch(D, t + 1, x) for t, x in enumerate(sys.argv[3:5])]
rng = np.random.default_rng([cfg["batch"]["seed"], 99])
out = cfg["out_dir"]
for path in (f"{out}/anchor_key.json", f"{out}/anchor_sheet.xlsx"):
    assert not os.path.exists(path), path
pick, meta = [], {}
for bt in batches:
    assert len(bt.items) >= n_per
    sel = rng.choice(len(bt.items), n_per, replace=False)
    for i in sel:
        it = int(bt.items[i])
        pick.append(it)
        meta[it] = dict(batch=bt.stage, y_orig=int(bt.y[i]), pi=float(bt.pi[it]), share=n_per / len(bt.items))
assert len(set(pick)) == len(pick)
key, n_unique = WD.write_sheet(D, pick, f"{out}/anchor_sheet.xlsx", cfg["batch"]["repeat_share"], rng)
assert n_unique == len(pick)
for k in key:
    it = D.pos[(k["qi"], k["di"])]
    k.update(meta[it])
json.dump(dict(n_per_batch=n_per, n_unique=len(pick), n_rows=len(key), rows=key), open(f"{out}/anchor_key.json", "w"),
          ensure_ascii=False)
yo = np.array([meta[i]["y_orig"] for i in pick])
bb = np.array([meta[i]["batch"] for i in pick])
print(f"anchor sheet: {len(key)} rows, {len(pick)} unique; original label shares batch 1 "
      f"{np.bincount(yo[bb == 1], minlength=3).tolist()}, batch 2 {np.bincount(yo[bb == 2], minlength=3).tolist()}")
print("DONE")
