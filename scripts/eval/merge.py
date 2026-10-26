"""Merge chunked evaluator outputs <out_dir>/eval_<tag>_r<lo>-<hi>_raw.csv and print the main-table means.
usage: python scripts/eval/merge.py <out_dir> <tag> <n_reps>
"""
import glob
import sys

import pandas as pd

out, tag, n_reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(f"{out}/eval_{tag}_r*-*_raw.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
assert sorted(df.rep.unique()) == list(range(n_reps)), sorted(df.rep.unique())
assert not df.duplicated(["rep", "method", "budget"]).any()
df.to_csv(f"{out}/eval_{tag}_raw.csv", index=False)
cols = ["n_labels", "cert_dec_bh", "fdp_bh", "cert_dec_by", "fdp_by", "cover_pairs", "z_sd", "rmse", "tau"]
g = df.groupby(["budget", "method"])[cols]
tab = g.mean().join(g.sem()[["cert_dec_bh", "fdp_bh"]], rsuffix="_se")
pd.set_option("display.width", 250)
print(f"files={len(files)} reps={df.rep.nunique()}")
print(tab.round(4).to_string())
tab.to_csv(f"{out}/eval_{tag}.csv")
