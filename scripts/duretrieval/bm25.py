"""BM25 (bm25s, k1=1.2, b=0.75) over the DuRetrieval corpus under one tokenization; saves the top-100 per query.

Tokenizations (latin/digit runs are lowercased words in every one; other characters are dropped):
  char1    every CJK character is a token
  char2    overlapping CJK character bigrams (a single-character run stays one token)
  jieba    jieba.lcut words containing a CJK/latin/digit character
  jieba_s  jieba.lcut_for_search words, same filter
Writes {duretrieval_dir}/systems/bm25/<tok>.npz with idx [Q, 100] int32 (corpus rows) and score [Q, 100] float32.
usage: PYTHONPATH=src python scripts/duretrieval/bm25.py configs/d3_duretrieval.yaml <tok>
"""
import os
import re
import sys
import time

import bm25s
import numpy as np
from witness import duretrieval as DR  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

cfg = load_config(sys.argv[1])
tok = sys.argv[2]
assert tok in ("char1", "char2", "jieba", "jieba_s"), tok
root = cfg["duretrieval_dir"]
K = 100
corpus, queries, _ = DR.load_retrieval(root)
ptxt = corpus["text"].tolist()
qtxt = queries["text"].tolist()

CJK = re.compile(r"[一-鿿]")
KEEP = re.compile(r"[一-鿿A-Za-z0-9]")


def _cjk_runs(s):
    # split into runs of CJK chars and runs of latin/digit chars; drop everything else
    return re.findall(r"[一-鿿]+|[A-Za-z0-9]+", s)


def t_char1(s):
    out = []
    for run in _cjk_runs(s):
        if CJK.match(run[0]):
            out.extend(run)
        else:
            out.append(run.lower())
    return out


def t_char2(s):
    out = []
    for run in _cjk_runs(s):
        if CJK.match(run[0]):
            out.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        else:
            out.append(run.lower())
    return out


if tok == "char1":
    f = t_char1
elif tok == "char2":
    f = t_char2
else:
    import jieba

    jieba.setLogLevel(60)
    jieba.initialize()
    cut = jieba.lcut if tok == "jieba" else jieba.lcut_for_search

    def f(s):
        return [w.lower() for w in cut(s) if KEEP.search(w)]


t0 = time.time()
P = [f(s) for s in ptxt]
Q = [f(s) for s in qtxt]
print(tok, "tokenized", round(time.time() - t0, 1), "s; mean doc toks", np.mean([len(x) for x in P]).round(1),
      "mean query toks", np.mean([len(x) for x in Q]).round(1))

t0 = time.time()
ret = bm25s.BM25(k1=1.2, b=0.75)
ret.index(P, show_progress=False)
res, sc = ret.retrieve(Q, k=K, show_progress=False, n_threads=4)  # [Q, K] each
print(tok, "indexed+retrieved", round(time.time() - t0, 1), "s")
os.makedirs(f"{root}/systems/bm25", exist_ok=True)
np.savez_compressed(f"{root}/systems/bm25/{tok}.npz", idx=res.astype(np.int32), score=sc.astype(np.float32))
print("DONE", tok)
