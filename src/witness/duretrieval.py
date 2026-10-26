"""DuRetrieval (D3) loaders shared by the upstream scripts in scripts/duretrieval/.

Layout under the config key `duretrieval_dir`:
  hf/DuRetrieval/data/{corpus,queries}-*.parquet   corpus (100,001 passages) and dev queries (2,000)
  hf/DuRetrieval-qrels/data/dev-*.parquet          qrels (9,839 rows: qid, pid, score)
  systems/<dense key>/{q,p}.npy, ids.json          fp16 L2-normalised embeddings (scripts/duretrieval/encode.py)
  systems/bm25/<tok>.npz                            BM25 top-100 idx/score (scripts/duretrieval/bm25.py)
  systems/rerank/<first stage>.npz                  reranked top-k idx/score (scripts/duretrieval/rerank.py)
  judgments/                                        judge items and LLM labels (scripts/duretrieval/judge_items.py)
Document and query indices are row positions in the corpus and queries parquet files.
usage: from witness import duretrieval as DR; corpus, queries, qrels = DR.load_retrieval(cfg["duretrieval_dir"])
"""
import glob

import numpy as np
import pandas as pd
import torch

K = 100
EXPECTED_SHAPES = ((100001, 2), (2000, 2), (9839, 3))  # corpus, queries, qrels


def _one(pattern):
    hits = glob.glob(pattern)
    assert len(hits) == 1, pattern
    return hits[0]


def load_retrieval(root):
    """Corpus (id, text), queries (id, text) and qrels (qid, pid, score) DataFrames."""
    corpus = pd.read_parquet(_one(f"{root}/hf/DuRetrieval/data/corpus-*.parquet"))
    queries = pd.read_parquet(_one(f"{root}/hf/DuRetrieval/data/queries-*.parquet"))
    qrels = pd.read_parquet(_one(f"{root}/hf/DuRetrieval-qrels/data/dev-*.parquet"))
    assert (corpus.shape, queries.shape, qrels.shape) == EXPECTED_SHAPES
    return corpus, queries, qrels


def load_qrels(root):
    """pids, qids and rel, where rel[qi] is the set of corpus row indices with qrels score > 0 for query row qi."""
    corpus, queries, qrels = load_retrieval(root)
    pids, qids = corpus["id"].tolist(), queries["id"].tolist()
    pidx = {p: i for i, p in enumerate(pids)}
    qidx = {q: i for i, q in enumerate(qids)}
    rel = [set() for _ in qids]
    for qid, pid, score in zip(qrels["qid"], qrels["pid"], qrels["score"]):
        if score > 0:
            rel[qidx[qid]].add(pidx[pid])
    assert all(len(s) > 0 for s in rel)
    return pids, qids, rel


def dense_topk(root, key, k=K):
    """Exact inner-product top-k of a dense system: idx [Q, k] int64, score [Q, k] float32."""
    # CPU on purpose: the GPU is shared with another job
    q = torch.from_numpy(np.load(f"{root}/systems/{key}/q.npy")).float()  # [Q, d]
    p = torch.from_numpy(np.load(f"{root}/systems/{key}/p.npy")).float()  # [N, d]
    idx, sc = [], []
    for i in range(0, q.shape[0], 200):
        v, j = torch.topk(q[i : i + 200] @ p.T, k, dim=1)  # [200, k]
        idx.append(j.numpy())
        sc.append(v.numpy())
    return np.concatenate(idx), np.concatenate(sc)


def bm25_topk(root, tok):
    """Stored BM25 top-100 of tokenization `tok`: idx [Q, 100] int32, score [Q, 100] float32."""
    z = np.load(f"{root}/systems/bm25/{tok}.npz")
    return z["idx"], z["score"]
