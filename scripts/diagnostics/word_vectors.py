"""Course principle 'word vectors vs sentence vectors' on D1: a controlled contrast with one tokenizer and one vocabulary.

Probe systems (not evaluated systems; written to <out_dir>/probes/<name>.npy so the evaluator's frame is unchanged):
  wv_mean     cosine of the mean of bge-base-zh-v1.5 INPUT token embeddings (static word-piece vectors, no context)
  wv_idf      the same with IDF weights over the D1 candidate collection (a SIF-like weighting without PC removal)
The sentence-vector counterpart is the evaluated system bge_base_zh (same tokenizer and embedding table, contextualised
by the 12-layer encoder and trained for retrieval), so the contrast isolates contextualisation + retrieval training.
Reports true DCG@5 (rho = 1/2, full human labels) with query-bootstrap 95% CIs for the probes, bge_base_zh,
text2vec_zh, m3e_base and the BM25 variants, and paired differences wv_* - bm25_jieba and bge_base_zh - wv_*.
usage: python scripts/diagnostics/word_vectors.py configs/testbeds/d1_t2ranking.yaml configs/eval/insample_d1.yaml
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from witness import core as W  # noqa: E402
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
base = load_config(sys.argv[1])
cfg = load_config(sys.argv[2])
D = load_testbed(base, cfg)
items = D.items
out = f"{cfg['out_dir']}/probes"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(8)
path = f"{base['models_dir']}/{base['dense']['bge_base_zh']['repo']}"
tok = AutoTokenizer.from_pretrained(path)
emb = AutoModel.from_pretrained(path).get_input_embeddings().weight.detach().float().numpy()  # [V, 768]
# drop structural tokens only; [UNK] keeps its embedding row, as the encoder also sees it (one D1 passage is a
# digit string that tokenizes to a single [UNK])
special = set(tok.all_special_ids) - {tok.unk_token_id}


def ids(texts):
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
    return [[i for i in x if i not in special] for x in enc]


q_ids = ids(items["query"].tolist())
p_ids = ids([p[: base["passage_chars"]] for p in items["passage"]])
df = np.zeros(emb.shape[0])
for x in p_ids:
    df[list(set(x))] += 1
idf = np.log(1 + (len(p_ids) - df + 0.5) / (df + 0.5))  # [V]


def vec(x, weighted):
    assert len(x) > 0
    w = idf[x] if weighted else np.ones(len(x))
    v = (w[:, None] * emb[x]).sum(0) / w.sum()
    return v / np.linalg.norm(v)


for name, weighted in (("wv_mean", False), ("wv_idf", True)):
    qcache = {}
    s = np.zeros(len(items), dtype=np.float32)
    for i, (q, x) in enumerate(zip(q_ids, p_ids)):
        key = tuple(q)
        if key not in qcache:
            qcache[key] = vec(q, weighted)
        s[i] = qcache[key] @ vec(x, weighted)
    np.save(f"{out}/{name}.npy", s)
    print("scored", name)

u = 0.5 * D.u0 + 0.5 * D.u1
names = ["wv_mean", "wv_idf", "bge_base_zh", "text2vec_zh", "m3e_base", "bm25_char1", "bm25_char2", "bm25_jieba"]
scores = np.stack([np.load(f"{out}/{n}.npy") if n.startswith("wv_") else
                   np.load(f"{cfg['out_dir']}/scores/{n}.npy") for n in names])
disc = W.discounts(D.qi, scores, cfg["k"])
per_q = np.stack([np.bincount(D.qi, weights=disc[s] * u, minlength=D.n_q) for s in range(len(names))])  # [S, Q]
rng = np.random.default_rng(0)
boot = np.stack([per_q[:, rng.integers(0, D.n_q, D.n_q)].mean(1) for _ in range(2000)])  # [B, S]
print(f"true DCG@{cfg['k']} at rho=1/2 on {D.n_q} queries (query-bootstrap 95% CI):")
for s, n in enumerate(names):
    print(f"   {n:12s} {per_q[s].mean():.4f} [{np.quantile(boot[:, s], .025):.4f}, {np.quantile(boot[:, s], .975):.4f}]")
ix = {n: i for i, n in enumerate(names)}
for a, b in (("wv_mean", "bm25_jieba"), ("wv_idf", "bm25_jieba"), ("bge_base_zh", "wv_mean"), ("bge_base_zh", "wv_idf"),
             ("wv_idf", "wv_mean")):
    d = boot[:, ix[a]] - boot[:, ix[b]]
    print(f"   {a} - {b}: {per_q[ix[a]].mean() - per_q[ix[b]].mean():+.4f} [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")
print("DONE")
