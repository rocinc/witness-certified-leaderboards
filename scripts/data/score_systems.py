"""Score every testbed (query, candidate) pair with one system; the scores define that system's ranking of the candidates.

Serves D1 (configs/testbeds/d1_t2ranking.yaml) and D2 (configs/testbeds/d2_dl19.yaml); the config's out_dir selects the testbed.
Systems: dense bi-encoders (cosine), BM25 under a tokenization (IDF over the testbed candidate collection), and the
Qwen3-Reranker cross-encoder (yes-probability).
Writes <out_dir>/scores/<name>.npy aligned with items.parquet rows.
usage: python scripts/data/score_systems.py <config.yaml> <name>
"""
import functools
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
name = sys.argv[2]
out = f"{cfg['out_dir']}/scores"
os.makedirs(out, exist_ok=True)
items = pd.read_parquet(f"{cfg['out_dir']}/items.parquet")
Q, P = items["query"].tolist(), [p[: cfg["passage_chars"]] for p in items["passage"]]
t0 = time.time()


def dense(spec):
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_per_process_memory_fraction(cfg["gpu_frac"])
    m = SentenceTransformer(f"{cfg['models_dir']}/{spec['repo']}", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 256
    uq = sorted(set(Q))
    qe = m.encode([spec["q_prefix"] + q for q in uq], batch_size=spec["batch"], normalize_embeddings=True,
                  prompt_name=spec.get("prompt_name"), show_progress_bar=False)
    qmap = dict(zip(uq, qe))
    pe = m.encode([spec["p_prefix"] + p for p in P], batch_size=spec["batch"], normalize_embeddings=True, show_progress_bar=False)
    qv = np.stack([qmap[q] for q in Q])
    return (qv.astype(np.float32) * pe.astype(np.float32)).sum(1)


RUN = re.compile(r"[一-鿿]+|[A-Za-z0-9]+")
CJK = re.compile(r"[一-鿿]")
WORD = re.compile(r"[a-z0-9]+")


@functools.lru_cache(maxsize=None)
def porter():
    # imported lazily so that the D1 tokenizers do not depend on nltk
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@functools.lru_cache(maxsize=None)
def stem(w):
    return porter().stem(w)


def tokenize(tok, s):
    out = []
    if tok == "ws":
        # English (D2): lowercased alphanumeric words
        return WORD.findall(s.lower())
    if tok == "ws_stem":
        # English (D2): "ws" words, Porter-stemmed (nltk default NLTK_EXTENSIONS mode)
        return [stem(w) for w in WORD.findall(s.lower())]
    if tok == "jieba":
        import jieba

        return [w.lower() for w in jieba.lcut(s) if RUN.search(w)]
    for run in RUN.findall(s):
        if not CJK.match(run[0]):
            out.append(run.lower())
        elif tok == "char1":
            out.extend(run)
        else:
            assert tok == "char2"
            out.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
    return out


def bm25(tok, k1=1.2, b=0.75):
    import collections

    docs = [tokenize(tok, p) for p in P]
    N = len(docs)
    df = collections.Counter(w for d in docs for w in set(d))
    avg = np.mean([len(d) for d in docs])
    qcache = {}
    s = np.zeros(N)
    for i, d in enumerate(docs):
        qt = qcache.setdefault(Q[i], tokenize(tok, Q[i]))
        tf = collections.Counter(d)
        for w in qt:
            if w in tf:
                idf = np.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s[i] += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
    return s


def rerank(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from witness.rerank_prompt import MAX_LEN, build

    torch.cuda.set_per_process_memory_fraction(cfg["gpu_frac"])
    path = f"{cfg['models_dir']}/{spec['repo']}"
    tk = AutoTokenizer.from_pretrained(path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
    yes, no = tk.convert_tokens_to_ids("yes"), tk.convert_tokens_to_ids("no")
    s = np.zeros(len(P), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(P), spec["batch"]):
            enc = tk([build(q, p) for q, p in zip(Q[i : i + spec["batch"]], P[i : i + spec["batch"]])], padding=True,
                     truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
            two = torch.stack([lg[:, no], lg[:, yes]], dim=1).float()
            s[i : i + spec["batch"]] = torch.softmax(two, dim=1)[:, 1].cpu().numpy()
            if i % 8000 == 0:
                print(name, i, len(P), round(time.time() - t0, 1), "s")
    return s


if name in cfg["dense"]:
    s = dense(cfg["dense"][name])
elif name.startswith("bm25_"):
    assert name[5:] in cfg["lexical"]
    s = bm25(name[5:])
else:
    assert name == "rerank_qwen3"
    s = rerank(cfg["reranker"])
assert np.isfinite(s).all() and len(s) == len(items)
np.save(f"{out}/{name}.npy", np.asarray(s, dtype=np.float32))
print("DONE", name, round(time.time() - t0, 1), "s")
