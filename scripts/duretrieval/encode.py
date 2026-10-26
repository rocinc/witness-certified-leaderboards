"""Encode the DuRetrieval corpus and queries with one dense embedding model (sentence-transformers, fp16, L2-normalised).

Each model uses the query/passage convention of its model card (CFG below; texts truncated to 256 tokens).
Writes {duretrieval_dir}/systems/<key>/q.npy [Q, d] fp16, p.npy [N, d] fp16 and ids.json (corpus and query ids in row
order).  The GPU is shared: the process is capped at the config's gpu_frac of device memory.
usage: PYTHONPATH=src python scripts/duretrieval/encode.py configs/d3_duretrieval.yaml <key>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from witness import duretrieval as DR  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

cfg = load_config(sys.argv[1])
key = sys.argv[2]
root = cfg["duretrieval_dir"]
MD = cfg["models_dir"]
CFG = {
    "bge_m3": dict(repo=f"{MD}/bge-m3", q_prefix="", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "e5_small": dict(repo=f"{MD}/multilingual-e5-small", q_prefix="query: ", p_prefix="passage: ", prompt_name=None, max_len=256, batch=32),
    # batch 8: the 0.6B decoder stays under the 7.5% GPU cap on the shared card (batch 32 hit the cap)
    "qwen3_emb": dict(repo=f"{MD}/Qwen3-Embedding-0.6B", q_prefix="", p_prefix="", prompt_name="query", max_len=256, batch=8),
    # additional dense rankers, each with the query/passage convention of its model card
    "bge_small_zh": dict(repo=f"{MD}/bge-small-zh-v1.5", q_prefix="为这个句子生成表示以用于检索相关文章：", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "bge_base_zh": dict(repo=f"{MD}/bge-base-zh-v1.5", q_prefix="为这个句子生成表示以用于检索相关文章：", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "gte_small_zh": dict(repo=f"{MD}/gte-small-zh", q_prefix="", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "gte_base_zh": dict(repo=f"{MD}/gte-base-zh", q_prefix="", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "m3e_base": dict(repo=f"{MD}/m3e-base", q_prefix="", p_prefix="", prompt_name=None, max_len=256, batch=32),
    "e5_base": dict(repo=f"{MD}/multilingual-e5-base", q_prefix="query: ", p_prefix="passage: ", prompt_name=None, max_len=256, batch=32),
    "text2vec_zh": dict(repo=f"{MD}/text2vec-base-chinese", q_prefix="", p_prefix="", prompt_name=None, max_len=256, batch=32),
}
enc = CFG[key]

# shared GPU: hard cap this process (over the cap only this process fails, never a neighbour)
torch.cuda.set_per_process_memory_fraction(float(cfg["gpu_frac"]))
torch.set_num_threads(4)

out = f"{root}/systems/{key}"
os.makedirs(out, exist_ok=True)
corpus, queries, _ = DR.load_retrieval(root)
pids = corpus["id"].tolist()
ptxt = [enc["p_prefix"] + t for t in corpus["text"]]
qids = queries["id"].tolist()
qtxt = [enc["q_prefix"] + t for t in queries["text"]]

m = SentenceTransformer(enc["repo"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
m.max_seq_length = enc["max_len"]
t0 = time.time()
qe = m.encode(qtxt, batch_size=enc["batch"], normalize_embeddings=True, prompt_name=enc["prompt_name"], show_progress_bar=False)  # [Q, d]
print(key, "queries", qe.shape, round(time.time() - t0, 1), "s")
t0 = time.time()
pe = m.encode(ptxt, batch_size=enc["batch"], normalize_embeddings=True, show_progress_bar=False)  # [N, d]
print(key, "corpus", pe.shape, round(time.time() - t0, 1), "s")
np.save(f"{out}/q.npy", qe.astype(np.float16))
np.save(f"{out}/p.npy", pe.astype(np.float16))
with open(f"{out}/ids.json", "w") as fh:
    json.dump({"pids": pids, "qids": qids}, fh)
print("DONE", key)
