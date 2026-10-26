"""Rerank the top-k of a first-stage DuRetrieval system with Qwen3-Reranker-0.6B (yes/no logit).

Score s = softmax([logit_no, logit_yes])[1] at the last position (official Qwen3-Reranker recipe, prompt in
src/witness/rerank_prompt.py).  first_stage is a dense key (systems/<key>/) or bm25_<tok> (systems/bm25/<tok>.npz).
Writes {duretrieval_dir}/systems/rerank/<first_stage>.npz with idx [Q, k] int32 (reranked order) and score [Q, k];
a partial file every 500 queries makes the run resumable.  The GPU is shared: the process is capped at gpu_frac.
usage: PYTHONPATH=src python scripts/duretrieval/rerank.py configs/d3_duretrieval.yaml <first_stage> [k=20]
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from witness import duretrieval as DR  # noqa: E402
from witness.config import load_config  # noqa: E402
from witness.rerank_prompt import MAX_LEN, build  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

cfg = load_config(sys.argv[1])
first = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
root = cfg["duretrieval_dir"]
BATCH = 8

# shared GPU: hard cap this process
torch.cuda.set_per_process_memory_fraction(float(cfg["gpu_frac"]))
torch.set_num_threads(4)

corpus, queries, _ = DR.load_retrieval(root)
ctext, qtext = corpus["text"].tolist(), queries["text"].tolist()
idx, _ = DR.bm25_topk(root, first[5:]) if first.startswith("bm25_") else DR.dense_topk(root, first)
idx = idx[:, :K]  # [Q, K]

path = f"{cfg['models_dir']}/Qwen3-Reranker-0.6B"
tok = AutoTokenizer.from_pretrained(path, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
assert yes_id != tok.unk_token_id and no_id != tok.unk_token_id

os.makedirs(f"{root}/systems/rerank", exist_ok=True)
PARTIAL = f"{root}/systems/rerank/{first}.partial.npz"
out_idx = np.zeros_like(idx)
out_score = np.zeros(idx.shape, dtype=np.float32)
start = 0
if os.path.exists(PARTIAL):  # resume: queries before `done` are already scored
    z = np.load(PARTIAL)
    assert z["idx"].shape == idx.shape
    out_idx, out_score, start = z["idx"].copy(), z["score"].copy(), int(z["done"])
    print("resuming from query", start, flush=True)
t0 = time.time()
with torch.no_grad():
    for qi in range(start, idx.shape[0]):
        texts = [build(qtext[qi], ctext[d]) for d in idx[qi]]
        scores = []
        for b in range(0, len(texts), BATCH):
            enc = tok(texts[b : b + BATCH], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to("cuda")
            # lm_head on the last position only: full [B, L, V] logits (V=151k) alone exceed the GPU cap
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]  # [B, V]
            two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()  # [B, 2]
            scores.append(torch.softmax(two, dim=1)[:, 1].cpu().numpy())
        s = np.concatenate(scores)  # [K]
        order = np.argsort(-s, kind="stable")
        out_idx[qi], out_score[qi] = idx[qi][order], s[order]
        if qi % 200 == 0:
            print("query", qi, "of", idx.shape[0], round(time.time() - t0, 1), "s", flush=True)
        if qi % 500 == 499:
            np.savez(PARTIAL, idx=out_idx, score=out_score, done=qi + 1)
np.savez_compressed(f"{root}/systems/rerank/{first}.npz", idx=out_idx.astype(np.int32), score=out_score)
if os.path.exists(PARTIAL):
    os.remove(PARTIAL)
print("DONE", first, K, round(time.time() - t0, 1), "s")
