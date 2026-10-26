"""Fine-tune a cross-encoder on the T2Ranking train split and score every D1 item with it.

The model maps (query, passage) to P(y = k), k in {0, 1, 2}, with the D1 grade map rel 0,1 -> 0; 2 -> 1; 3 -> 2.
Training pairs: `n_per_rel` pairs per 0-3 grade sampled from qrels.train.tsv, after removing every train query whose
text equals a D1 query text (D1 uses dev qids only; the text check guards against duplicated queries across splits).
Writes
  <d1_dir>/predictors/ft_ce.npy   [n, 3] float32 class probabilities aligned with items.parquet rows
  <ckpt_dir>/                     fine-tuned model, tokenizer and the run config
usage: python scripts/data/finetune_ce.py configs/data/finetune_ce_d1.yaml
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
print("config", json.dumps(cfg))
print(f"torch {torch.__version__} cuda {torch.version.cuda}")
seed = cfg["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
rng = np.random.default_rng(seed)
D = cfg["data_dir"]
t0 = time.time()

items = pd.read_parquet(f"{cfg['d1_dir']}/items.parquet")
qrels = pd.read_csv(f"{D}/qrels.train.tsv", sep="\t")[["qid", "pid", "rel"]]
assert not qrels.duplicated(["qid", "pid"]).any()
queries = pd.read_csv(f"{D}/queries.train.tsv", sep="\t", quoting=3).set_index("qid")["text"]
assert queries.index.is_unique
shared_qid = set(qrels.qid) & set(items.qid)
print(f"train qids={qrels.qid.nunique()}  D1 qids={items.qid.nunique()}  shared qids={len(shared_qid)}")
assert not shared_qid
d1_text = set(items["query"].str.strip())
leak = queries[queries.str.strip().isin(d1_text)]
print(f"train queries whose text equals a D1 query: {len(leak)} (distinct texts {leak.str.strip().nunique()}); removed")
qrels = qrels[~qrels.qid.isin(set(leak.index))]
assert qrels.qid.isin(queries.index).all()

parts = []
for r in range(4):
    sub = qrels[qrels.rel == r]
    parts.append(sub.iloc[rng.choice(len(sub), size=cfg["n_per_rel"], replace=False)])
train = pd.concat(parts).reset_index(drop=True)
train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
need = set(train.pid)
texts = {}
for chunk in pd.read_csv(f"{D}/collection.tsv", sep="\t", quoting=3, chunksize=500_000,
                         dtype={"pid": np.int64, "text": str}):
    hit = chunk[chunk.pid.isin(need)]
    texts.update(zip(hit.pid, hit.text))
assert len(texts) == len(need)
train["query"] = train.qid.map(queries)
train["passage"] = train.pid.map(texts)
assert train["query"].notna().all() and train["passage"].notna().all()
label = np.select([train.rel == 3, train.rel == 2], [2, 1], 0)  # [N_train]
print(f"train pairs={len(train)} queries={train.qid.nunique()} grade counts={np.bincount(label).tolist()} "
      f"data prep {time.time() - t0:.0f}s")

torch.cuda.set_per_process_memory_fraction(cfg["gpu_frac"])
path = f"{cfg['models_dir']}/{cfg['repo']}"
tk = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3).cuda()
if cfg["grad_ckpt"]:
    model.gradient_checkpointing_enable()


def encode(q, p):
    return tk(q, [s[: cfg["passage_chars"]] for s in p], truncation="only_second", max_length=cfg["max_len"],
              padding=True, return_tensors="pt").to("cuda")


Q, P = train["query"].tolist(), train["passage"].tolist()
B = cfg["batch"]
steps = cfg["epochs"] * math.ceil(len(train) / B)
decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                         {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_frac"] * steps), steps)
scaler = torch.amp.GradScaler()
model.train()
t1 = time.time()
step, run_loss = 0, []
for ep in range(cfg["epochs"]):
    order = rng.permutation(len(train))
    for i in range(0, len(order), B):
        idx = order[i : i + B]
        enc = encode([Q[j] for j in idx], [P[j] for j in idx])
        tgt = torch.as_tensor(label[idx], device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits  # [B, 3]
        loss = torch.nn.functional.cross_entropy(logits.float(), tgt)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_loss.append(loss.item())
        if step % 250 == 0 or step == steps:
            print(f"step {step}/{steps} loss(last 250)={np.mean(run_loss[-250:]):.4f} "
                  f"elapsed {time.time() - t1:.0f}s peak alloc {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
train_s = time.time() - t1
peak_train = torch.cuda.max_memory_allocated() / 2**30
peak_reserved = torch.cuda.max_memory_reserved() / 2**30
print(f"training done in {train_s:.0f}s  peak allocated {peak_train:.2f} GiB  peak reserved {peak_reserved:.2f} GiB")
os.makedirs(cfg["ckpt_dir"], exist_ok=True)
model.save_pretrained(cfg["ckpt_dir"])
tk.save_pretrained(cfg["ckpt_dir"])
json.dump(cfg, open(f"{cfg['ckpt_dir']}/run_config.json", "w"))

model.eval()
DQ, DP = items["query"].tolist(), items["passage"].tolist()
prob = np.zeros((len(items), 3), dtype=np.float32)  # [n, 3]
t2 = time.time()
with torch.no_grad():
    for i in range(0, len(items), cfg["score_batch"]):
        enc = encode(DQ[i : i + cfg["score_batch"]], DP[i : i + cfg["score_batch"]])
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits  # [b, 3]
        prob[i : i + len(logits)] = torch.softmax(logits.float(), -1).cpu().numpy()
assert np.isfinite(prob).all()
out = f"{cfg['d1_dir']}/predictors"
os.makedirs(out, exist_ok=True)
np.save(f"{out}/ft_ce.npy", prob)
y = np.select([items.rel == 3, items.rel == 2], [2, 1], 0)
print(f"scored D1 n={len(items)} in {time.time() - t2:.0f}s; mean prob {prob.mean(0).round(3).tolist()}; "
      f"all-item accuracy {(prob.argmax(1) == y).mean():.3f}; peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(f"total {time.time() - t0:.0f}s")
print("DONE")
