"""D3 human-label batch: decision-aware Poisson sample on DuRetrieval (2,000 queries, 7 systems), blinded sheet + key.

Frame: every (query, doc) in some system's top-k. Two strata: L = qrels documents in the frame (qrels errors),
U = the rest (holes or negatives). Inclusion probabilities follow M3: pi_d proportional to
sqrt(sum_p w_p g_pd^2 v_d), with v_d from the two DeepSeek judges and pair weights w_p from the judge plug-in
estimates (pairs already decided by more than 3 SE get no weight). Each stratum is water-filled to its expected size.
The sheet is in random order, so any labelled prefix of N rows is itself a Poisson sample with pi_d * N / n.
Writes <out_dir>/frame.parquet, batch<stage>_sheet.xlsx, batch<stage>_key.json.
usage: python scripts/d3/make_batch.py configs/d3_duretrieval.yaml
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as ILLEGAL
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation
from witness import core as W  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
B = cfg["batch"]
PD, OUT, K = cfg["duretrieval_dir"], cfg["out_dir"], cfg["k"]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(B["seed"])


def one(pattern):
    import glob

    hits = glob.glob(pattern)
    assert len(hits) == 1, pattern
    return hits[0]


corpus = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/corpus-*.parquet"))
queries = pd.read_parquet(one(f"{PD}/hf/DuRetrieval/data/queries-*.parquet"))
qrels = pd.read_parquet(one(f"{PD}/hf/DuRetrieval-qrels/data/dev-*.parquet"))
pidx = {p: i for i, p in enumerate(corpus["id"])}
qidx = {q: i for i, q in enumerate(queries["id"])}
n_q = len(queries)
rel = {(qidx[q], pidx[p]) for q, p, s in zip(qrels["qid"], qrels["pid"], qrels["score"]) if s > 0}


def top_k(name, spec):
    if spec["kind"] == "dense":
        q = torch.from_numpy(np.load(f"{PD}/systems/{name}/q.npy")).float()  # [Q, dim]
        p = torch.from_numpy(np.load(f"{PD}/systems/{name}/p.npy")).float()  # [N, dim]
        return np.concatenate([torch.topk(q[i:i + 200] @ p.T, K, dim=1).indices.numpy() for i in range(0, len(q), 200)])
    if spec["kind"] == "bm25":
        return np.load(f"{PD}/systems/bm25/{spec['tok']}.npz")["idx"][:, :K]
    assert spec["kind"] == "rerank"
    return np.load(f"{PD}/systems/rerank/{spec['first']}.npz")["idx"][:, :K]


systems = list(cfg["systems"])
tops = {s: top_k(s, cfg["systems"][s]) for s in systems}
assert all(t.shape == (n_q, K) for t in tops.values())
pairs_qd = sorted({(q, int(d)) for t in tops.values() for q in range(n_q) for d in t[q]})
frame = pd.DataFrame(pairs_qd, columns=["qi", "di"])
pos = {qd: i for i, qd in enumerate(pairs_qd)}
disc = np.zeros((len(systems), len(frame)))
for s_i, s in enumerate(systems):
    for q in range(n_q):
        for r, d in enumerate(tops[s][q]):
            disc[s_i, pos[(q, int(d))]] = 1.0 / np.log2(r + 2)
frame["in_qrels"] = [qd in rel for qd in pairs_qd]
print(f"frame items={len(frame)} qrels in frame={frame.in_qrels.sum()} systems={systems}")

key_of = {}
for f in cfg["meta_files"]:
    for i, m in json.load(open(f"{PD}/judgments/{f}.json", encoding="utf8")).items():
        key_of.setdefault((m["qi"], m["di"]), []).append(i)
for j in cfg["judges"]:
    lab = {}
    for f in cfg["judge_files"][j]:
        for r in map(json.loads, open(f"{PD}/judgments/{f}.jsonl", encoding="utf8")):
            lab[r["id"]] = r["label"]
    vals = []
    for qd in pairs_qd:
        got = {lab[i] for i in key_of.get(qd, []) if i in lab}
        assert len(got) == 1, (j, qd, got)  # several item ids can point at one (query, doc); same prompt, one label
        vals.append(got.pop())
    frame[j] = vals
frame["query"] = [queries["text"].iloc[q] for q in frame.qi]
frame["passage"] = [corpus["text"].iloc[d] for d in frame.di]
np.savez_compressed(f"{OUT}/frame_disc.npz", disc=disc, systems=np.array(systems), n_q=n_q,
                    qi=frame.qi.to_numpy(), di=frame.di.to_numpy())  # discounts for the D3 analysis
if os.environ.get("FRAME_ONLY") == "1":  # discounts only; frame.parquet (with the drawn batch) is left untouched
    print("DONE (discounts only)")
    sys.exit(0)
frame.to_parquet(f"{OUT}/frame.parquet")

# --- allocation (M3): judge plug-in for pair weights, judge spread for v ---
rho = cfg["rho"]
qi = frame.qi.to_numpy()
Jg = np.stack([W.gain(frame[j].to_numpy(), rho) for j in cfg["judges"]])  # [n_judges, n]
f = Jg.mean(0)
in_l = frame.in_qrels.to_numpy()
f_plug = np.where(in_l, 1.0, f)  # the usual practice: qrels count as relevant, holes patched by the judges
pairs, g = W.pair_coefs(disc)
lo, hi = W.query_bootstrap_ci(g, f_plug, qi, n_q, rng, reps=500)
d_hat, se_hat = g @ f_plug / n_q, (hi - lo) / (2 * 1.96)
# Stage 1 has no design-based SE yet: the plug-in SE treats the judges as truth and would declare almost every pair
# decided, so no pair is dropped; the floor is the smallest DCG@10 difference this stage aims to resolve.
assert B["stage"] == 1
pair_w = 1.0 / np.maximum(np.abs(d_hat), B["delta_floor"]) ** 2
for (a, b), d, s, w in zip(pairs, d_hat, se_hat, pair_w):
    print(f"{systems[a]} - {systems[b]}: plug-in {d:+.4f} se {s:.4f} weight {w:.3g}")
assert (pair_w > 0).any()
# v: Bernoulli-type spread of the gain given the judges, plus a floor so no item has pi = 0
v_u = f * (1 - f) + 0.05
v_l = (1 - f) * f + 0.05  # in L the open question is whether the qrels label is wrong
wgt = W.allocation_weight(g, np.where(in_l, v_l, v_u), pair_w)
pi = np.zeros(len(frame))
for mask, n in ((~in_l, B["n_unlabelled"]), (in_l, B["n_qrels"])):
    pi[mask] = W.sampling_probs(n, wgt[mask])
xi = rng.random(len(frame)) < pi
frame["pi"], frame["sampled"] = pi, xi
print(f"sampled U={int((xi & ~in_l).sum())} L={int((xi & in_l).sum())}; pi range U [{pi[~in_l].min():.2e}, {pi[~in_l].max():.3f}]")
frame.to_parquet(f"{OUT}/frame.parquet")

# --- blinded sheet in random order, with exact repeats ---
rows = np.flatnonzero(xi)
rng.shuffle(rows)
n_rep = int(round(B["repeat_share"] * len(rows)))
order = list(rows)
for r in rng.choice(rows[: len(rows) // 2], size=n_rep, replace=False):  # repeats land in the second half
    order.insert(int(rng.integers(len(order) // 2, len(order) + 1)), r)
stage = B["stage"]
wb = Workbook()
ws0 = wb.active
ws0.title = "说明"
NOTE = [
    "任务：判断每个“段落”是否包含能直接回答“查询”所表达的信息需求的内容。",
    "0 = 不相关，或只是主题相关但不含答案。",
    "1 = 部分回答：含相关信息，但不完整、不够具体或需要推断。",
    "2 = 直接且完整地回答。",
    "只依据给定段落本身作答，不要使用搜索引擎或外部知识。可以用常识理解查询所指的对象（如缩写、歌名），但答案必须在段落里。",
    "在“标签”列选择 0、1 或 2。拿不准时选第一直觉，并在“备注”写一两个字（如“拿不准”“查询有歧义”“段落截断”）。",
    "少数条目会重复出现，用来检查一致性，请照常独立判断，不要回头翻之前的选择。",
    "行序是随机的：从第 1 行起连续标到任意一行都可用，不必标完；请不要跳行。",
    "多人分工时按行号切成连续的段，每人一段，并在“标注人”列写自己的代号。",
    "完成后保持 xlsx 格式保存，不要改动“序号”列。",
]
for line in NOTE:
    ws0.append([line])
ws0.column_dimensions["A"].width = 120
ws = wb.create_sheet("标注")
ws.append(["序号", "查询", "段落", "标签", "备注", "标注人"])
for c in "ABCDEF":
    ws[f"{c}1"].font = Font(bold=True)
key = []
for n, r in enumerate(order, start=1):
    # control characters cannot be stored in xlsx and are invisible anyway
    ws.append([n, ILLEGAL.sub("", frame["query"].iloc[r]), ILLEGAL.sub("", frame["passage"].iloc[r]), None, None, None])
    key.append(dict(row=n, qi=int(frame.qi.iloc[r]), di=int(frame.di.iloc[r]), stratum="L" if in_l[r] else "U",
                    pi=float(pi[r]), stage=stage, repeat=bool(order.index(r) != n - 1)))
dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True)
ws.add_data_validation(dv)
dv.add(f"D2:D{len(order) + 1}")
for c, w in zip("ABCDEF", (6, 28, 110, 8, 16, 10)):
    ws.column_dimensions[c].width = w
for row in ws.iter_rows(min_row=2):
    for cell in row[1:3]:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
wb.save(f"{OUT}/batch{stage}_sheet.xlsx")
json.dump(dict(stage=stage, systems=systems, pairs=pairs.tolist(), n_rows=len(order), n_unique=len(rows),
               rows=key), open(f"{OUT}/batch{stage}_key.json", "w"), ensure_ascii=False)
print(f"sheet rows={len(order)} unique={len(rows)} repeats={n_rep}")
print("DONE")
