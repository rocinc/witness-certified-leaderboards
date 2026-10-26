"""Build the DuRetrieval (query, passage) items that the LLM judges label (scripts/data/judge.py).

Every item row is {"id": "<tag>:<qi>:<di>", "query", "passage", "tag"}; qi / di are query / corpus row indices.
Subcommands (each writes <name>.jsonl and <name>_meta.json under {duretrieval_dir}/judgments/):
  pool    items: the pre-registered sets (seed 0)
            C: query where >=2 of the 3 dense top-1 docs coincide and that doc is not in qrels -> one item (cap 400)
            D: query where the 3 dense top-1 docs are pairwise different and none is in qrels -> one item per doc
               (cap 400 // 3 queries)
            P: 300 random qrels positives.  N: 300 random corpus docs outside every system's top-100 for that query
            pool: for 300 random queries, the union of every system's top-10 minus qrels positives (with ranks)
          "every system" is the 3 dense, the 3 BM25 and every file in systems/rerank/.
  extend  items_ext (tag poolx): top-10 of the 7 additional dense systems on the 300 pool queries, outside qrels and
          outside the pool items.  Needs items_meta.json.
  full    items_full (tag poolf): top-10 of the 7 D3 systems on every query, outside qrels and outside pool / poolx.
          Needs items_meta.json and items_ext_meta.json.
  qrels   items_qrels (tag qrels): every qrels positive.
usage: PYTHONPATH=src python scripts/duretrieval/judge_items.py configs/d3_duretrieval.yaml {pool|extend|full|qrels}
"""
import json
import os
import sys

import numpy as np
from witness import duretrieval as DR  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

SEED = 0
CAP = 400
N_CONTROL = 300
N_POOL_QUERIES = 300
DENSE = ["e5_small", "bge_m3", "qwen3_emb"]
LEX = ["char1", "char2", "jieba"]
NEW = ["bge_small_zh", "bge_base_zh", "gte_small_zh", "gte_base_zh", "m3e_base", "e5_base", "text2vec_zh"]
FULL = ["e5_small", "bge_m3", "qwen3_emb", "bm25_char1", "bm25_char2", "bm25_jieba", "rerank_bge_m3"]


def write_items(path, rows):
    with open(path, "w", encoding="utf8") as fh:
        for it in rows:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_meta(path, meta):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(meta, fh, ensure_ascii=False)


def read_meta(path):
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def build_pool(root, out, ctext, qtext, rel):
    rng = np.random.default_rng(SEED)
    dense = {k: DR.dense_topk(root, k) for k in DENSE}
    lex = {k: DR.bm25_topk(root, k) for k in LEX}
    rerank = {}
    if os.path.isdir(f"{root}/systems/rerank"):
        for fn in sorted(os.listdir(f"{root}/systems/rerank")):
            # an unfinished rerank run would otherwise enter the pool as a system
            assert fn.endswith(".npz") and not fn.endswith(".partial.npz"), fn
            z = np.load(f"{root}/systems/rerank/{fn}")
            rerank["rerank_" + fn[:-4]] = (z["idx"], z["score"])
    every = {**dense, **{"bm25_" + k: v for k, v in lex.items()}, **rerank}
    print("pool rankers:", list(every), flush=True)
    nq = len(qtext)
    items, meta = [], {}

    def add(tag, qi, di, **extra):
        iid = f"{tag}:{qi}:{di}"
        items.append(dict(id=iid, query=qtext[qi], passage=ctext[di], tag=tag))
        meta[iid] = dict(qi=int(qi), di=int(di), tag=tag, **extra)

    c_items, d_items = [], []
    for qi in range(nq):
        tops = [int(dense[k][0][qi, 0]) for k in DENSE]
        miss = [t not in rel[qi] for t in tops]
        if not all(miss):
            continue
        vals, counts = np.unique(tops, return_counts=True)
        if counts.max() >= 2:
            di = int(vals[counts.argmax()])
            n_lex = sum(int(lex[k][0][qi, 0]) == di for k in LEX)
            c_items.append((qi, di, int(counts.max()), n_lex))
        elif len(vals) == 3:
            d_items.append((qi, tops))
    print("C candidates", len(c_items), "D candidate queries", len(d_items), flush=True)

    for qi, di, n_dense, n_lex in [c_items[i] for i in rng.permutation(len(c_items))[:CAP]]:
        add("C", qi, di, n_dense=n_dense, n_lex=n_lex)
    for qi, tops in [d_items[i] for i in rng.permutation(len(d_items))[: CAP // 3]]:
        for di in tops:
            add("D", qi, int(di))

    pos = [(qi, di) for qi in range(nq) for di in rel[qi]]
    for i in rng.permutation(len(pos))[:N_CONTROL]:
        add("P", *pos[i])

    banned = [set(int(x) for ranker in every.values() for x in ranker[0][qi]) for qi in range(nq)]
    for qi in rng.integers(0, nq, N_CONTROL):
        while True:
            di = int(rng.integers(0, len(ctext)))
            if di not in banned[qi] and di not in rel[qi]:
                break
        add("N", int(qi), di)

    for qi in rng.permutation(nq)[:N_POOL_QUERIES]:
        seen = {}
        for name, (idx, _) in every.items():
            for r, di in enumerate(idx[qi, :10]):
                di = int(di)
                if di not in rel[qi]:
                    seen.setdefault(di, {})[name] = r + 1
        for di, ranks in seen.items():
            add("pool", int(qi), di, ranks=ranks)

    write_items(f"{out}/items.jsonl", items)
    write_meta(f"{out}/items_meta.json", meta)
    tags = {}
    for it in items:
        tags[it["tag"]] = tags.get(it["tag"], 0) + 1
    print("items", len(items), tags, flush=True)


def new_ranks(names, tops, queries, rel, judged):
    """{(q, d): {system: rank}} over the top-10 of `names` on `queries`, skipping qrels positives and judged pairs."""
    ranks = {}
    for name in names:
        for q in queries:
            for r, d in enumerate(tops[name][q, :10]):
                d = int(d)
                if d in rel[q] or (q, d) in judged:
                    continue
                ranks.setdefault((q, d), {})[name] = r + 1
    return ranks


def write_ranked(out, name, tag, ranks, ctext, qtext):
    write_items(f"{out}/{name}.jsonl",
                [dict(id=f"{tag}:{q}:{d}", query=qtext[q], passage=ctext[d], tag=tag) for (q, d) in sorted(ranks)])
    write_meta(f"{out}/{name}_meta.json", {f"{tag}:{q}:{d}": dict(qi=q, di=d, tag=tag, ranks=r) for (q, d), r in ranks.items()})


def build_extend(root, out, ctext, qtext, rel):
    meta = read_meta(f"{out}/items_meta.json")
    judged = {(m["qi"], m["di"]) for m in meta.values() if m["tag"] == "pool"}
    pool_q = sorted({m["qi"] for m in meta.values() if m["tag"] == "pool"})
    ranks = new_ranks(NEW, {k: DR.dense_topk(root, k)[0] for k in NEW}, pool_q, rel, judged)
    print(f"new unjudged pairs: {len(ranks)} over {len(pool_q)} queries (already judged pool: {len(judged)})", flush=True)
    write_ranked(out, "items_ext", "poolx", ranks, ctext, qtext)


def build_full(root, out, ctext, qtext, rel):
    tops = {k: DR.dense_topk(root, k)[0] for k in FULL[:3]}
    tops.update({"bm25_" + t: DR.bm25_topk(root, t)[0] for t in ("char1", "char2", "jieba")})
    tops["rerank_bge_m3"] = np.load(f"{root}/systems/rerank/bge_m3.npz")["idx"]
    judged = set()
    for name in ("items_meta", "items_ext_meta"):
        judged |= {(m["qi"], m["di"]) for m in read_meta(f"{out}/{name}.json").values() if m["tag"] in ("pool", "poolx")}
    ranks = new_ranks(FULL, tops, range(len(qtext)), rel, judged)
    print(f"new unjudged pairs: {len(ranks)} over {len(qtext)} queries", flush=True)
    write_ranked(out, "items_full", "poolf", ranks, ctext, qtext)


def build_qrels(out, ctext, qtext, rel):
    items, meta = [], {}
    for q, docs in enumerate(rel):
        for d in sorted(docs):
            iid = f"qrels:{q}:{d}"
            items.append(dict(id=iid, query=qtext[q], passage=ctext[d], tag="qrels"))
            meta[iid] = dict(qi=q, di=d, tag="qrels")
    write_items(f"{out}/items_qrels.jsonl", items)
    write_meta(f"{out}/items_qrels_meta.json", meta)
    print("qrels positives:", len(meta))


cfg = load_config(sys.argv[1])
cmd = sys.argv[2]
assert cmd in ("pool", "extend", "full", "qrels"), cmd
root = cfg["duretrieval_dir"]
out = f"{root}/judgments"
os.makedirs(out, exist_ok=True)
corpus, queries, _ = DR.load_retrieval(root)
ctext, qtext = corpus["text"].tolist(), queries["text"].tolist()
_, _, rel = DR.load_qrels(root)
if cmd == "pool":
    build_pool(root, out, ctext, qtext, rel)
elif cmd == "extend":
    build_extend(root, out, ctext, qtext, rel)
elif cmd == "full":
    build_full(root, out, ctext, qtext, rel)
else:
    build_qrels(out, ctext, qtext, rel)
