"""D2 testbed: TREC DL 2019 passage deep judgments, every judged passage is a candidate with a 0-3 label.

Keeps all 43 judged queries, pulls the judged passages from the MS MARCO corpus (streamed), and writes
  <out_dir>/items.parquet       one row per (query, candidate): qi, qid, pid, rel (0-3), query, passage  (D1 schema)
  <out_dir>/judge_items.jsonl   LLM-judge input, id "dl19:<qi>:<pid>"
usage: python scripts/data/build_d2.py configs/testbeds/d2_dl19.yaml
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
cfg = load_config(sys.argv[1])
D = cfg["data_dir"]
os.makedirs(cfg["out_dir"], exist_ok=True)

items = pd.read_csv(f"{D}/test.tsv", sep="\t", dtype={"query-id": np.int64, "corpus-id": np.int64, "score": np.int64})
items = items.rename(columns={"query-id": "qid", "corpus-id": "pid", "score": "rel"})
assert not items.duplicated(["qid", "pid"]).any()
assert items.rel.isin([0, 1, 2, 3]).all()
qsel = np.sort(items.qid.unique())
print(f"queries={len(qsel)} items={len(items)}")

need_q = {str(q) for q in qsel}
queries = {}
for line in open(f"{D}/queries.jsonl", encoding="utf8"):
    r = json.loads(line)
    if r["_id"] in need_q:
        queries[int(r["_id"])] = r["text"]
assert len(queries) == len(qsel)

need = {str(p) for p in items.pid}
texts = {}
head = '{"_id": "'
for line in open(f"{D}/corpus.jsonl", encoding="utf8"):
    # cheap id peek so that only the ~9k needed lines of the 8.8M-line corpus are json-parsed
    assert line.startswith(head)
    if line[len(head): line.index('"', len(head))] in need:
        r = json.loads(line)
        assert r["title"] == ""
        texts[int(r["_id"])] = r["text"]
print(f"passages found={len(texts)} of {len(need)}")
assert len(texts) == len(need)

items["query"] = items.qid.map(queries)
items["passage"] = items.pid.map(texts)
assert items["query"].notna().all() and items["passage"].notna().all()
qi = {q: i for i, q in enumerate(qsel)}
items["qi"] = items.qid.map(qi)
items = items.sort_values(["qi", "pid"]).reset_index(drop=True)
items[["qi", "qid", "pid", "rel", "query", "passage"]].to_parquet(f"{cfg['out_dir']}/items.parquet")
with open(f"{cfg['out_dir']}/judge_items.jsonl", "w", encoding="utf8") as fh:
    for r in items.itertuples():
        fh.write(json.dumps(dict(id=f"dl19:{r.qi}:{r.pid}", query=r.query, passage=r.passage, tag="dl19"), ensure_ascii=False) + "\n")
print(items.rel.value_counts().sort_index().to_dict())
print("candidates per query", items.groupby("qi").size().describe().round(1).to_dict())
print("passage chars", items.passage.str.len().describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict())
print("DONE")
