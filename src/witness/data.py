"""Load a fully labelled WITNESS testbed (D1 T2Ranking, D2 TREC DL19) into arrays shared by the evaluators.

Symbols (see witness.core): items d with query qi[d] and true grade y[d] in {0,1,2} (0-3 labels mapped 0,1->0;
2->1; 3->2); endpoint gains u0 = 1[y=2], u1 = 1[y>=1]; disc [S, n]; pair contrasts g [P, n]; G = [g; disc].
Judge labels J [n_judges, n] in {-1 (refused), 0, 1, 2}; cross-encoder probability ce [n].
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from witness import core as W


def load_testbed(base, cfg):
    out = cfg["out_dir"]
    items = pd.read_parquet(f"{out}/items.parquet")
    qi = items.qi.to_numpy()
    n_q = int(qi.max()) + 1
    assert np.array_equal(np.unique(qi), np.arange(n_q))
    rel = items.rel.to_numpy()
    y = np.select([rel == 3, rel == 2], [2, 1], 0)
    systems = list(base["dense"]) + ["bm25_" + t for t in base["lexical"]] + [cfg["reranker"]]
    missing = [s for s in systems if not os.path.exists(f"{out}/scores/{s}.npy")]
    assert not missing, missing
    scores = np.stack([np.load(f"{out}/scores/{s}.npy") for s in systems])  # [S, n]
    disc = W.discounts(qi, scores, cfg["k"])  # [S, n]
    pairs, g = W.pair_coefs(disc)  # [P, 2], [P, n]
    pool = [systems.index(s) for s in cfg["pool_systems"]]
    h = (W.discounts(qi, scores[pool], cfg["pool_k"]) > 0).mean(0)  # pool share of each item
    ids = [f"{cfg['id_prefix']}:{a}:{b}" for a, b in zip(items.qi, items.pid)]
    J = []
    for j in cfg["judges"]:
        lab = {r["id"]: (-1 if r.get("status") == "refused" else r["label"])
               for r in map(json.loads, open(f"{out}/judged_{j}.jsonl", encoding="utf8"))}
        assert all(i in lab for i in ids), j
        J.append([lab[i] for i in ids])
    J = np.array(J)  # [n_judges, n]
    ce = np.load(f"{out}/scores/{cfg['reranker']}.npy").astype(float)
    return SimpleNamespace(items=items, qi=qi, n_q=n_q, y=y, u0=(y == 2).astype(float), u1=(y >= 1).astype(float),
                           systems=systems, disc=disc, pairs=pairs, g=g, G=np.vstack([g, disc]), P=len(pairs),
                           h=h, in_union=(disc > 0).any(0), n_top=(disc > 0).sum(0), J=J, judges=list(cfg["judges"]),
                           ce=ce)
