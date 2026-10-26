"""Diagnostic (oracle): run-level judge-minus-human score per system, to reconcile the document-level provenance gap
(judges over-credit lexically retrieved documents, scripts/diagnostics/provenance_bias.py) with run-level reports that LLM
judges favour LLM-based systems (Balog et al. SIGIR 2025).

Per system s: true DCG@k, judge-only DCG@k (every item scored by the judges' mean gain, no qrels), and their
difference; then the judge-minus-human difference averaged by family, with a query-bootstrap 95% CI of
(bm25 family mean) - (neural family mean).
usage: python scripts/diagnostics/run_level.py <testbed.yaml> <eval.yaml> <rho>
"""
import os
import sys

import numpy as np
from witness.data import load_testbed  # noqa: E402
from witness.config import load_config  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
D = load_testbed(load_config(sys.argv[1]), load_config(sys.argv[2]))
rho = float(sys.argv[3])
u = (1 - rho) * D.u0 + rho * D.u1
ans = D.J >= 0
gain = np.where(D.J == 2, 1.0, np.where(D.J == 1, rho, 0.0))
f = np.where(ans.sum(0) > 0, (gain * ans).sum(0) / np.maximum(ans.sum(0), 1), D.ce)
lex = np.array([s.startswith("bm25") for s in D.systems])
per_q_true = np.stack([np.bincount(D.qi, weights=D.disc[s] * u, minlength=D.n_q) for s in range(len(D.systems))])
per_q_judge = np.stack([np.bincount(D.qi, weights=D.disc[s] * f, minlength=D.n_q) for s in range(len(D.systems))])
diff = per_q_judge - per_q_true  # [S, Q]
print(f"rho={rho} queries={D.n_q}")
for s, name in enumerate(D.systems):
    print(f"  {name:14s} true={per_q_true[s].mean():.4f} judge={per_q_judge[s].mean():.4f} judge-true={diff[s].mean():+.4f}")
gap = diff[lex].mean() - diff[~lex].mean()
rng = np.random.default_rng(0)
boots = []
for _ in range(2000):
    q = rng.integers(0, D.n_q, D.n_q)
    boots.append(diff[lex][:, q].mean() - diff[~lex][:, q].mean())
print(f"(judge-true) bm25 family mean - neural family mean = {gap:+.4f}  "
      f"query-bootstrap 95% [{np.quantile(boots, .025):+.4f}, {np.quantile(boots, .975):+.4f}]")
print("DONE")
