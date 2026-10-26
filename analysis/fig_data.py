"""Collect the numbers behind the paper figures into one JSON (computation side; plotting is analysis/make_figures.py).

Figure 1: certified share of decidable pairs vs human-label budget, D1 and D2, every calibrated method cross-fitted
          (eval_xm: every method cross-fitted, hence exactly design-unbiased), with the zero-label conventions'
          false-certification proportions as reference values.
Figure 2: provenance coupling on three independent human sources (D1 T2Ranking, D2 TREC DL19, D3 DuRetrieval):
          state-weighted human-relevance gap lex_only - neural_only with 95% CIs, at rho = 0, 1/2, 1, read from the
          diagnostic logs (numbers copied from logs are asserted against the log text, never typed in).
usage: PYTHONPATH=src python analysis/fig_data.py ${WITNESS_ROOT}/outputs results/figs/fig_data.json
(<outputs> holds d1/, d2/ with the merged evaluator CSVs and logs/ with the diagnostic logs)
"""
import json
import re
import sys

import pandas as pd

wd, out = sys.argv[1], sys.argv[2]
# figure key -> evaluator method (cross-fitted versions; classical has no fitted predictor)
METHODS = {"classical": "classical", "ppi_tuned": "ppi_tuned_xfit", "active": "active_xfit", "witness": "witness_xfit",
           "witness_fam": "witness_fam_xfit"}
fig1 = {}
for d in ("d1", "d2"):
    df = pd.read_csv(f"{wd}/{d}/eval_xm_raw.csv")
    g = df.groupby(["method", "budget"])["cert_dec_bh"]
    m, se = g.mean(), g.sem()
    fig1[d] = {k: {"budget": [int(b) for b in m[v].index], "cert": m[v].tolist(), "se": se[v].tolist()}
               for k, v in METHODS.items()}
    f = df.groupby("method")["fdp_bh"].mean()
    fig1[d]["zero_label_fdp"] = {k: float(f[k]) for k in ("official", "judge_plugin")}
    fig1[d]["zero_label_cert"] = {k: float(df[df.method == k]["cert_dec_bh"].mean()) for k in ("official", "judge_plugin")}


def grab(path, pattern):
    txt = open(path, encoding="utf8").read()
    hits = re.findall(pattern, txt)
    assert len(hits) == 1, (path, pattern, hits)
    return [float(x) for x in hits[0]]


num = r"([+-]?\d*\.\d+)"
fig2 = {}
for rho in ("0", "0.5", "1"):
    fig2.setdefault("D3 DuRetrieval", {})[rho] = grab(
        f"{wd}/logs/d3_findings_rho{rho}.log", rf"gap u\(lex_only\) - u\(neural_only\) = {num} \[{num}, {num}\]")
    for d, name in (("d1", "D1 T2Ranking"), ("d2", "D2 TREC DL19")):
        # scripts/diagnostics/provenance_bias.py reports the judge-error gap b = f - u; the judges' gain f is constant within a joint
        # state, so the human-relevance gap is its negative (the interval flips)
        b, lo, hi = grab(f"{wd}/logs/provbias_{d}_rho{rho}.log",
                         rf"gap b\(lex_only\) - b\(neural_only\) = {num}  query-bootstrap 95% \[{num}, {num}\]")
        fig2.setdefault(name, {})[rho] = [-b, -hi, -lo]
# Figure 4: false-certification proportion (mean over repetitions, = empirical FDR) of the two zero-label conventions and
# of WITNESS, across testbeds and simulated qrels mechanisms (truth = full human labels).
# WITNESS points are the cross-fitted method (witness_xfit, exactly design-unbiased, as in Figure 1).  The zero-label rows
# come from the in-sample runs; the cross-fitted twins redraw the same qrels (same seed and repetition streams), which is
# asserted row by row.  The grade-blind rows are the main-table runs (eval_xm).
# (label, testbed, in-sample tag, cross-fitted tag, WITNESS budget)
SCEN = [("D1 grade-blind", "d1", "v3c0", "xm", 3200), ("D1 grade-dep.", "d1", "v3c1", "xf_grade", 3200),
        ("D2 grade-blind", "d2", "v3c0", "xm", 200), ("D2 grade-dep.", "d2", "v3c1", "xf_grade", 200),
        ("D2 no pool bias", "d2", "sens_b0", "xf_poolbias_b0", 200),
        ("D2 weak pool bias", "d2", "sens_b2", "xf_poolbias_b2", 200),
        ("D2 strong pool bias", "d2", "sens_b6", "xf_poolbias_b6", 200),
        ("D2 ghost pool", "d2", "sens_ghost", "xf_poolbias_ghost", 200)]
METRICS = ["fdp_bh", "cert_dec_bh", "cover_pairs", "nwrong_bh"]


def assert_same_streams(old, new):
    for m in ("official", "judge_plugin"):
        a = old[(old.method == m) & (old.budget == 0)].sort_values("rep")
        b = new[(new.method == m) & (new.budget == 0)].sort_values("rep")
        assert a.rep.tolist() == b.rep.tolist() and (a[METRICS].values == b[METRICS].values).all(), m



fig4 = []
for label, d, tag, tag_xf, budget in SCEN:
    df = pd.read_csv(f"{wd}/{d}/eval_{tag}_raw.csv")
    df_xf = pd.read_csv(f"{wd}/{d}/eval_{tag_xf}_raw.csv")
    assert_same_streams(df, df_xf)
    row = dict(label=label, budget=budget, reps=int(df.rep.nunique()))
    for m, src, b in (("official", df, 0), ("judge_plugin", df, 0), ("witness", df_xf[df_xf.method == "witness_xfit"], budget)):
        s = src[(src.method == ("witness_xfit" if m == "witness" else m)) & (src.budget == b)]
        assert len(s) == row["reps"] and s.rep.nunique() == row["reps"], (label, m, len(s))
        row[m] = dict(fdp=float(s.fdp_bh.mean()), fdp_se=float(s.fdp_bh.sem()), cert=float(s.cert_dec_bh.mean()),
                      cover=float(s.cover_pairs.mean()))
    fig4.append(row)
# Figure 5: official leaderboard collapse vs pool-bias strength b (D2): simulated wrong certifications per repetition
# (mean +- se) against the analytic prediction (expected official contrast under the mechanism) parsed from logs/theory_flip_d2.log.
fig5 = dict(b=[], wrong=[], wrong_se=[], fdp=[], witness_fdp=[], pred_b=[], pred_wrong=[])
for t, b in (("b0", 0.0), ("b2", 2.0), ("b25", 2.5), ("b30", 3.0), ("b35", 3.5), ("b50", 5.0), ("b6", 6.0)):
    df = pd.read_csv(f"{wd}/d2/eval_sens_{t}_raw.csv")
    df_xf = pd.read_csv(f"{wd}/d2/eval_xf_poolbias_{t}_raw.csv")
    assert_same_streams(df, df_xf)
    s = df[df.method == "official"]
    w = df_xf[(df_xf.method == "witness_xfit") & (df_xf.budget == 200)]
    assert len(w) == len(s) and w.rep.nunique() == len(w)
    fig5["b"].append(b)
    fig5["wrong"].append(float(s.nwrong_bh.mean()))
    fig5["wrong_se"].append(float(s.nwrong_bh.sem()))
    fig5["fdp"].append(float(s.fdp_bh.mean()))
    fig5["witness_fdp"].append(float(w.fdp_bh.mean()))
    fig5.setdefault("witness_wrong", []).append(float(w.nwrong_bh.mean()))
    fig5.setdefault("witness_wrong_se", []).append(float(w.nwrong_bh.sem()))
txt5 = open(f"{wd}/logs/theory_flip_d2_refined.log").read()
for m in re.finditer(r"b=\s*([\d.]+):.*of which flipped (\d+)", txt5):
    fig5["pred_b"].append(float(m.group(1)))
    fig5["pred_wrong"].append(int(m.group(2)))
# refined prediction: query-CLT SE in the test statistic and the BH cut at the expected number of rejections
fig5["pred_refined"] = [float(x) for x in re.findall(r"expected wrongly certified ([\d.]+)", txt5)]
assert len(fig5["pred_b"]) >= 5 and len(fig5["pred_refined"]) == len(fig5["pred_b"])
json.dump(dict(fig1=fig1, fig2=fig2, fig4=fig4, fig5=fig5), open(out, "w"), indent=1)
print("fig5", {k: v for k, v in fig5.items()})
for r in fig4:
    print(r["label"], r["reps"], {m: round(r[m]["fdp"], 3) for m in ("official", "judge_plugin", "witness")})
print(json.dumps(dict(fig2=fig2), indent=1))
print("DONE")
