"""Paper figures from results/figs/fig_data.json (plotting only; numbers come from analysis/fig_data.py).

Figure 1 (fig1_cert_budget): certified share of decidable pairs vs human-label budget, D1 (left) and D2 (right),
  mean over repetitions with +-1 se bars; Okabe-Ito colours with distinct markers and line styles (colour is never the
  only cue).  Zero-label conventions are not points on these axes (they use no labels and certify wrong conclusions);
  their false-certification proportions are stated in the caption.
Figure 2 (fig2_coupling): provenance coupling on three independent human sources, human-relevance gap
  lex_only - neural_only at the same joint judge verdict, point and 95% CI, at rho = 0, 1/2, 1.
Figure 3 (fig3_d3_leaderboard): D3 DuRetrieval pairwise conclusions, anchor-calibrated multiple-imputation human estimate
  with its BH-adjusted region vs official qrels and judge plug-in points (filled = certified).  Reads
  results/figs/d3_results.json (official, judge_plugin) and results/figs/d3_results_anchor.json (human estimate).
Geometry is provisional (textwidth 468.33 pt, STIXGeneral) until the target template is fixed.
usage: python analysis/make_figures.py results/figs/fig_data.json results/figs
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figkit"))
import paperstyle as ps  # noqa: E402
from figcheck import check_figure, check_pdf  # noqa: E402

data = json.load(open(sys.argv[1]))
ps.setup(textwidth_pt=468.33, font="STIXGeneral", fig_dir=sys.argv[2])
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STY = {  # Okabe-Ito on white (all >= 3:1 against white); markers and dashes carry identity redundantly
    "classical": dict(color="#000000", marker="s", ls=(0, (1, 1.2)), label="classical"),
    "ppi_tuned": dict(color="#CC79A7", marker="v", ls=(0, (4, 1.5)), label="PPI++"),
    "active": dict(color="#009E73", marker="^", ls=(0, (2, 1, 1, 1)), label="active inference"),
    "witness": dict(color="#0072B2", marker="o", ls="-", label="WITNESS"),
    "witness_fam": dict(color="#D55E00", marker="D", ls=(0, (5, 1)), label="WITNESS + provenance"),
}
# display names for the testbed codes carried by the data keys (the JSON keeps the internal codes)
TESTBED = {"D1": "T2Ranking", "D2": "TREC DL 2019", "D3": "DuRetrieval"}


def fig4_label(key):
    """'D2 ghost pool' -> 'TREC DL 2019, ghost pool'."""
    code, rest = key.split(" ", 1)
    return f"{TESTBED[code]}, {rest}"

# ---------------- Figure 1 ----------------
fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.2), sharey=True)
print("[fig1 verification]")
for ax, d, title in zip(axes, ("d1", "d2"), ("T2Ranking (weak judges)", "TREC DL 2019 (strong judges)")):
    for m, st in STY.items():
        r = data["fig1"][d][m]
        x, y, se = np.array(r["budget"]), np.array(r["cert"]), np.array(r["se"])
        ax.errorbar(x, y, yerr=se, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"],
                    capsize=1.5, elinewidth=0.6, markersize=3.2, lw=1.0)
        print(f"  {d} {m}: " + "  ".join(f"n={b} cert={c:.3f}±{s:.3f}" for b, c, s in zip(x, y, se)))
    ax.set_xscale("log")
    ps.plain_log_ticks(ax.xaxis, list(data["fig1"][d]["witness"]["budget"]))
    ax.set_xlabel("human labels")
    ax.set_title(title, fontsize=ps.FS_LABEL)
    ax.set_ylim(0, 0.7)
    zl = data["fig1"][d]
    print(f"  {d} zero-label: official cert={zl['zero_label_cert']['official']:.3f} fdp={zl['zero_label_fdp']['official']:.3f}; "
          f"judge_plugin cert={zl['zero_label_cert']['judge_plugin']:.3f} fdp={zl['zero_label_fdp']['judge_plugin']:.3f}")
    print(f"  {d} x-range: {ax.get_xlim()}  y-range: {ax.get_ylim()}")
axes[0].set_ylabel("certified share of decidable pairs")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="outside upper center", ncols=5)
check_figure(fig, "fig1_cert_budget")
p1 = ps.save(fig, "fig1_cert_budget")
plt.close(fig)
check_pdf(p1, ps.FULL)

# ---------------- Figure 2 ----------------
fig, ax = plt.subplots(figsize=(ps.HALF, 2.0))
srcs = ["D1 T2Ranking", "D2 TREC DL19", "D3 DuRetrieval"]  # data keys
src_names = ["T2Ranking", "TREC DL 2019", "DuRetrieval"]
rhos = [("0", "ρ = 0", "s", "#0072B2"), ("0.5", "ρ = 1/2", "o", "#D55E00"), ("1", "ρ = 1", "^", "#009E73")]
print("[fig2 verification]")
for i, s in enumerate(srcs):
    for j, (rk, rl, mk, col) in enumerate(rhos):
        pt, lo, hi = data["fig2"][s][rk]
        yy = i + (j - 1) * 0.22
        ax.errorbar(pt, yy, xerr=[[pt - lo], [hi - pt]], fmt=mk, color=col, ms=3.2, capsize=1.5, elinewidth=0.7,
                    label=rl if i == 0 else None)
        print(f"  {s} rho={rk}: gap={pt:+.4f} CI=[{lo:+.4f}, {hi:+.4f}] y={yy:.2f}")
ax.axvline(0, color="0.55", lw=0.6, ls="--")
ax.set_yticks(range(len(srcs)), src_names)
ax.invert_yaxis()
ax.set_xlabel("human relevance gap: BM25-only − neural-only")
# legend outside the axes: inside, it covers the long D3 intervals (figcheck does not test legend-vs-errorbar)
ps.legend_above(ax, ncols=3)
print(f"  x-range: {ax.get_xlim()}  reference line x=0")
check_figure(fig, "fig2_coupling")
p2 = ps.save(fig, "fig2_coupling")
plt.close(fig)
check_pdf(p2, ps.HALF)

# ---------------- Figure 3 ----------------
# D3 DuRetrieval: the 21 pairwise conclusions at rho = 1/2.  Human design-based estimate (+provenance calibration,
# two batches, labels brought to one annotation standard by multiple imputation) with its BH-adjusted region; official
# qrels and judge plug-in point estimates.  Filled marker = certified
# (endpoint IUT + BH), open = not certified.  Pairs sorted by the human estimate.
d3 = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "d3_results.json")))
# the final D3 result: anchor-calibrated multiple imputation (scripts/d3/anchor.py [4]); official and judge_plugin are
# standard-independent and stay from d3_results.json
d3["methods"].update(json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "d3_results_anchor.json")))["methods"])
short = {"e5_small": "e5-s", "bge_m3": "bge-m3", "qwen3_emb": "qwen3", "bm25_char1": "bm25-c1", "bm25_char2": "bm25-c2",
         "bm25_jieba": "bm25-jb", "rerank_bge_m3": "rerank"}
names = [f"{short[d3['systems'][a]]} − {short[d3['systems'][b]]}" for a, b in d3["pairs"]]
hum = d3["methods"]["design_fam_anchor_mi"]
est_h = np.array(hum["est"][1][:21])
cert_h = np.array(hum["cert_plus"]) | np.array(hum["cert_minus"])
# Interval consistent with the certification rule: certification needs both endpoints (rho = 0, 1) past the BH cut,
# so the displayed region is [min_e(est_e - z* se_e), max_e(est_e + z* se_e)] with z* the BH-adjusted critical value
# (p* = alpha k*/m, k* rejections among m = 2P one-sided nulls).  It excludes 0 iff the pair is certified (asserted).
from scipy.stats import norm  # noqa: E402

e0, e1 = np.array(hum["est"][0][:21]), np.array(hum["est"][2][:21])
s0, s1 = np.sqrt(np.array(hum["var"][0][:21])), np.sqrt(np.array(hum["var"][2][:21]))
k_star = int(np.sum(hum["cert_plus"]) + np.sum(hum["cert_minus"]))
z_star = norm.isf(d3["alpha"] * max(k_star, 1) / (2 * 21))
lo_h = np.minimum(e0 - z_star * s0, e1 - z_star * s1)
hi_h = np.maximum(e0 + z_star * s0, e1 + z_star * s1)
assert np.array_equal(lo_h > 0, np.array(hum["cert_plus"])) and np.array_equal(hi_h < 0, np.array(hum["cert_minus"]))
print(f"[fig3] BH: k*={k_star} of m=42, z*={z_star:.3f}; interval excludes 0 <=> certified (asserted)")
# two panels with their own x scales: cross-family pairs (dense/reranker vs BM25, |Delta| ~ .6-1) would otherwise
# squeeze the within-family pairs, where the conventions and the human evidence disagree, into a sliver
from matplotlib.lines import Line2D  # noqa: E402

lex = np.array([d3["systems"][a].startswith("bm25") != d3["systems"][b].startswith("bm25") for a, b in d3["pairs"]])
panels = [("cross-family pairs (neural vs BM25)", np.flatnonzero(lex)), ("within-family pairs", np.flatnonzero(~lex))]
fig, axes = plt.subplots(1, 2, figsize=(ps.FULL, 2.9))
print("[fig3 verification] n_labels", d3["n_labels"], "queries", d3["n_q"])
METH = (("design_fam_anchor_mi", "o", "#0072B2", 0.0), ("official", "s", "#000000", -0.27), ("judge_plugin", "^", "#D55E00", 0.27))
for ax, (title, idx) in zip(axes, panels):
    order = idx[np.argsort(est_h[idx])]
    yy = np.arange(len(order))
    ax.errorbar(est_h[order], yy, xerr=[est_h[order] - lo_h[order], hi_h[order] - est_h[order]], fmt="none",
                ecolor="#0072B2", elinewidth=0.9, capsize=1.5, zorder=2)
    for mname, mk, col, off in METH:
        m = d3["methods"][mname]
        e = np.array(m["est"][1][:21])
        c = np.array(m["cert_plus"]) | np.array(m["cert_minus"])
        for j, p in enumerate(order):
            ax.plot(e[p], yy[j] + off, marker=mk, ms=3.6, color=col, mfc=col if c[p] else "white", mew=0.8, ls="none",
                    zorder=3)
        print(f"  [{title}] {mname}: certified {int(c[order].sum())}/{len(order)}; " + "; ".join(
            f"{names[p]} {e[p]:+.3f}{'*' if c[p] else ''}" for p in order))
    for p in order:
        print(f"  [{title}] human {names[p]}: {est_h[p]:+.4f} region [{lo_h[p]:+.4f}, {hi_h[p]:+.4f}] "
              f"(rho=0 {e0[p]:+.4f}±{z_star * s0[p]:.4f}, rho=1 {e1[p]:+.4f}±{z_star * s1[p]:.4f}) "
              f"{'certified' if cert_h[p] else 'open'}")
    if title.startswith("within"):
        ax.axvline(0, color="0.55", lw=0.6, ls="--", zorder=1)
    ax.set_yticks(yy, [names[p] for p in order])
    ax.set_title(title, fontsize=ps.FS_LABEL)
    ax.set_xlabel("ΔDCG@10 (ρ = 1/2), first − second")
    print(f"  [{title}] x-range: {ax.get_xlim()}")
handles = [Line2D([], [], marker=mk, color=col, ls="none", ms=3.6) for _, mk, col, _ in METH]
handles += [Line2D([], [], marker="o", color="0.4", mfc="0.4", ls="none", ms=3.6),
            Line2D([], [], marker="o", color="0.4", mfc="white", ls="none", ms=3.6)]
fig.legend(handles, ["human labels (WITNESS)", "official qrels", "LLM-judge patched", "certified", "not certified"],
           loc="outside upper center", ncols=5)
check_figure(fig, "fig3_d3_leaderboard")
p3 = ps.save(fig, "fig3_d3_leaderboard")
plt.close(fig)
check_pdf(p3, ps.FULL)

# ---------------- Figure 4 ----------------
# False-certification proportion (empirical FDR over repetitions, +-1 se) of the two zero-label conventions and of
# WITNESS across testbeds and simulated qrels mechanisms; dashed line = nominal alpha.  Truth = full human labels.
rows = data["fig4"]
fig, ax = plt.subplots(figsize=(ps.HALF, 2.6))
M4 = (("official", "s", "#000000", "official qrels", -0.22), ("judge_plugin", "^", "#D55E00", "LLM-judge patched", 0.0),
      ("witness", "o", "#0072B2", "WITNESS", 0.22))
print("[fig4 verification] alpha line at 0.05")
for mname, mk, col, lab, off in M4:
    x = np.array([r[mname]["fdp"] for r in rows])
    se = np.array([r[mname]["fdp_se"] for r in rows])
    ax.errorbar(x, np.arange(len(rows)) + off, xerr=se, fmt=mk, color=col, ms=3.4, capsize=1.2, elinewidth=0.6,
                label=lab)
    for r in rows:
        print(f"  {r['label']} ({r['reps']} reps, WITNESS n={r['budget']}) {mname}: fdp={r[mname]['fdp']:.3f}"
              f"±{r[mname]['fdp_se']:.3f} cert={r[mname]['cert']:.3f} cover={r[mname]['cover']:.3f}")
ax.axvline(0.05, color="0.55", lw=0.6, ls="--")
ax.set_yticks(range(len(rows)), [fig4_label(r["label"]) for r in rows])
ax.invert_yaxis()
ax.set_xlim(-0.03, 1.03)
ax.set_xlabel("false-certification proportion")
# three entries do not fit one row at half width: two rows above the axes
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2)
print(f"  x-range: {ax.get_xlim()}")
check_figure(fig, "fig4_false_cert")
p4 = ps.save(fig, "fig4_false_cert")
plt.close(fig)
check_pdf(p4, ps.HALF)

# ---------------- Figure 5 ----------------
# Official leaderboard collapse vs pool-bias strength b (D2, BM25 pool): wrongly certified pairs per repetition (mean
# +- se over 20 repetitions) and the analytic prediction from the mechanism alone (expected official sign wrong and
# |E[Delta_off]| > 3 sd under Bernoulli labelling); WITNESS (n = 200) on the same runs.
f5 = data["fig5"]
fig, ax = plt.subplots(figsize=(ps.HALF, 2.2))
ax.errorbar(f5["b"], f5["wrong"], yerr=f5["wrong_se"], color="#000000", marker="s", ms=3.4, capsize=1.5, lw=1.0,
            label="official qrels (simulated)")
ax.plot(f5["pred_b"], f5["pred_wrong"], color="#CC79A7", ls=(0, (1, 1.2)), marker="x", ms=3.2, lw=0.9,
        label="prediction, labelling noise only")
ax.plot(f5["pred_b"], f5["pred_refined"], color="#009E73", ls=(0, (4, 1.5)), marker="+", ms=4.0, lw=1.0,
        label="prediction, query-CLT test + BH")
ax.errorbar(f5["b"], f5["witness_wrong"], yerr=f5["witness_wrong_se"], color="#0072B2", marker="o", ms=3.4,
            capsize=1.5, lw=1.0, label="WITNESS, n = 200")
ax.set_xlabel("pool-bias strength b")
ax.set_ylabel("wrongly certified pairs (of 21)")
ax.set_ylim(-0.5, 11.5)
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=2)
print("[fig5 verification]")
for b, w, s, fd, ww, ws, wf in zip(f5["b"], f5["wrong"], f5["wrong_se"], f5["fdp"], f5["witness_wrong"],
                                   f5["witness_wrong_se"], f5["witness_fdp"]):
    print(f"  b={b}: official wrong={w:.2f}±{s:.2f} fdp={fd:.3f}; WITNESS wrong={ww:.2f}±{ws:.2f} fdp={wf:.3f}")
print("  prediction (labelling noise only):", list(zip(f5["pred_b"], f5["pred_wrong"])))
print("  prediction (query-CLT + BH):", list(zip(f5["pred_b"], f5["pred_refined"])))
print(f"  x-range: {ax.get_xlim()}  y-range: {ax.get_ylim()}")
check_figure(fig, "fig5_collapse")
p5 = ps.save(fig, "fig5_collapse")
plt.close(fig)
check_pdf(p5, ps.HALF)
print("DONE")
