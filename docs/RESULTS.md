# Results → commands

Conventions: run from the repo root with `export PYTHONPATH=src WITNESS_ROOT=<root>`; `$O` = `$WITNESS_ROOT/outputs`; every command's stdout goes to `$O/logs/<name>.log` (the log name is in the output column); evaluator CSVs land in the testbed's `out_dir` (`$O/d1`, `$O/d2`, `$O/d3`). `TB1=configs/testbeds/d1_t2ranking.yaml`, `TB2=configs/testbeds/d2_dl19.yaml`, `D3=configs/d3_duretrieval.yaml`.

Released copies: every evaluator CSV and every log named below is in the repository under `results/runs/` with the same relative path (`results/runs/d1/`, `results/runs/d2/`, `results/runs/logs/`), and the D3 result JSON files are `results/figs/d3_results{,_anchor}.json`; so each number can be checked against its file without running anything. The inputs the analyses read (testbed items without text, system scores, judge grades, the D3 frame and design files, and the human labels as CSV) are the data-artifact bundle of the release, and `WITNESS_ROOT=<bundle>` runs every command below that does not re-encode or re-judge. Where a command takes `<batch1.xlsx>`, `<batch2.xlsx>` or `<anchor.xlsx>`, the released `outputs/d3/labels/{batch1,batch2,anchor}.csv` give the same result. Logs of six diagnostics (`model_based_d{1,2}`, `run_level_d{1,2}_rho{0,1}`) were printed to the terminal and are not part of `results/runs/`; their commands are cheap. Seven D3 diagnostics read the dense embeddings or the corpus of the upstream DuRetrieval pipeline as well (`drift.py`, `pu_real_qrels.py`, `judge_ladder.py`, `content_columns.py`, `ladder_selection.py`, `em_exclusion.py`, `em_trajectory.py`), and `word_vectors.py` loads a word-vector model; these need the `D3 retrieval` step of the inputs table first, and their released logs are the record otherwise.

Chunked runs: `evaluate.py <testbed> <eval> lo hi` writes `eval_<tag>_r<lo>-<hi>_raw.csv`, then `python scripts/eval/merge.py $O/d1 <tag> <reps>` writes `eval_<tag>_raw.csv` and `eval_<tag>.csv`. Loops used: D1 `for lo in 0 5 10 15; do ... $lo $((lo+5)); done` (20 reps) or `for lo in 0 5` (10 reps); D2 `for lo in 0 20; do ... $lo $((lo+20)); done` (40 reps).

## Inputs (built once)

| step | command(s) | output file(s) |
|---|---|---|
| D1 testbed, 13 systems | `python scripts/data/build_d1.py $TB1`; `bash scripts/data/run_d1_systems.sh` | `$O/d1/items.parquet`, `$O/d1/scores/*.npy`; logs `d1_build`, `d1_<system>` |
| D2 testbed | `python scripts/data/build_d2.py $TB2`; `python scripts/data/score_systems.py $TB2 <system>` per system | `$O/d2/items.parquet`, `$O/d2/scores/*.npy` |
| LLM judges (D1; D2 adds `--en`) | `python scripts/data/judge.py $O/d1/judge_items.jsonl $O/d1/judged_flash.jsonl deepseek-v4-flash <duretrieval_judge_cache.jsonl>`; same with `deepseek-v4-pro` → `judged_pro.jsonl` | `$O/d{1,2}/judged_{flash,pro}.jsonl`; logs `d{1,2}_judge_{flash,pro}` |
| D3 retrieval + judges | `scripts/duretrieval/{download,bm25,encode,rerank,judge_items}.py $D3 ...` (args in each docstring) | `duretrieval_dir` of `$D3` |
| D3 label batches (D3 human labels required) | `python scripts/d3/make_batch.py $D3`; `python scripts/d3/make_next_batch.py $D3 2 400 <batch1.xlsx>`; `python scripts/d3/make_anchor.py $D3 60 <batch1.xlsx> <batch2.xlsx>` | `$O/d3/batch{1,2}_sheet.xlsx`, `batch{1,2}_key.json`, `anchor_key.json`; logs `d3_batch1`, `d3_batch2` (planner log) |

## Phenomenon

Zero-label conventions certify wrong conclusions; official qrels collapse in a phase transition with pool bias.

| result | command(s) | output file(s) |
|---|---|---|
| Official false-certification rate D1 .695, D2 .981 (`official`, fdp_bh) | D1: chunk loop `python scripts/eval/evaluate.py $TB1 configs/eval/insample_d1.yaml $lo $((lo+5))`; `python scripts/eval/merge.py $O/d1 v3c0 20`. D2: loop `... $TB2 configs/eval/insample_d2.yaml $lo $((lo+20))`; `merge.py $O/d2 v3c0 40` | `$O/d{1,2}/eval_v3c0_raw.csv`; logs `v3_d{1,2}_c0_r*` |
| Grade-dependent qrels variant (Fig. 4) | same loops with `configs/eval/insample_grade_d{1,2}.yaml`; `merge.py ... v3c1 20/40` | `$O/d{1,2}/eval_v3c1_raw.csv` |
| Judge plug-in false certification D1 .428 (judges exact agreement .50–.54); D2 0 | as above (`judge_plugin` rows); agreement in the evaluator log header | `eval_v3c0_raw.csv`; log `v3_d1_c0_r0` (`judges ... exact agreement`) |
| D2 judges binary agreement .77–.80 | `python scripts/diagnostics/d2_judge_stats.py $TB2 flash:$O/d2/judged_flash.jsonl:$O/logs/d2_judge_flash.log pro:$O/d2/judged_pro.jsonl:$O/logs/d2_judge_pro.log` | log `d2_stats` |
| Pool-bias sweep b=0/2/2.5/3/3.5/5/6: official fdp .031/.050/.194/.500/.900/.988/.996 | `for b in b0 b2 b25 b30 b35 b50 b6 ghost; do python scripts/eval/evaluate.py $TB2 configs/eval/poolbias_d2_$b.yaml; done` | `$O/d2/eval_sens_{b0,b2,b25,b30,b35,b50,b6,ghost}_raw.csv`; logs `sens_d2_*` |
| Cross-fitted WITNESS (`witness_xfit`) on the same qrels draws as the Fig. 4/5 settings: fdp 0 everywhere except D2 grade-dependent .003 (one repetition of 40 at .100) | D1 grade-dependent: `for lo in 0 10; do python scripts/eval/evaluate.py $TB1 configs/eval/xfit_grade_d1.yaml $lo $((lo+10)); done`; `merge.py $O/d1 xf_grade 20`. D2 grade-dependent: `for lo in 0 20; do ... $TB2 configs/eval/xfit_grade_d2.yaml $lo $((lo+20)); done`; `merge.py $O/d2 xf_grade 40`. Pool-bias sweep: `for b in b0 b2 b25 b30 b35 b50 b6 ghost; do python scripts/eval/evaluate.py $TB2 configs/eval/xfit_poolbias_d2_$b.yaml; done`. The grade-blind rows are `eval_xm_raw.csv` (Fig. 1) | `$O/d{1,2}/eval_xf_grade_raw.csv`, `$O/d2/eval_xf_poolbias_{b0,b2,b25,b30,b35,b50,b6,ghost}_raw.csv`; logs `xf_grade_d{1,2}_r*`, `xf_poolbias_d2_*`, `merge_xf_grade_d{1,2}` |

## Real data D3

Human-certified DuRetrieval leaderboard (843 labels in two batches + 120 anchor re-labels, 21 pairs).

| result | command(s) | output file(s) |
|---|---|---|
| Official and judge plug-in each claim 19/21; human design supports 13–14/21 (design / design_fam), 0 reversed (D3 human labels required) | `python scripts/d3/analyze.py $D3 $O/logs/d3_batch2.log <batch1.xlsx> <batch2.xlsx>` | log `d3_analyze_multi`; `$O/d3/d3_results.json` |
| Qrels entries judged irrelevant 20.9% [6.9, 35.7]; unlabelled frame items relevant 21.6% [16.7, 26.8] (~11,000 holes vs 8,989 qrels); per-system missed hole mass .47–.93 DCG@10; judges binary agreement .88–.89 (D3 human labels required) | `for r in 0 0.5 1; do python scripts/d3/findings.py $D3 <batch1.xlsx> $r; done` | logs `d3_findings_rho{0,0.5,1}` |
| Annotator drift: batch-2 gain lower .060 [.033, .089], cell Wald Σz² = 7.32 (df 4, p .12), 1/2 boundary shift (D3 human labels required) | `python scripts/d3/drift.py $D3 <batch1.xlsx> <batch2.xlsx>` | log `d3_drift` |
| Anchor re-labelling: batch 1 too lenient −.072 [−.125, −.030]; agreement exact .86/.93, binary .95/.95; anchor-calibrated MI (M=200) certifies 12/21, between-imputation share of the variance median .38; regression calibration (section [3]) 17/21 (D3 human labels required) | `python scripts/d3/anchor.py $D3 <anchor.xlsx> <batch1.xlsx> <batch2.xlsx>` | log `d3_anchor`; `$O/d3/d3_results_anchor.json` (entry `design_fam_anchor_mi`: MI estimate, variance and BH certification of section [4], same schema as `d3_results.json`) |

## Coupling finding

| result | command(s) | output file(s) |
|---|---|---|
| Provenance gap (BM25-only − neural-only human relevance, ρ=½) D1 −.042 [−.054, −.029], D2 −.097 [−.191, −.024] | `for r in 0 0.5 1; do python scripts/diagnostics/provenance_bias.py $TB1 configs/eval/insample_d1.yaml $r; done`; same with `$TB2 configs/eval/insample_d2.yaml` | logs `provbias_d{1,2}_rho{0,0.5,1}` |
| Provenance gap D3 −.119 [−.204, −.029] (D3 human labels required) | the `findings.py` loop above | logs `d3_findings_rho*` |
| Fine-tuned CE (T2Ranking train, never saw the systems): same-decile gap −.037 [−.058, −.017] | `python scripts/data/finetune_ce.py configs/data/finetune_ce_d1.yaml`; `python scripts/diagnostics/finetuned_ce.py $TB1 configs/eval/insample_d1.yaml $O/d1/predictors/ft_ce.npy` | `$O/d1/predictors/ft_ce.npy`; logs `ft_ce`, `diag_ftce` |
| Run level misses it: D1 +.002 / +.012 (CI ∋ 0), D2 ρ=0 −.453 | `for r in 0 1; do python scripts/diagnostics/run_level.py $TB1 configs/eval/insample_d1.yaml $r; done`; same for `$TB2 configs/eval/insample_d2.yaml` | stdout (log `run_level_d{1,2}_rho{0,1}`) |

## Theory checks

| result | command(s) | output file(s) |
|---|---|---|
| Phase-transition prediction (expected official sign flips under the qrels propensity, query-CLT test + BH): expected wrong certifications .07/.39/2.47/7.22/10.05 vs realised .10/1.65/3.65/7.60/10.25 (b=2/2.5/3/3.5/5) | `python scripts/theory/official_flips.py $TB2 configs/eval/poolbias_d2_b0.yaml 0,1,2,2.5,3,3.5,4,5,6`; realised = `official` rows of the pool-bias sweep | log `theory_flip_d2_refined`; `$O/d2/eval_sens_*_raw.csv` |
| Mechanism-general prediction, D1 within 1.3 se: dense 8.64 vs 8.40±.34, BM25 28.9 vs 29.1±.59, query depth 20.5 vs 21.1±.48 (logistic 41.7); D2 dense .63 vs .85±.12, BM25 .25 vs .33±.14, query depth .18 vs 1.57±.43 | `python scripts/theory/official_flips_general.py <TB> <EVAL>` for (TB, EVAL, log) = (`$TB1`, `main_d1`, `mech_theory_xm_d1`), (`d1_without_bge_base`, `mechanism_hidden_dense_d1`, `mech_theory_mhd_d1`), (`d1_without_bm25_char2`, `mechanism_hidden_bm25_d1`, `mech_theory_mhb_d1`), (`d1_without_bge_base`, `mechanism_query_depth_d1`, `mech_theory_mqd_d1`) and the D2 analogues (`$TB2`/`main_d2`, `d2_without_bge_m3`/`mechanism_hidden_dense_d2`, `d2_without_bm25_stem`/`mechanism_hidden_bm25_d2`, `d2_without_bge_m3`/`mechanism_query_depth_d2`); realised = `official` rows of the mechanism-family runs (Method) | logs `mech_theory_{xm,mhd,mhb,mqd}_d{1,2}` |
| Monte Carlo of the mechanism (D2 4,000 draws, D1 2,000): D2 query depth 1.04±.04 (s_q fixed .37); D2 dense .69, D2 BM25 .64; D1 cells within .03 of analytic | `python scripts/theory/official_flips_mc.py <TB> <EVAL> 4000 0` for the four D2 pairs; `... <TB> <EVAL> 2000 0` for the three D1 mechanism pairs (no D1 logistic run) | logs `mech_mc_{mhd,mhb,mqd}_d{1,2}`, `mech_mc_xm_d2` |
| Label complexity: E[K] MAE 0.29 pairs over 12 (method, budget) cells (max .73), Brier .048; old deterministic count MAE 4.6, max 11.9; variance ratio .99–1.06 | `python scripts/theory/label_complexity.py sim $TB1 configs/eval/main_d1.yaml 0 20 $O/d1/complexity/sim.pkl`; `... sim $TB2 configs/eval/main_d2.yaml 0 40 $O/d2/complexity/sim.pkl`; `... report 0.05 d1 $O/d1/complexity/sim.pkl $O/d1/eval_xm_raw.csv d2 $O/d2/complexity/sim.pkl $O/d2/eval_xm_raw.csv` | `$O/d{1,2}/complexity/sim.pkl`; logs `cc_d1`, `cc_d2`, `cc_report` |
| Power-corrected optimal design (t = z* + z_β): D2 n=600 E[K] 3.7 → 11.5 vs WITNESS 10.24; oracle optimum D1 n=12,800 51.7 vs 44.6 (86%) | `python scripts/theory/label_complexity.py opt $TB1 configs/eval/main_d1.yaml $O/d1/complexity/opt.pkl` (and D2); `... power $TB1 configs/eval/main_d1.yaml` (and D2) | logs `cc_opt_d{1,2}`, `cc_power_d{1,2}` |
| Ex-ante D3 planner: 14.3, BH self-consistent 15.3, realised 14 (D3 human labels required) | `python scripts/theory/label_complexity.py d3 $D3 <batch1.xlsx> <batch2.xlsx>` | log `cc_d3` |
| Labels to close the open D3 pairs (z\* + z_β = 3.03): 7 of 9 within-family pairs share a sign at both endpoints, joint budget 6,259 expected labels (3,065 for the easiest three, 5,263 for six), 843 bought; single-pair 1/n extrapolation 2,605 to 21,038; 2 pairs are ties (D3 human labels required) | `python scripts/theory/d3_remaining_budget.py $D3 <batch1.xlsx> <batch2.xlsx>` | log `d3_remaining_budget` |
| E-process validity: mixture process P(sup E ≥ 20) up to .465 (withdrawn); betting e-process ≤ level | `python scripts/theory/eprocess_validity.py` | log `test_eprocess` |
| Model-based vs design-based: coverage .31 with judge cells only, .967 with per-system columns (D1); D2 .944 | `python scripts/diagnostics/model_based.py $TB1 configs/eval/insample_d1.yaml 0.5 3200 10`; `python scripts/diagnostics/model_based.py $TB2 configs/eval/insample_d2.yaml 0.5 200 30` | stdout (log `model_based_d{1,2}`) |
| Layer L not sampled (bias = qrels errors + missing grades): coverage .38 (D1) / .53 (D2), false certification .15–.17 (D1) | `abl_L_not_sampled` rows of the in-sample runs (Phenomenon) and of the pool-bias sweep | `$O/d{1,2}/eval_v3c0_raw.csv`, `$O/d2/eval_sens_*_raw.csv` |

## Method

D1 loop = 4 chunks of 5 (20 reps) unless noted; D2 loop = 2 chunks of 20 (40 reps).

| result | command(s) | output file(s) |
|---|---|---|
| Main cross-fitted table: D1 n=3,200 WITNESS .187 / provenance .208 vs active .072 / PPI++ .119; n=12,800 .626 / .646 vs .549 / .520; D2 n=60 .096 vs .034/.020, n=200 .505 vs .409/.307; fdp ≤ .017, coverage .92–.97 (Fig. 1) | `python scripts/eval/evaluate.py $TB1 configs/eval/main_d1.yaml $lo $((lo+5))` (D1 loop); `merge.py $O/d1 xm 20`; `... $TB2 configs/eval/main_d2.yaml $lo $((lo+20))` (D2 loop); `merge.py $O/d2 xm 40` | `$O/d{1,2}/eval_xm_raw.csv` (md5 8b019645 / 2a0193b0); logs `xm_d{1,2}_r*`, `merge_xm_d{1,2}` |
| Module ablations (D1 n=12,800 / D2 n=200): uniform allocation .554 / .325; judge predictor .570 / .243; no strata / no extras ≈0 (.633 / .520) | same runs (`abl_*_xfit` methods of `main_d{1,2}.yaml`) | `$O/d{1,2}/eval_xm_raw.csv` |
| Cross-fitting cost vs in-sample ≤ 1–3 se | D1 `for lo in 0 5`: `evaluate.py $TB1 configs/eval/crossfit_check_d1.yaml $lo $((lo+5))`; `merge.py $O/d1 xfit 10`; D2 loop with `crossfit_check_d2.yaml`; `merge.py $O/d2 xfit 40` | `$O/d{1,2}/eval_xfit_raw.csv` |
| PU on real qrels ties judge predictor 13/21 vs 13/21 (SE ratio 1.01–1.03); PU underestimates holes −.054 [−.080, −.031]; within-stratum qrels logit +.65/SD (judge), +.72/SD (encoder); adaptive refit 13/21, SE ratio .99–1.01, calibration −.044 vs −.021 (D3 human labels required) | `python scripts/d3/pu_real_qrels.py $D3 <batch1.xlsx> <batch2.xlsx>` (sections [1]–[4] = `d3_pu`, [5] = `d3_pu_adapt2`, [6]–[7] = `d3_pu_gap`) | logs `d3_pu`, `d3_pu_adapt2`, `d3_pu_gap` |
| Judge-strength ladder (7 rungs, AUC .874 → .605): Spearman +.93 (perm. p .007); weakest-rung SE ratio .956 [.928, .992]; selection-free prior .885 (D3 human labels required) | `python scripts/d3/judge_ladder.py $D3 <batch1.xlsx> <batch2.xlsx>` | log `d3_judge_ladder` |
| Strong judges absorb content features: cosine column / selection-free PU prior / cosine + provenance 13–14/21, SE ratio .995–1.018 vs judge + provenance; ladder non-monotonicity is one pair (bm25_char1 vs char2, p ≈ .046) (D3 human labels required) | `python scripts/d3/content_columns.py $D3 <batch1.xlsx> <batch2.xlsx>` | log `d3_content_cols` |
| Why the PU selection factor hurts on real qrels: source-only qrels inclusion in the n_top 5–7 stratum .80 vs observed .68; posterior .54 vs human .73 (D3 human labels required) | `python scripts/d3/ladder_selection.py $D3 <batch1.xlsx> <batch2.xlsx>` | log `d3_ladder_ntop` |
| Content-shift mechanism D2: WITNESS n=60/200 .142/.478, judge predictor .045/.311, active .054/.439, PPI++ .032/.305, fdp ≤ .008; D1 n=12,800: WITNESS .592, provenance .616, judge .576, uniform .593, active .522, PPI++ .513 | D1 loop `evaluate.py $TB1 configs/eval/content_shift_d1.yaml ...`; `merge.py $O/d1 xmsh 20`; D2 loop with `content_shift_d2.yaml`; `merge.py $O/d2 xmsh 40` | `$O/d{1,2}/eval_xmsh_raw.csv` (D1 md5 d6147b2c); logs `xmsh_d{1,2}_r*` |
| Mechanism family (hidden dense / hidden BM25 / query depth): design-based fdp ≤ .004, coverage .90–.99, WITNESS best or tied in every cell | D1 `for lo in 0 5`: `evaluate.py configs/testbeds/d1_without_bge_base.yaml configs/eval/mechanism_hidden_dense_d1.yaml $lo $((lo+5))`, `d1_without_bm25_char2.yaml` + `mechanism_hidden_bm25_d1.yaml`, `d1_without_bge_base.yaml` + `mechanism_query_depth_d1.yaml`; `merge.py $O/d1 {mhd,mhb,mqd} 10`. D2 loop: `d2_without_bge_m3.yaml` + `mechanism_hidden_dense_d2.yaml`, `d2_without_bm25_stem.yaml` + `mechanism_hidden_bm25_d2.yaml`, `d2_without_bge_m3.yaml` + `mechanism_query_depth_d2.yaml`; `merge.py $O/d2 {mhd,mhb,mqd} 40` | `$O/d{1,2}/eval_{mhd,mhb,mqd}_raw.csv`; logs `mech_m*_d*_r*`, `merge_m*_d*` |
| Rivals: Oosterhuis (KDD'24) valid, D1 n=12,800 .266; LARA (SIGIR'25) D1 false certification .12–.16 | D1 `for lo in 0 5`: `evaluate.py $TB1 configs/eval/rivals_d1.yaml $lo $((lo+5))`; `merge.py $O/d1 rivals 10`; D2 loop with `rivals_d2.yaml`; `merge.py $O/d2 rivals 40` | `$O/d{1,2}/eval_rivals_raw.csv`; logs `rivals_d{1,2}_r*`, `rivals_d{1,2}_merge` |
| Sequential certification (betting e-process, 20 reps): n=3,200/6,400/12,800 .040/.229/.425 (±.014/.021/.010), fdp 0; sequential z .188/.401/.592; n=4,800 .139 vs mixture .064 | `for lo in 0 5 10 15; do python scripts/eval/sequential.py $TB1 configs/eval/insample_d1.yaml configs/eval/sequential_d1.yaml $lo $((lo+5)); done` (no merge step; read the four chunks together) | `$O/d1/seq_seqbet_r{0-5,5-10,10-15,15-20}_raw.csv`; logs `seqbet_r*` |
| Adaptive witness, simulation (D2, 40 reps): two-stage adaptive n=60/200 .095/.461 (original), .082/.468 (content shift), not above one-stage (.104/.499, .162/.516) | `for lo in 0 10 20 30; do python scripts/eval/evaluate.py $TB2 configs/eval/adaptive_d2.yaml $lo $((lo+10)); done`; `merge.py $O/d2 ad_xm 40`; same with `adaptive_shift_d2.yaml`, `merge.py $O/d2 ad_xmsh 40` | `$O/d2/eval_ad_xm{,sh}_raw.csv`; logs `ad2_d2_r*`, `adsh2_d2_r*`, `merge_ad_xm{,sh}_d2` |
| Adaptive 1/π-weighted variant worse: D2 n=200 −.067±.029, D1 n=12,800 −.045±.017; unit-weight D1 +.05–.07 | `evaluate.py $TB2 configs/eval/adaptive_weighted_d2.yaml` and `adaptive_weighted_shift_d2.yaml` (unchunked); D1 `evaluate.py $TB1 configs/eval/adaptive_weighted_d1.yaml 0 10` and `adaptive_weighted_shift_d1.yaml 0 10` | `$O/d2/eval_adw_xm{,sh}_raw.csv`, `$O/d1/eval_adw_xm{,sh}_r0-10_raw.csv`; logs `adw_d{1,2}`, `adshw_d{1,2}` |
| Zero-label PU EM is weakly identified: U-stratum P(y≥1) at 30/300/510 iterations D3 .121/.020/.003 (human .189) | `python scripts/diagnostics/em_trajectory.py $D3 $TB2 configs/eval/adaptive_d2.yaml` | log `gap_em_traj` |
| Removing pool share and n_top from the PU features: D3 .214 (human [.154, .224]), D2 .401 (truth .415) at 1000 iterations | `python scripts/diagnostics/em_exclusion.py d3 ex 1000 $D3`; `python scripts/diagnostics/em_exclusion.py d2 {pu,ex} 1000 $TB2 configs/eval/main_d2.yaml` (D1: `d1 {pu,ex} 1000 $TB1 configs/eval/main_d1.yaml`) | logs `emx_d3_ex`, `emx_d2`, `emx_d2_ex`, `emx_d1_{pu,ex}` |
| PU without exclusion restriction (D2): n=200 .514 vs .495, paired +.020±.015 | `python scripts/eval/evaluate.py $TB2 configs/eval/pu_exclusion_d2.yaml` | `$O/d2/eval_xmex_raw.csv` (md5 bfb7ed50); log `xmex_d2` |

## Course principle points

| result | command(s) | output file(s) |
|---|---|---|
| Tokenisation D1 (DCG@5): char1 .782 / char2 .768 / jieba .766 | `python scripts/diagnostics/word_vectors.py $TB1 configs/eval/insample_d1.yaml` (prints the BM25 rows) | log `probe_wordvec` |
| Tokenisation D3: human char1, jieba − char2 +.131/+.138 vs official +.047/+.055; anchor-calibrated +.106/+.111 (z ≈ 1.9–2.0, not certified) (D3 human labels required) | `analyze.py` and `anchor.py` commands of Real data D3 | logs `d3_analyze_multi`, `d3_anchor` |
| D2 stemming +.064 (bm25_ws .796 → bm25_ws_stem .860) | `d2_judge_stats.py` command of Phenomenon | log `d2_stats` |
| Word vs sentence vectors: static word-vector mean .762 ≈ BM25; contextual sentence vectors +.055–.067 | `python scripts/diagnostics/word_vectors.py $TB1 configs/eval/insample_d1.yaml` | `$O/d1/probes/wv_{mean,idf}.npy`; log `probe_wordvec` |
| Fine-tuning vs prompting: fine-tuned CE AUC .695, judges + Qwen CE .701, combined .724 | `finetune_ce.py` + `finetuned_ce.py` commands of Coupling finding | log `diag_ftce` |

## Figures

`python analysis/fig_data.py $O results/figs/fig_data.json` (log `fig_data`), `cp $O/d3/d3_results.json $O/d3/d3_results_anchor.json results/figs/`, then `python analysis/make_figures.py results/figs/fig_data.json results/figs`. From the released copies, `python analysis/fig_data.py results/runs <out>/fig_data.json` recomputes the same figure data.

| figure | inputs |
|---|---|
| Fig. 1 `fig1_cert_budget` | `$O/d{1,2}/eval_xm_raw.csv` |
| Fig. 2 `fig2_coupling` | `$O/logs/d3_findings_rho{0,0.5,1}.log`, `$O/logs/provbias_d{1,2}_rho{0,0.5,1}.log` |
| Fig. 3 `fig3_d3_leaderboard` | WITNESS points, markers and BH regions: `results/figs/d3_results_anchor.json` (anchor-calibrated MI, `scripts/d3/anchor.py` section [4]); official and judge plug-in points: `results/figs/d3_results.json` (`scripts/d3/analyze.py`) |
| Fig. 4 `fig4_false_cert` | official and judge plug-in: `$O/d{1,2}/eval_v3c0_raw.csv`, `eval_v3c1_raw.csv`, `$O/d2/eval_sens_{b0,b2,b6,ghost}_raw.csv`; WITNESS (cross-fitted): `$O/d{1,2}/eval_xm_raw.csv` (grade-blind), `$O/d{1,2}/eval_xf_grade_raw.csv`, `$O/d2/eval_xf_poolbias_{b0,b2,b6,ghost}_raw.csv` (configs `xfit_grade_d{1,2}`, `xfit_poolbias_d2_*`) |
| Fig. 5 `fig5_collapse` | official: `$O/d2/eval_sens_{b0,b2,b25,b30,b35,b50,b6}_raw.csv`; WITNESS (cross-fitted): `$O/d2/eval_xf_poolbias_{b0,b2,b25,b30,b35,b50,b6}_raw.csv`; prediction: `$O/logs/theory_flip_d2_refined.log` |

`fig_data.py` asserts that the `official` and `judge_plugin` rows of each cross-fitted run equal those of the in-sample run (same qrels draws).

## Not computed

- Literature positioning of the coupling finding (no command; it is a citation, not a number).
