#!/usr/bin/env bash
# One-command check of the reported results from the released data artifacts (CPU only, no LLM calls, no API keys).
#   1. fetch the artifact bundle into <root> and verify every file against MANIFEST.md5
#   2. unit tests of the guarantees
#   3. D3 (DuRetrieval): the human-certified leaderboard and its anchor-calibrated version, recomputed from the
#      released human labels and compared value by value with results/figs/d3_results{,_anchor}.json
#   4. main table on TREC DL 2019 (D2): repetitions 0-1 rerun and compared row by row with the released
#      40-repetition table (every repetition is seeded by (seed, rep), so a subset reproduces its rows exactly);
#      with --full-d1 the same check also runs on T2Ranking (D1), 6 to 14 minutes per repetition in the original logs
#   5. figure data recomputed from results/runs and compared with results/figs/fig_data.json, then every figure
#      redrawn from the recomputed data
# Verified on Ubuntu 22.04.5 LTS (x86_64) with Python 3.12.3, GNU bash 5.1, GNU coreutils 8.32 (md5sum, tar) and
# curl 7.81; see README "Environment" for macOS and Windows.
# usage: bash scripts/reproduce.sh <root> [--full-d1]    (<root> is created if absent)
set -euo pipefail

ROOT="$1"
FULL_D1="${2:-}"
[ -z "$FULL_D1" ] || [ "$FULL_D1" = "--full-d1" ]
BUNDLE_URL="https://github.com/rocinc/witness-certified-leaderboards/releases/download/v1.0/witness_artifacts_v1.tar.gz"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH=src WITNESS_ROOT="$ROOT"
LOGS="$ROOT/reproduce_logs"
FIGS="$ROOT/figs"
mkdir -p "$LOGS" "$FIGS"

for tool in curl tar md5sum python; do
  command -v "$tool" >/dev/null || { echo "missing '$tool' (see README, Environment)" >&2; exit 1; }
done

echo "[1/5] artifacts -> $ROOT"
if [ ! -f "$ROOT/MANIFEST.md5" ]; then
  curl -L --fail -o "$ROOT/witness_artifacts_v1.tar.gz" "$BUNDLE_URL"
  tar -xzf "$ROOT/witness_artifacts_v1.tar.gz" -C "$ROOT" --strip-components=1
fi
(cd "$ROOT" && md5sum --quiet -c MANIFEST.md5)
echo "    every file matches MANIFEST.md5"

echo "[2/5] unit tests"
python -m pytest -q tests

echo "[3/5] D3 leaderboard from the human labels"
L="$ROOT/outputs/d3/labels"
python scripts/d3/analyze.py configs/d3_duretrieval.yaml results/runs/logs/d3_batch2.log \
  "$L/batch1.csv" "$L/batch2.csv" > "$LOGS/d3_analyze_multi.log"
python scripts/d3/anchor.py configs/d3_duretrieval.yaml "$L/anchor.csv" "$L/batch1.csv" "$L/batch2.csv" \
  > "$LOGS/d3_anchor.log"
python scripts/check/compare_json.py "$ROOT/outputs/d3/d3_results.json" results/figs/d3_results.json
python scripts/check/compare_json.py "$ROOT/outputs/d3/d3_results_anchor.json" results/figs/d3_results_anchor.json

echo "[4/5] main table, repetitions 0-1"
python scripts/eval/evaluate.py configs/testbeds/d2_dl19.yaml configs/eval/main_d2.yaml 0 2 > "$LOGS/xm_d2_r0-2.log"
python scripts/check/compare_reps.py "$ROOT/outputs/d2/eval_xm_r0-2_raw.csv" results/runs/d2/eval_xm_raw.csv
if [ "$FULL_D1" = "--full-d1" ]; then
  python scripts/eval/evaluate.py configs/testbeds/d1_t2ranking.yaml configs/eval/main_d1.yaml 0 2 > "$LOGS/xm_d1_r0-2.log"
  python scripts/check/compare_reps.py "$ROOT/outputs/d1/eval_xm_r0-2_raw.csv" results/runs/d1/eval_xm_raw.csv
fi

echo "[5/5] figures"
python analysis/fig_data.py results/runs "$FIGS/fig_data.json" > "$LOGS/fig_data.log"
python scripts/check/compare_json.py "$FIGS/fig_data.json" results/figs/fig_data.json
cp "$ROOT/outputs/d3/d3_results.json" "$ROOT/outputs/d3/d3_results_anchor.json" "$FIGS/"
python analysis/make_figures.py "$FIGS/fig_data.json" "$FIGS" > "$LOGS/make_figures.log"
echo "    figures written to $FIGS (verification blocks in $LOGS/make_figures.log)"

echo "REPRODUCED: D3 results, main-table repetitions and figure data match the released results"
