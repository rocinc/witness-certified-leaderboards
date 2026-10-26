#!/bin/bash
# D1 testbed: build the items, then score every system (one GPU job at a time; the three BM25 runs are CPU-only and
# run in parallel).  Skips any output that already exists.
# usage: WITNESS_ROOT=<root> bash scripts/data/run_d1_systems.sh   (from the repository root)
set -euo pipefail
: "${WITNESS_ROOT:?set WITNESS_ROOT}"
export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
CFG=configs/testbeds/d1_t2ranking.yaml
OUT=$WITNESS_ROOT/outputs/d1
LOGS=$WITNESS_ROOT/outputs/logs
mkdir -p "$LOGS"
[ -f "$OUT/items.parquet" ] || python scripts/data/build_d1.py $CFG > "$LOGS/d1_build.log" 2>&1
for s in bm25_char1 bm25_char2 bm25_jieba; do
  [ -f "$OUT/scores/$s.npy" ] || python scripts/data/score_systems.py $CFG $s > "$LOGS/d1_$s.log" 2>&1 &
done
for s in bge_small_zh bge_base_zh gte_base_zh m3e_base text2vec_zh e5_small e5_base bge_m3 qwen3_emb rerank_qwen3; do
  [ -f "$OUT/scores/$s.npy" ] || python scripts/data/score_systems.py $CFG $s > "$LOGS/d1_$s.log" 2>&1
done
wait
