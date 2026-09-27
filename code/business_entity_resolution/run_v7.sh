#!/bin/bash
# v7 = v6 recipe + BIDIRECTIONAL blocking: forward top-50 per S1, plus every S2/S3 record's top-3 S1
# businesses from a reverse search against the full S1 file; reverse features for every pair
# (filter and final model). Val: reverse top-10 recovers 49% of the pairs forward top-50 misses.
# Runs unattended; stops at the first error or if the val recall gate fails.
# Run from the project root:  bash code/business_entity_resolution/run_v7.sh
set -euo pipefail
PY=.venv/bin/python
SRC=code/business_entity_resolution/src
LOG=cache/run_v7.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

step "0/6 gate: val blocking recall with bidirectional blocking (must beat 0.9540)"
$PY -u $SRC/eval_blocking.py --translit --bidir --rev-add 3 --k 50 --out val_candidates_v7.parquet > cache/eval_blocking_v7.log 2>&1
cat cache/eval_blocking_v7.log >> "$LOG"
RECALL=$(grep -E "^ +51 " cache/eval_blocking_v7.log | awk '{print $2}')
step "   bidirectional pair recall = $RECALL"
$PY -c "import sys; sys.exit(0 if float('$RECALL') > 0.9540 else 1)" || { step "GATE FAILED: stopping"; exit 1; }

step "1/6 make_pairs (bidirectional, 80% of train entities to fit 8 GB; val/holdout complete)"
$PY -u $SRC/make_pairs.py --run v7 --translit --bidir --rev-add 3 --train-frac 0.8 >> "$LOG" 2>&1
step "2/6 add name-uniqueness features"
$PY -u $SRC/add_name_features.py --from-run v7 --run v7n >> "$LOG" 2>&1
step "3/6 cascade: filter + final (reverse features used by both)"
$PY -u $SRC/train_cascade.py --run v7n --from-run v7n --thresholds 0.01,0.03 >> "$LOG" 2>&1
step "4/6 final stage: + character features"
$PY -u $SRC/train_final.py --from-run v7n --run v7 --variants A,char --min-gain -1 >> "$LOG" 2>&1
step "5/6 test run"
mkdir -p submissions/v7_bidir
$PY -u $SRC/predict.py --run v7 --out-dir submissions/v7_bidir >> "$LOG" 2>&1
step "6/6 validators"
python3 utils/validate_submission.py --matching submissions/v7_bidir/matching_results.tsv \
    --candidate submissions/v7_bidir/candidate_pairs.tsv --test-dir dataset/test --check-ids >> "$LOG" 2>&1
step "DONE"
