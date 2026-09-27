#!/bin/bash
# Resume v7 after the pair build: cap reverse-found candidates per S1 (hub fix), then cascade,
# final stage, test run, validators. Run from the project root.
set -euo pipefail
PY=.venv/bin/python
SRC=code/business_entity_resolution/src
LOG=cache/run_v7.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step "2b/6 cap reverse-found candidates per S1 (20) on the v7 pairs"
$PY -u $SRC/cap_pairs.py --from-run v7n --run v7c >> "$LOG" 2>&1
step "3/6 cascade: filter + final (reverse features used by both)"
$PY -u $SRC/train_cascade.py --run v7c --from-run v7c --thresholds 0.01,0.03 >> "$LOG" 2>&1
step "4/6 final stage: + character features"
$PY -u $SRC/train_final.py --from-run v7c --run v7 --variants A,char --min-gain -1 >> "$LOG" 2>&1
step "5/6 test run"
mkdir -p submissions/v7_bidir
$PY -u $SRC/predict.py --run v7 --out-dir submissions/v7_bidir >> "$LOG" 2>&1
step "6/6 validators"
python3 utils/validate_submission.py --matching submissions/v7_bidir/matching_results.tsv \
    --candidate submissions/v7_bidir/candidate_pairs.tsv --test-dir dataset/test --check-ids >> "$LOG" 2>&1
step "DONE"
