#!/bin/bash
# new v8 = v7 (bidirectional blocking, filter, char features) + transformer cross-encoder score.
# Run after train_ce.py --from-run v7c --run v8 has finished. Stops at the first error.
set -euo pipefail
PY=.venv/bin/python
SRC=code/business_entity_resolution/src
LOG=cache/run_v8_ce.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
step "1/4 final stage: A / char / char + cross-encoder (holdout comparison)"
$PY -u $SRC/train_final.py --from-run v7c --run v8 --variants A,char,char_ce --ce-run v8 --min-gain -1 >> "$LOG" 2>&1
grep -E "^\[ *[0-9]+s\] (A|char|char_ce) |SAVED" "$LOG" | tail -4 | tee -a "$LOG"
step "2/4 smoke test on 2,000 test S1"
$PY -u $SRC/predict.py --run v8 --limit 2000 --out-dir cache/smoke_v8 >> "$LOG" 2>&1
step "3/4 full test run"
mkdir -p submissions/v8_ce
$PY -u $SRC/predict.py --run v8 --out-dir submissions/v8_ce >> "$LOG" 2>&1
step "4/4 official validator"
python3 utils/validate_submission.py --matching submissions/v8_ce/matching_results.tsv \
    --candidate submissions/v8_ce/candidate_pairs.tsv --test-dir dataset/test --check-ids >> "$LOG" 2>&1
step "DONE"
