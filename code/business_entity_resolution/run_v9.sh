#!/bin/bash
# v9 = v8 (bidirectional blocking, filter, char features, transformer cross-encoder) with DEEPER reverse
# blocking: every S2/S3 record's top-10 S1 businesses (was top-3), still capped at 20 reverse-found per S1.
# Target: the blocking ceiling. v7/v8 val pair recall 0.9706 (ceiling F0.5 0.9895 < LB top 0.9907); the
# v7 gate measured that reverse top-10 recovers 49% of what forward top-50 misses (top-3: ~38%).
# The transformer is NOT retrained: v8's two fold models are reused (folds are by S1 hash in every run,
# so train scores stay out-of-fold). Same memory profile as v7 (k=50, 80% of train entities).
# Runs unattended; stops at the first error or failed gate.  Run from the project root:
#     bash code/business_entity_resolution/run_v9.sh
set -euo pipefail
PY=.venv/bin/python
SRC=code/business_entity_resolution/src
LOG=cache/run_v9.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

step "1/7 make_pairs: forward top-50 + reverse top-10 (cap 20 per S1), 80% of train entities"
$PY -u $SRC/make_pairs.py --run v9 --translit --bidir --k 50 --rev-add 10 --train-frac 0.8 >> "$LOG" 2>&1
step "   gate: val pair recall must beat v7/v8 (0.9706) by at least 0.002"
$PY - <<'EOF' 2>&1 | tee -a "$LOG" || { step "GATE FAILED: deeper blocking does not add enough recall, v8 stays the best"; exit 1; }
import sys
import pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "code/business_entity_resolution/src")
from metrics import parse_ids
split = pd.read_parquet("cache/split.parquet")
gt = pd.read_parquet("cache/train_ground_truth.parquet")
ids = set(split.loc[split.split == "val", "source1_entity_id"])
n_true = gt[gt.source1_entity_id.isin(ids)].matched_entity_ids.map(lambda x: len(parse_ids(x))).sum()
lab = pq.read_table("cache/runs/v9/pairs_val.parquet", columns=["label"]).column("label").to_numpy()
r = lab.sum() / n_true
print(f"   val pair recall {r:.4f} (v7/v8 0.9706), {len(lab) / len(ids):.1f} pairs per S1")
sys.exit(0 if r >= 0.9726 else 1)
EOF
step "2/7 add name-uniqueness features"
$PY -u $SRC/add_name_features.py --from-run v9 --run v9n >> "$LOG" 2>&1
step "3/7 cascade: filter + final"
$PY -u $SRC/train_cascade.py --run v9n --from-run v9n --thresholds 0.01,0.03 >> "$LOG" 2>&1
step "4/7 transformer scores on the new survivors (v8's fine-tuned fold models, out-of-fold on train)"
$PY -u $SRC/train_ce.py --from-run v9n --run v9 --reuse v8 >> "$LOG" 2>&1
step "5/7 final stage: A / char / char + cross-encoder"
$PY -u $SRC/train_final.py --from-run v9n --run v9 --variants A,char,char_ce --ce-run v9 --min-gain -1 >> "$LOG" 2>&1
grep -E "^\[ *[0-9]+s\] (A|char|char_ce) |SAVED" "$LOG" | tail -4 | tee -a "$LOG"
HOLD=$(grep -E "SAVED" "$LOG" | tail -1 | sed -E 's/.*holdout ([0-9.]+).*/\1/')
step "   gate: holdout $HOLD must beat v8 (0.9794)"
$PY -c "import sys; sys.exit(0 if float('$HOLD') > 0.9794 else 1)" || { step "GATE FAILED: v8 stays the best"; exit 1; }
step "6/7 smoke test on 2,000 test S1"
$PY -u $SRC/predict.py --run v9 --limit 2000 --out-dir cache/smoke_v9 >> "$LOG" 2>&1
step "7/7 full test run + official validator"
mkdir -p submissions/v9_rev10
$PY -u $SRC/predict.py --run v9 --out-dir submissions/v9_rev10 >> "$LOG" 2>&1
python3 utils/validate_submission.py --matching submissions/v9_rev10/matching_results.tsv \
    --candidate submissions/v9_rev10/candidate_pairs.tsv --test-dir dataset/test --check-ids >> "$LOG" 2>&1
step "DONE"
