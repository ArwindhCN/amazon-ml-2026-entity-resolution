# Business Entity Resolution — reproduction guide

Pipeline: **raw TSV → Parquet → cleaning → bidirectional blocking → learned candidate filter →
pair features + transformer cross-encoder score → LightGBM match probability → per-entity decision →
`matching_results.tsv` + `candidate_pairs.tsv`**.

No external data or APIs are used. Learned artefacts, all built from the provided training files only:
the LightGBM models (MIT licence), an Indian-script → Latin word dictionary, and two fine-tuned copies
of the pretrained cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0, 22.7M parameters).
That pretrained model is not stored in this public repository: download it once into
`src/model/pretrained/ms-marco-MiniLM-L-6-v2/` as described in `src/model/README.md`. Trained models
are not published either (they are rebuilt by the steps below from the challenge data).

## Environment

- Tested on macOS (Apple silicon, M2), 8 GB RAM, Python 3.14. Peak memory ≈ 4–6 GB.
- The cross-encoder runs on the Apple GPU (MPS) or an NVIDIA GPU (CUDA) when present, else on the CPU
  (works, but several times slower).
- Install the pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

All commands below run from the folder that contains `dataset/` (the challenge's
`student_resource/` layout), with `dataset/train/` and `dataset/test/` holding the original TSVs.

## Steps

| # | Command | What it does | Time* |
|---|---|---|---|
| 0 | `.venv/bin/python code/business_entity_resolution/src/convert_to_parquet.py` | Copies every TSV to `cache/*.parquet` and checks each copy is identical | 1 min |
| 1 | `.venv/bin/python code/business_entity_resolution/src/make_split.py` | Splits TRAIN S1 entities into train 300k / val 50k / holdout 50k (seed 42, by entity, stratified by country) | 10 s |
| 2 | `.venv/bin/python code/business_entity_resolution/src/learn_translit.py` | Learns the Indian-script → Latin word dictionary from train-split pairs only → `src/model/translit.json` | 20 s |
| 3 | `.venv/bin/python -u code/business_entity_resolution/src/make_pairs.py --run v7 --translit --bidir --rev-add 3 --train-frac 0.8` | Bidirectional blocking (forward top-50 + each record's top-3 reverse S1, max 20 reverse-found per S1) + per-pair features + labels → `cache/runs/v7/` | 55 min |
| 4 | `.venv/bin/python -u code/business_entity_resolution/src/add_name_features.py --from-run v7 --run v7n` | Adds the S1 name-uniqueness features → `cache/runs/v7n/` | 3 min |
| 5 | `.venv/bin/python -u code/business_entity_resolution/src/train_cascade.py --run v7c --from-run v7n --thresholds 0.01,0.03` | Candidate filter (out-of-fold) + final model on its survivors; filter cutoff chosen on val → `src/model/v7c/` | 12 min |
| 6 | `.venv/bin/python -u code/business_entity_resolution/src/train_ce.py --from-run v7c --run v8` | Fine-tunes the cross-encoder on the filter survivors in 2 folds by S1 entity (out-of-fold train scores; val/holdout scored by both) → `src/model/v8/ce_fold0,1`, `cache/runs/v8/ce_*.parquet` | 100 min |
| 7 | `.venv/bin/python -u code/business_entity_resolution/src/train_final.py --from-run v7c --run v8 --variants A,char,char_ce --ce-run v8 --min-gain -1` | Final stage with character 3-gram features + cross-encoder score → `src/model/v8/` | 6 min |
| 8 | `.venv/bin/python -u code/business_entity_resolution/src/predict.py --run v8` | Whole pipeline on TEST → `output/matching_results.tsv`, `output/candidate_pairs.tsv` | ~6.5 h |

`run_v7.sh` chains steps 3–5 (and the v7 final stage) with a validation-recall gate; `run_v8_ce.sh`
chains steps 7–8 plus a smoke test and the validator. When our shipped v7 model was built, the pair
file was produced before the per-S1 cap existed, and `cap_pairs.py --from-run v7n --run v7c` applied the
identical cap afterwards (see `run_v7_resume.sh`). Per-pair features don't depend on the other candidates,
so this is equivalent to steps 3–4 above, where the cap sits inside blocking.

\*On the 8 GB laptop above.

Steps 1–7 build the models in `src/model/` (the final run is v8); with trained models in place,
steps 0 and 8 regenerate the outputs. Most of step 8's time is the
cross-encoder scoring ~7.2M test pairs twice on the laptop GPU.

## Validation of the outputs

```bash
# official validator from the challenge kit, not included here (candidate_pairs.tsv has ~7M IDs since the cascade, so the full check fits in memory)
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
# same rules for candidate_pairs.tsv, streamed line by line
.venv/bin/python code/business_entity_resolution/src/check_candidates.py \
    --candidate output/candidate_pairs.tsv --matching output/matching_results.tsv --check-ids
```

## Source files (`src/`)

| File | Role |
|---|---|
| `convert_to_parquet.py` | Step 0: TSV → Parquet, verified identical |
| `make_split.py` | Train/val/holdout split of S1 entities with leakage checks |
| `normalize.py` | Cleaning: Unicode/accents, case, punctuation, NULL/N/A, look-alike digits, transliteration |
| `learn_translit.py` | Learns Indian-script → Latin word dictionary from train-split ground truth |
| `blocking.py` | Candidate generation: rarity-based keys, sorted per-country index, top-k per S1, plus bidirectional (reverse) search with the per-S1 cap (stage 1) |
| `features.py` | 38 pair features (name/address similarities, house numbers, group comparisons) |
| `make_pairs.py` | Labelled pairs with features for train/val/holdout |
| `s1_context.py` | Whole-file S1 context: how many S1 businesses share an address, street or core name |
| `add_name_features.py` | Adds the name-uniqueness features to an existing run's pairs |
| `train_cascade.py` | Learned candidate filter + final model on the survivors (the cascade) |
| `train_final.py` | Final stage: character 3-gram features, cross-encoder score, ensembles; saves the chosen model |
| `cross_encoder.py` | Transformer cross-encoder: load, fine-tune, score (MPS / CUDA / CPU) |
| `train_ce.py` | Out-of-fold fine-tuning + scoring of the cross-encoder on the filter survivors (`--reuse` to reuse a run's fold models) |
| `reverse.py` | Reverse competition searched against the full S1 file (v6 final-stage version) |
| `cap_pairs.py` | Applies the per-S1 cap on reverse-found candidates to an existing run's pairs |
| `run_v7.sh`, `run_v7_resume.sh`, `run_v8_ce.sh`, `run_v9.sh` (in the folder above `src/`) | Unattended pipelines for the v7/v8/v9 experiments |
| `redecide.py` | Re-applies a re-tuned decision to saved test probabilities |
| `train.py` | LightGBM training |
| `decide.py` | Competition (each S2/S3 to one S1) + expected-F0.5 per-entity decision; tuning |
| `predict.py` | End-to-end test prediction and output writing |
| `metrics.py` | Macro F0.5 exactly as in the problem statement (self-tested) |
| `eval_blocking.py` | Blocking recall / ceiling measurement on val |
| `analyze_errors.py` | Val error analysis |
| `rule.py`, `tune_rule.py` | The first (rule-based) baseline, kept for reference |
| `check_candidates.py` | Memory-light validation of `candidate_pairs.tsv` |
