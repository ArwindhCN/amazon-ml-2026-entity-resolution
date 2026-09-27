#!/bin/bash
# Builds the final package Runtime_Terrors_submission.zip (structure from the challenge README):
#   output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/{src,README.md,requirements.txt},
#   Documentation_template.md
# Ships the FINAL run's model + the pretrained MiniLM + the translit dictionary (not the older experiment models).
# Run from the project root:   bash code/business_entity_resolution/make_submission_zip.sh v8 submissions/v8_ce
set -euo pipefail
RUN=$1
SUB=$2
OUT=Runtime_Terrors_submission.zip
STAGE=$(mktemp -d)/pkg
CODE=$STAGE/code/business_entity_resolution
mkdir -p $STAGE/output $CODE/src/model

if [ "${SKIP_VALIDATE:-0}" != 1 ]; then   # SKIP_VALIDATE=1 only for files already validated (saves ~2 GB RAM)
    python3 utils/validate_submission.py --matching $SUB/matching_results.tsv --candidate $SUB/candidate_pairs.tsv \
        --test-dir dataset/test --check-ids | tail -1
fi
cp $SUB/matching_results.tsv $SUB/candidate_pairs.tsv $STAGE/output/
rsync -a --exclude __pycache__ --exclude model code/business_entity_resolution/src/ $CODE/src/
cp code/business_entity_resolution/src/model/translit.json $CODE/src/model/
rsync -a code/business_entity_resolution/src/model/pretrained code/business_entity_resolution/src/model/$RUN $CODE/src/model/
cp code/business_entity_resolution/README.md code/business_entity_resolution/requirements.txt \
   code/business_entity_resolution/run_v7.sh code/business_entity_resolution/run_v7_resume.sh \
   code/business_entity_resolution/run_v8_ce.sh code/business_entity_resolution/run_v9.sh \
   code/business_entity_resolution/make_submission_zip.sh $CODE/
cp Documentation_template.md $STAGE/
# the shipped model must be the one that made the shipped outputs
cmp -s code/business_entity_resolution/src/model/$RUN/decision.json $SUB/decision.json \
    || { echo "ERROR: $SUB was not produced by model $RUN"; exit 1; }
rm -f $OUT
(cd $STAGE && zip -r -q -X ../$OUT . -x "*.DS_Store")
mv $STAGE/../$OUT .
unzip -Z1 $OUT | grep -E '^(output/.|Documentation|code/business_entity_resolution/[^/]+$|code/business_entity_resolution/src/model/[^/]+/?$)' | sort
ls -la $OUT
