"""
Re-apply a (re-tuned) decision to saved test probabilities, without re-running blocking/features.

    .venv/bin/python code/business_entity_resolution/src/redecide.py --run v2 --dir submissions/v2_lgbm

Reads <dir>/test_scored_pairs.parquet (written by predict.py) and rewrites <dir>/matching_results.tsv
with src/model/<run>/decision.json. candidate_pairs.tsv is unchanged (same candidates).
"""
import argparse

import pandas as pd

import decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    params = decide.load_params(args.run)
    scored = pd.read_parquet(f"{args.dir}/test_scored_pairs.parquet")
    matches = decide.select(decide.compete(scored), params)
    lists = matches.groupby("s1")["cand"].agg(",".join).to_dict()
    s1_order = pd.read_parquet(f"{args.cache_dir}/test_source1.parquet", columns=["entity_id"])["entity_id"]
    with open(f"{args.dir}/matching_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        fh.writelines(f"{s}\t{lists.get(s, '')}\n" for s in s1_order)
    print(f"{params} -> {len(matches):,} matched pairs, {len(lists):,} of {len(s1_order):,} S1 with a match")


if __name__ == "__main__":
    main()
