"""
Measure blocking quality on the val split (50k S1 entities) against the FULL train S2+S3 pool.

Run from the project root:
    .venv/bin/python code/business_entity_resolution/src/eval_blocking.py [--cap 300] [--k 100]

Reports, for different k (candidates kept per S1):
  - pair recall     : share of true (S1, match) pairs that made it into the candidates
  - entities full   : share of S1 entities (with matches) whose matches ALL survived
  - avg candidates  : per S1 entity
  - ceiling F0.5    : the score a PERFECT model would get on these candidates (the upper bound)
Saves the candidates to cache/val_candidates.parquet.
"""
import argparse
import json
import time

import pandas as pd

from blocking import candidates
from learn_translit import TRANSLIT_FILE
from metrics import f05_macro, parse_ids
from normalize import clean_address, clean_name, load_clean

INDIC = r"[ऀ-෿]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--k", type=int, default=100, help="max candidates kept per S1")
    ap.add_argument("--translit", action="store_true")
    ap.add_argument("--core-name-key", action="store_true")
    ap.add_argument("--bidir", action="store_true", help="bidirectional blocking (v7)")
    ap.add_argument("--rev-add", type=int, default=3)
    ap.add_argument("--out", default="val_candidates.parquet")
    args = ap.parse_args()
    tr = json.loads(TRANSLIT_FILE.read_text()) if args.translit else None
    cache = args.cache_dir
    t0 = time.time()

    split = pd.read_parquet(f"{cache}/split.parquet")
    val_ids = set(split.loc[split["split"] == "val", "source1_entity_id"])
    query = load_clean(cache, "train_source1", translit=tr)
    s1_full = query[["entity_id", "country", "name_c", "addr_c"]] if args.bidir else None
    query = query[query["entity_id"].isin(val_ids)].reset_index(drop=True)

    def load_pool(country):  # all train S2+S3 records of one country
        return pd.concat([load_clean(cache, "train_source2", country, tr),
                          load_clean(cache, "train_source3", country, tr)], ignore_index=True)

    cand = candidates(query, load_pool, cap=args.cap, k=args.k, core_name_key=args.core_name_key,
                      reverse_s1=s1_full, rev_add=args.rev_add)
    print(f"blocking: {len(cand):,} candidate pairs in {time.time() - t0:.0f}s")
    cand.to_parquet(f"{cache}/{args.out}", index=False)

    # Ground truth for val, as pairs and as sets.
    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet")
    gt = gt[gt["source1_entity_id"].isin(val_ids)]
    truth = {s: parse_ids(m) for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
    tp = gt.assign(cand=gt["matched_entity_ids"].map(parse_ids)).explode("cand").dropna(subset=["cand"])
    tp = tp.rename(columns={"source1_entity_id": "s1"})[["s1", "cand"]]
    n_with_matches = sum(1 for t in truth.values() if t)

    print(f"\n=== cap={args.cap}: blocking quality vs k (candidates kept per S1) ===")
    print(f"{'k':>6} {'pair recall':>12} {'entities full':>14} {'avg cands':>10} {'ceiling F0.5':>13}")
    ks = sorted({5, 10, 20, 30, 50, 100, args.k} - {x for x in [100] if x > args.k})
    if args.bidir:   # reverse-added pairs have rank k+1: count them in the "all" row only
        ks = ks + [args.k + 1]
    for k in ks:
        c = cand[cand["rank"] <= k]
        hit = tp.merge(c[["s1", "cand"]], on=["s1", "cand"], how="left", indicator=True)
        hit = hit.assign(found=hit["_merge"] == "both")
        full = hit.groupby("s1")["found"].all().sum() / n_with_matches
        # Perfect model on these candidates: predict exactly the true matches that survived.
        found = hit[hit["found"]].groupby("s1")["cand"].apply(set).to_dict()
        print(f"{k:>6} {hit['found'].mean():>12.4f} {full:>14.4f} "
              f"{len(c) / len(query):>10.1f} {f05_macro(found, truth):>13.4f}")
        if k == max(ks):
            hit50 = hit

    # Where are the misses at k=50? Look up just the true-match records.
    recs = []
    for name in ["train_source2", "train_source3"]:
        d = pd.read_parquet(f"{cache}/{name}.parquet")
        recs.append(d[d["entity_id"].isin(set(tp["cand"]))])
    recs = pd.concat(recs)
    info = pd.DataFrame({"country": recs["country"].to_numpy(),
                         "name_c": clean_name(recs["business_name"]).to_numpy(),
                         "addr_c": clean_address(recs["business_address"]).to_numpy()},
                        index=recs["entity_id"].to_numpy())
    h = hit50.join(info, on="cand")
    h["name_script"] = h["name_c"].str.contains(INDIC).map({True: "Indian script", False: "Latin"})
    h["source"] = h["cand"].str[:2]
    h["address"] = (h["addr_c"] == "").map({True: "empty", False: "present"})
    print(f"\n=== pair recall at k={max(ks)} ({'forward + reverse' if args.bidir else 'forward'}), broken down ===")
    for col in ["country", "name_script", "source", "address"]:
        print(h.groupby(col)["found"].agg(pairs="size", recall="mean").round(4).to_string(), "\n")
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
