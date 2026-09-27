"""
Build labelled candidate pairs with features for the train / val / holdout splits.

Run from the project root (after make_split.py):
    .venv/bin/python -u code/business_entity_resolution/src/make_pairs.py --run v3 --translit

For every S1 entity in those splits: blocking against the FULL train S2+S3 pool (exactly as on
test), then features for every candidate pair, then label = is this pair in the ground truth?
Output: cache/runs/<run>/pairs_<split>.parquet (s1, cand, label, features...) and
        cache/runs/<run>/config.json (settings that prediction must reuse, e.g. translit).
"""
import argparse
import json
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking import REV_CAP_PER_S1, iter_candidates
from features import pair_features
from metrics import parse_ids
from learn_translit import TRANSLIT_FILE
from normalize import load_clean
from s1_context import add_s1_context

SPLITS = ["train", "val", "holdout"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--run", required=True, help="name of this experiment, e.g. v3")
    ap.add_argument("--translit", action="store_true", help="translate Indian-script names")
    ap.add_argument("--bidir", action="store_true", help="bidirectional blocking (v7)")
    ap.add_argument("--rev-add", type=int, default=3, help="reverse candidates added per record")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="use this share of the TRAIN-split entities (val/holdout always complete)")
    args = ap.parse_args()
    cache = args.cache_dir
    out = f"{cache}/runs/{args.run}"
    os.makedirs(out, exist_ok=True)
    config = {"k": args.k, "translit": args.translit, "train_frac": args.train_frac,
              "bidir": args.bidir, "rev_add": args.rev_add, **({"rev_cap": REV_CAP_PER_S1} if args.bidir else {})}
    with open(f"{out}/config.json", "w") as fh:
        json.dump(config, fh)
    translit = json.loads(TRANSLIT_FILE.read_text()) if args.translit else None
    print(f"run {args.run}: {config}", flush=True)
    t0 = time.time()

    split = pd.read_parquet(f"{cache}/split.parquet")
    split = split[split["split"].isin(SPLITS)]
    if args.train_frac < 1.0:   # deterministic subset of train entities (by hash of the id)
        h = pd.util.hash_array(split["source1_entity_id"].to_numpy(dtype=object)) % 1000
        split = split[(split["split"] != "train") | (h < args.train_frac * 1000)]
    which = dict(zip(split["source1_entity_id"], split["split"]))
    query = add_s1_context(load_clean(cache, "train_source1", translit=translit))  # full S1 file
    s1_full = query[["entity_id", "country", "name_c", "addr_c"]] if args.bidir else None
    query = query[query["entity_id"].isin(which)].reset_index(drop=True)

    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet")
    gt = gt[gt["source1_entity_id"].isin(which)]
    pairs = gt.assign(cand=gt["matched_entity_ids"].map(parse_ids)).explode("cand").dropna(subset=["cand"])
    true_keys = pd.Index(pairs["source1_entity_id"] + "|" + pairs["cand"])
    print(f"{len(query):,} S1 entities, {len(true_keys):,} true pairs", flush=True)

    def load_pool(country):
        return pd.concat([load_clean(cache, "train_source2", country, translit),
                          load_clean(cache, "train_source3", country, translit)], ignore_index=True)

    writers, n = {}, {s: 0 for s in SPLITS}
    for bi, b in enumerate(iter_candidates(query, load_pool, k=args.k, with_text=True,
                                           q_extra=("addr_share", "street_share"),
                                           reverse_s1=s1_full, rev_add=args.rev_add,
                                           log=lambda m: print(m, flush=True))):
        f = pair_features(b)
        f.insert(0, "s1", b["s1"].to_numpy())
        f.insert(1, "cand", b["cand"].to_numpy())
        f.insert(2, "label", (b["s1"] + "|" + b["cand"]).isin(true_keys).to_numpy())
        f["split"] = f["s1"].map(which)
        for s, part in f.groupby("split"):
            t = pa.Table.from_pandas(part.drop(columns="split"), preserve_index=False)
            if s not in writers:
                writers[s] = pq.ParquetWriter(f"{out}/pairs_{s}.parquet", t.schema)
            writers[s].write_table(t)
            n[s] += len(part)
        if bi % 20 == 0:
            print(f"  batch {bi}: {sum(n.values()):,} pairs so far, {time.time() - t0:.0f}s", flush=True)
    for w in writers.values():
        w.close()
    print(f"done: {n} pairs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
