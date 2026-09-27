"""
Apply the per-S1 cap on reverse-found candidates (blocking.cap_reverse, REV_CAP_PER_S1) to an existing
run's labelled pairs, with exactly the rule blocking now applies at prediction time.

    .venv/bin/python -u code/business_entity_resolution/src/cap_pairs.py --from-run v7n --run v7c

Equivalent to re-running make_pairs with the cap: per-pair features do not depend on the other
candidates (group features are recomputed on the filter survivors later), so dropping rows is exact.
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

from blocking import REV_CAP_PER_S1, cap_reverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cap", type=int, default=REV_CAP_PER_S1)
    args = ap.parse_args()
    t0 = time.time()
    src, out = f"{args.cache_dir}/runs/{args.from_run}", f"{args.cache_dir}/runs/{args.run}"
    os.makedirs(out, exist_ok=True)
    for split in ["train", "val", "holdout"]:
        path = f"{src}/pairs_{split}.parquet"
        keys = pq.read_table(path, columns=["s1", "cand", "from_fwd", "rev_rank", "rev_score_ratio"]).to_pandas()
        keys["row"] = np.arange(len(keys))
        rev = keys[keys["from_fwd"] == 0].rename(columns={"s1": "q", "cand": "cid"})
        kept = cap_reverse(rev, args.cap)
        drop = np.zeros(len(keys), bool)
        drop[rev["row"].to_numpy()] = True
        drop[kept["row"].to_numpy()] = False
        pf, writer, start = pq.ParquetFile(path), None, 0
        for batch in pf.iter_batches(batch_size=1_000_000):
            n = batch.num_rows
            t = batch.filter(~drop[start:start + n]) if drop[start:start + n].any() else batch
            import pyarrow as pa
            t = pa.Table.from_batches([t]) if not isinstance(t, pa.Table) else t
            writer = writer or pq.ParquetWriter(f"{out}/pairs_{split}.parquet", t.schema)
            writer.write_table(t)
            start += n
        writer.close()
        print(f"{split}: {len(keys):,} -> {len(keys) - drop.sum():,} pairs (reverse-only {len(rev):,} -> {len(kept):,}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    config = json.loads(open(f"{src}/config.json").read())
    config.update({"rev_cap": args.cap, "pairs_from": args.from_run})
    open(f"{out}/config.json", "w").write(json.dumps(config))


if __name__ == "__main__":
    main()
