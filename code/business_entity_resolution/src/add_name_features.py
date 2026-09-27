"""
Add the S1 name-uniqueness features (s1_context.NameIndex) to an existing run's labelled pairs.

    .venv/bin/python -u code/business_entity_resolution/src/add_name_features.py --from-run v3 --run v5

Reuses the blocking + features of --from-run (no new blocking): writes cache/runs/<run>/pairs_*.parquet
= the old pairs + s1_name_share + cand_name_s1_count. Counts come from the FULL train S1 file,
exactly as predict.py computes them from the full test S1 file.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from learn_translit import TRANSLIT_FILE
from normalize import clean_name, load_clean, transliterate
from s1_context import NameIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    t0 = time.time()
    src, out = f"{args.cache_dir}/runs/{args.from_run}", f"{args.cache_dir}/runs/{args.run}"
    os.makedirs(out, exist_ok=True)
    config = json.loads(open(f"{src}/config.json").read())
    tr = json.loads(TRANSLIT_FILE.read_text()) if config.get("translit") else None

    s1 = load_clean(args.cache_dir, "train_source1", translit=tr)
    index = NameIndex(s1)
    s1_share = dict(zip(s1["entity_id"], index.count(s1["name_c"].tolist(), s1["country"].tolist())))
    s1_country = dict(zip(s1["entity_id"], s1["country"]))
    print(f"name index built ({len(s1):,} S1) in {time.time() - t0:.0f}s; generic words per country:",
          {c: len(g) for c, g in index.generic.items()}, flush=True)

    cands = set()
    for split in ["train", "val", "holdout"]:
        cands |= set(pq.read_table(f"{src}/pairs_{split}.parquet", columns=["cand"])["cand"].to_pylist())
    cand_name = {}
    for n in ["train_source2", "train_source3"]:   # names only, candidates only: light on memory
        raw = pd.read_parquet(f"{args.cache_dir}/{n}.parquet", columns=["entity_id", "business_name"])
        raw = raw[raw["entity_id"].isin(cands)]
        names = clean_name(raw["business_name"])
        cand_name.update(zip(raw["entity_id"], transliterate(names, tr) if tr else names))
        del raw, names
    print(f"{len(cand_name):,} candidate names loaded in {time.time() - t0:.0f}s", flush=True)

    for split in ["train", "val", "holdout"]:
        pf = pq.ParquetFile(f"{src}/pairs_{split}.parquet")
        writer = None
        for batch in pf.iter_batches(batch_size=1_000_000):
            d = batch.to_pandas()
            d["s1_name_share"] = d["s1"].map(s1_share).to_numpy(np.float32)
            d["cand_name_s1_count"] = index.count([cand_name[c] for c in d["cand"]],
                                                  [s1_country[s] for s in d["s1"]])
            t = pa.Table.from_pandas(d, preserve_index=False)
            writer = writer or pq.ParquetWriter(f"{out}/pairs_{split}.parquet", t.schema)
            writer.write_table(t)
        writer.close()
        print(f"{split} done ({time.time() - t0:.0f}s)", flush=True)
    config.update({"name_index": True, "pairs_from": args.from_run})
    open(f"{out}/config.json", "w").write(json.dumps(config))
    print(f"done -> {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
