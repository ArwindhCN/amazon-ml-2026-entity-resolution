"""
Step 0 of the pipeline: copy the raw TSV files into Parquet (a fast, compressed binary format).

Run from the project root:
    .venv/bin/python code/business_entity_resolution/src/convert_to_parquet.py

- The raw TSVs in dataset/ are only read, never modified.
- Parquet copies are written to cache/ (same file names, .parquet extension).
- Every copy is checked: reloaded from Parquet, it must be identical to the TSV.
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd

FILES = [
    "train/train_source1", "train/train_source2", "train/train_source3",
    "train/train_ground_truth",
    "test/test_source1", "test/test_source2", "test/test_source3",
]


def read_tsv(path):
    # sep="\t"              -> files are tab-separated
    # dtype=str             -> everything stays text (IDs like S1-00001 stay text)
    # keep_default_na=False -> empty cells and words like "NULL" / "N/A" stay as-is, not NaN
    # quoting=QUOTE_NONE    -> a " inside an address is a normal character
    return pd.read_csv(path, sep="\t", dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE)


def mb(path):
    return path.stat().st_size / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--only", choices=["train", "test"], default=None,
                    help="convert only the train or only the test files (prediction needs only test)")
    args = ap.parse_args()

    data, cache = Path(args.data_dir), Path(args.cache_dir)
    cache.mkdir(exist_ok=True)

    for name in FILES:
        if args.only and not name.startswith(args.only + "/"):
            continue
        src = data / f"{name}.tsv"
        dst = cache / f"{Path(name).name}.parquet"

        t0 = time.time()
        df = read_tsv(src)
        t_tsv = time.time() - t0

        df.to_parquet(dst, index=False, compression="zstd")

        t0 = time.time()
        back = pd.read_parquet(dst)
        t_pq = time.time() - t0

        # The copy must be exact: same columns, same row order, same values.
        assert back.equals(df), f"{dst} differs from {src}"

        print(f"{Path(name).name:22s} {len(df):>10,} rows | "
              f"TSV {mb(src):6.0f} MB read in {t_tsv:5.1f}s | "
              f"Parquet {mb(dst):5.0f} MB read in {t_pq:4.1f}s | identical")
        del df, back  # free memory before the next file


if __name__ == "__main__":
    main()
