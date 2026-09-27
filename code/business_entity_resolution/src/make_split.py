"""
Split TRAIN Source 1 entities into train / val / holdout, grouped by S1 entity.

Run from the project root (after convert_to_parquet.py):
    .venv/bin/python code/business_entity_resolution/src/make_split.py

Why split by S1 entity: an entity and ALL its matched S2/S3 records land on the same side,
so nothing learned from training entities reveals the answers for val / holdout.

- train   : entities the model learns from
- val     : entities we tune on (thresholds, feature choices)
- holdout : entities we look at rarely, to report an honest score
- unused  : the rest. Not scored, but their S2/S3 records stay in the search pool as look-alikes.

Output: cache/split.parquet  (source1_entity_id, country, split)
"""
import argparse
from pathlib import Path

import pandas as pd

from metrics import f05_macro, parse_ids

SIZES = {"train": 300_000, "val": 50_000, "holdout": 50_000}
SEED = 42  # fixed, so the split is identical on every run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    cache = Path(ap.parse_args().cache_dir)

    s1 = pd.read_parquet(cache / "train_source1.parquet", columns=["entity_id", "country"])
    gt = pd.read_parquet(cache / "train_ground_truth.parquet")

    # 1. Shuffle once, then number the entities inside each country: 0, 1, 2, ...
    df = s1.rename(columns={"entity_id": "source1_entity_id"}).sample(frac=1, random_state=SEED)
    pos = df.groupby("country").cumcount()

    # 2. Stratify by country: each country gets its fair share of every split,
    #    so every split has the same country mix. Countries are read from the data, not hard-coded.
    share = df["country"].map(df["country"].value_counts(normalize=True))
    df["split"] = "unused"
    lo = pd.Series(0.0, index=df.index)
    for name, size in SIZES.items():
        hi = lo + (share * size).round()
        df.loc[(pos >= lo) & (pos < hi), "split"] = name
        lo = hi

    # 3. Attach the true matches (as sets) for the checks below.
    gt["matches"] = gt["matched_entity_ids"].map(parse_ids)
    df = df.merge(gt[["source1_entity_id", "matches"]], on="source1_entity_id", how="left")
    assert df["matches"].notna().all(), "some S1 entities have no ground-truth row"

    # 4. Leakage checks.
    assert df["source1_entity_id"].is_unique, "an S1 entity appears twice"
    used = df[df["split"] != "unused"]
    matched = used["matches"].explode().dropna()
    assert matched.is_unique, "an S2/S3 record belongs to entities in two splits"
    print("Leakage checks passed: every S1 entity is in one split; no S2/S3 record is shared.\n")

    # 5. Summary: every split should look alike.
    used = used.assign(n=used["matches"].map(len))
    summary = used.groupby("split").agg(
        entities=("n", "size"),
        singletons=("n", lambda x: (x == 0).mean()),
        avg_matches=("n", "mean"),
    )
    by_country = pd.crosstab(used["split"], used["country"], normalize="index")
    print(summary.join(by_country.add_prefix("share_")).round(3).to_string())

    # 6. Scorer sanity check on val, using real data.
    val = used[used["split"] == "val"]
    truth = dict(zip(val["source1_entity_id"], val["matches"]))
    print(f"\nval: predict the truth        -> F0.5 = {f05_macro(truth, truth):.4f}  (must be 1.0)")
    print(f"val: predict empty everywhere -> F0.5 = {f05_macro({}, truth):.4f}  "
          f"(must equal the val singleton share, {(val['n'] == 0).mean():.4f})")

    out = cache / "split.parquet"
    df[["source1_entity_id", "country", "split"]].to_parquet(out, index=False)
    print(f"\nSaved {out}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
