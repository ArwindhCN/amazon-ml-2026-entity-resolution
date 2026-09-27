"""
Error analysis on val: where does the model + decision layer lose F0.5?

Run from the project root (after train.py and decide.py):
    .venv/bin/python code/business_entity_resolution/src/analyze_errors.py --run v2 [--n 15]

Prints: the F0.5 lost per entity type (singletons vs entities with matches, by country, by
Indian-script share), then example false merges and misses with their text.
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

import decide
from features import FEATURES
from metrics import f05_entity, parse_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cache = args.cache_dir

    booster = decide.load_booster(args.run)
    params = decide.load_params(args.run)
    d = pd.read_parquet(f"{cache}/runs/{args.run}/pairs_val.parquet")
    d["p"] = decide.predict_proba(booster, d)
    sel = decide.select(decide.compete(d[["s1", "cand", "p"]]), params)
    pred = sel.groupby("s1")["cand"].apply(set).to_dict()

    split = pd.read_parquet(f"{cache}/split.parquet")
    val = split[split["split"] == "val"].set_index("source1_entity_id")
    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet").set_index("source1_entity_id")
    truth = {s: parse_ids(gt.at[s, "matched_entity_ids"]) for s in val.index}

    ent = pd.DataFrame({"country": val["country"]})
    ent["n_true"] = [len(truth[s]) for s in ent.index]
    ent["f"] = [f05_entity(pred.get(s, set()), truth[s]) for s in ent.index]
    ent["type"] = np.where(ent["n_true"] == 0, "singleton", "has matches")
    print(f"val F0.5 {ent['f'].mean():.4f}\n")
    print("F0.5 lost (share of total loss) by entity type and country:")
    loss = (1 - ent["f"]).groupby([ent["type"], ent["country"]]).agg(["size", "mean", "sum"])
    loss["share_of_loss"] = loss["sum"] / loss["sum"].sum()
    print(loss.rename(columns={"size": "entities", "mean": "avg_loss"}).round(4).to_string(), "\n")

    # Pair-level error types.
    d["pred"] = d.set_index(["s1", "cand"]).index.isin(sel.set_index(["s1", "cand"]).index)
    fp, fn = d[d["pred"] & ~d["label"]], d[~d["pred"] & d["label"]]
    print(f"false merges {len(fp):,} | missed (in candidates) {len(fn):,} | "
          f"missed by blocking {sum(ent['n_true']) - d['label'].sum():,}")
    for name, e in [("false merges", fp), ("misses", fn)]:
        print(f"\n{name}: num agreement {e['num'].value_counts(normalize=True).round(3).to_dict()} | "
              f"indic {e['indic'].mean():.3f} | cand addr empty {e['addr_empty'].mean():.3f}")

    # Examples with text.
    s1 = pd.read_parquet(f"{cache}/train_source1.parquet").set_index("entity_id")
    ids = set(fp["cand"]) | set(fn["cand"])
    pool = pd.concat([pd.read_parquet(f"{cache}/train_source{i}.parquet").query("entity_id in @ids")
                      for i in (2, 3)]).set_index("entity_id")
    rng = np.random.default_rng(0)
    for name, e in [("FALSE MERGES", fp), ("MISSES", fn)]:
        print(f"\n===== {name} (random {args.n}) =====")
        for _, r in e.iloc[rng.choice(len(e), min(args.n, len(e)), replace=False)].iterrows():
            a, c = s1.loc[r["s1"]], pool.loc[r["cand"]]
            print(f"p={r['p']:.2f} n_true={len(truth[r['s1']])} | {a.business_name[:40]} | {a.business_address[:60]}")
            print(f"     -> {c.business_name[:40]} | {c.business_address[:60]}")


if __name__ == "__main__":
    main()
