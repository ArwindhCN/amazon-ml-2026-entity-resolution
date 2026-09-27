"""
Cascade: learned candidate FILTER, then the final matching model on the survivors only.

Built from an existing run's labelled pairs (no new blocking needed):
    .venv/bin/python -u code/business_entity_resolution/src/train_cascade.py --run v4 --from-run v3

Why: the organisers rank solutions with smaller candidate sets higher (candidate_pairs.tsv must be
the exact set the final model scores). Blocking keeps ~50 candidates per S1; most are obvious
non-matches. A cheap filter removes them, and only its survivors go to the final model.

Stage 1, filter:  LightGBM on PAIR_FEATURES (each pair on its own) -> p_filter; keep p_filter >= t.
                  Its probabilities for TRAIN pairs are out-of-fold (2 folds by S1 entity), so the
                  final model is trained on realistic survivor sets, not over-clean ones.
Stage 2, final:   group features recomputed on the survivors; LightGBM on PAIR + GROUP features;
                  then competition + per-entity decision (decide.py), tuned on val.
t is chosen on val: the smallest candidate set whose val F0.5 is within --tol of the best.
Saves src/model/<run>/: filter.txt, lgbm.txt, decision.json, config.json (with filter_t).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import decide
from features import GROUP_FEATURES, PAIR_FEATURES, group_features, read_pairs
from metrics import f05_macro_pairs, parse_ids

MODEL_ROOT = Path(__file__).with_name("model")
FILTER_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                     num_threads=0, seed=42, verbose=-1)
FINAL_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    num_threads=0, seed=42, verbose=-1)
FINAL_FEATURES = PAIR_FEATURES + GROUP_FEATURES


def fit(params, X, y, Xv, yv, names, rounds=3000):
    """LightGBM with early stopping on val (stops when val log-loss stops improving)."""
    dtr = lgb.Dataset(X, y, feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def survivors(d, p, t):
    """Keep pairs with filter probability >= t and recompute group features on them."""
    k = d[p >= t].reset_index(drop=True)
    return group_features(k, k["s1"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--run", required=True)
    ap.add_argument("--from-run", required=True, help="run whose pairs are reused, e.g. v3")
    ap.add_argument("--thresholds", default="0.001,0.003,0.01,0.03")
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--filter-rounds", type=int, default=600,
                    help="max filter trees (400 gave the same F0.5 as 2575 in v4, 7x faster)")
    args = ap.parse_args()
    t0 = time.time()
    src = f"{args.cache_dir}/runs/{args.from_run}"
    out = MODEL_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)

    cols = ["s1", "cand", "label"] + PAIR_FEATURES
    data = {s: read_pairs(f"{src}/pairs_{s}.parquet", cols) for s in ["train", "val", "holdout"]}
    X = {s: d[PAIR_FEATURES].to_numpy(np.float32) for s, d in data.items()}
    y = {s: d["label"].to_numpy() for s, d in data.items()}
    log(f"loaded pairs: " + ", ".join(f"{s} {len(d):,}" for s, d in data.items()))

    # ---- Stage 1: filter. Out-of-fold probabilities for train; full-train model for val/holdout/test.
    fold = (pd.util.hash_array(data["train"]["s1"].to_numpy(dtype=object)) % 2).astype(bool)
    p_filter = {"train": np.zeros(len(fold))}
    for f in [False, True]:
        m = fit(FILTER_PARAMS, X["train"][fold != f], y["train"][fold != f], X["val"], y["val"], PAIR_FEATURES,
                args.filter_rounds)
        p_filter["train"][fold == f] = m.predict(X["train"][fold == f])
        log(f"filter fold {int(f)}: {m.best_iteration} trees")
    filt = fit(FILTER_PARAMS, X["train"], y["train"], X["val"], y["val"], PAIR_FEATURES, args.filter_rounds)
    filt.save_model(str(out / "filter.txt"))
    for s in ["val", "holdout"]:
        p_filter[s] = filt.predict(X[s])
    log(f"filter (all train): {filt.best_iteration} trees -> saved")
    del X

    # ---- Ground truth for scoring.
    split = pd.read_parquet(f"{args.cache_dir}/split.parquet")
    gt = pd.read_parquet(f"{args.cache_dir}/train_ground_truth.parquet")
    n_true_all = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(lambda s: len(parse_ids(s)))))
    ids = {s: split.loc[split["split"] == s, "source1_entity_id"].to_numpy() for s in ["val", "holdout"]}
    n_true = {s: {i: n_true_all[i] for i in ids[s]} for s in ids}

    # ---- Stage 2 for each candidate threshold t.
    results = []
    for t in [float(x) for x in args.thresholds.split(",")]:
        kept = {s: survivors(data[s], p_filter[s], t) for s in data}
        final = fit(FINAL_PARAMS, kept["train"][FINAL_FEATURES].to_numpy(np.float32), kept["train"]["label"].to_numpy(),
                    kept["val"][FINAL_FEATURES].to_numpy(np.float32), kept["val"]["label"].to_numpy(), FINAL_FEATURES)
        scored = {}
        for s in ["val", "holdout"]:
            k = kept[s]
            scored[s] = decide.compete(pd.DataFrame({"s1": k["s1"], "cand": k["cand"], "label": k["label"],
                                                     "p": final.predict(k[FINAL_FEATURES].to_numpy(np.float32))}))
        val_f, params = decide.tune_decision(scored["val"], ids["val"], n_true["val"])
        sel = decide.select(scored["holdout"], params)
        hold_f = f05_macro_pairs(ids["holdout"], n_true["holdout"], sel["s1"], sel["label"])
        size = len(kept["val"]) / len(ids["val"])
        kept_true = kept["val"]["label"].sum() / sum(n_true["val"].values())
        log(f"t={t:<6} val cands/S1 {size:5.2f} | true pairs kept {kept_true:.4f} | final {final.best_iteration} trees"
            f" | val F0.5 {val_f:.4f} | holdout {hold_f:.4f} | {params}")
        results.append(dict(t=t, size=size, val=val_f, holdout=hold_f, params=params, model=final))

    best_val = max(r["val"] for r in results)
    chosen = min((r for r in results if r["val"] >= best_val - args.tol), key=lambda r: r["size"])
    chosen["model"].save_model(str(out / "lgbm.txt"))
    (out / "decision.json").write_text(json.dumps(chosen["params"], indent=2) + "\n")
    config = json.loads(Path(f"{src}/config.json").read_text())
    config.update({"cascade": True, "filter_t": chosen["t"], "from_run": args.from_run})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(f"CHOSEN t={chosen['t']}: val cands/S1 {chosen['size']:.2f}, val F0.5 {chosen['val']:.4f}, "
        f"holdout {chosen['holdout']:.4f} -> saved to {out}")


if __name__ == "__main__":
    main()
