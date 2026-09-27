"""
Train the LightGBM pair classifier: P(this candidate is a true match of this S1 record).

Run from the project root (after make_pairs.py):
    .venv/bin/python -u code/business_entity_resolution/src/train.py --run v2

- Learns from train-split pairs only. Val pairs are used only to decide when to stop adding trees.
- Saves the model to src/model/<run>/lgbm.txt (LightGBM text format; LightGBM is MIT licensed)
  and copies the run's config.json next to it (prediction must reuse the same settings).
"""
import argparse
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES

MODEL_ROOT = Path(__file__).with_name("model")
PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,          # max leaves per tree: how complex each tree can be
    min_data_in_leaf=200,    # a leaf must cover >= 200 pairs: avoids memorising rare cases
    feature_fraction=0.8,    # each tree sees a random 80% of features (less overfitting)
    bagging_fraction=0.8,    # ... and a random 80% of rows
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=0,           # all cores
    seed=42,
    verbose=-1,
)


def load(runs, split):
    d = pd.read_parquet(f"{runs}/pairs_{split}.parquet", columns=FEATURES + ["label"])
    return d[FEATURES].to_numpy(np.float32), d["label"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    t0 = time.time()
    runs = f"{args.cache_dir}/runs/{args.run}"
    model_file = MODEL_ROOT / args.run / "lgbm.txt"

    X, y = load(runs, "train")
    Xv, yv = load(runs, "val")
    print(f"train pairs {len(y):,} ({y.mean():.3f} positive) | val pairs {len(yv):,}", flush=True)
    dtrain = lgb.Dataset(X, y, feature_name=FEATURES, free_raw_data=True)
    dval = lgb.Dataset(Xv, yv, reference=dtrain)
    del X
    booster = lgb.train(PARAMS, dtrain, num_boost_round=args.rounds, valid_sets=[dval],
                        valid_names=["val"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    model_file.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_file))
    shutil.copy(f"{runs}/config.json", model_file.parent / "config.json")
    print(f"saved {model_file} ({booster.best_iteration} trees) in {time.time() - t0:.0f}s")

    imp = pd.Series(booster.feature_importance("gain"), index=FEATURES)
    print("\nfeature importance (share of total gain):")
    print((imp / imp.sum()).sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
