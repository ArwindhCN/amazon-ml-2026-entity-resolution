"""
From pair probabilities to the final match list of every S1 entity.

Step 1, competition: each S2/S3 record belongs to at most one S1 (verified on train). If several
        S1 records claim the same candidate, only the one with the highest probability keeps it.
Step 2, per-entity choice. Two strategies; tune_decision picks the better one on val:
  "threshold": predict every candidate with p >= t.
  "expected":  for each S1, choose HOW MANY of its top candidates to predict (0, 1, 2, ...) by
               maximising the EXPECTED F0.5. Per entity F0.5 = 1.25*TP / (n_pred + 0.25*n_true),
               so with probabilities sorted p1 >= p2 >= ...:
                   predict top m:    E ~ 1.25 * (p1 + ... + pm) / (m + 0.25 * (sum of all p + miss))
                   predict nothing:  E = P(no true match) ~ lam * (1-p1)(1-p2)...
               miss = expected true matches blocking lost; lam shifts the singleton trade-off.
               This is where "predict nothing" wins for likely singletons.

Run directly to tune on val (and report holdout) after train.py:
    .venv/bin/python -u code/business_entity_resolution/src/decide.py --run v2
"""
import argparse
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES
from metrics import f05_macro_pairs, parse_ids

MODEL_ROOT = Path(__file__).with_name("model")


def model_dir(run):
    """Everything one experiment needs at prediction time: lgbm.txt, config.json, decision.json."""
    return MODEL_ROOT / run


def load_booster(run):
    return lgb.Booster(model_file=str(model_dir(run) / "lgbm.txt"))


def load_config(run):
    return json.loads((model_dir(run) / "config.json").read_text())


def compete(df):
    """df: s1, cand, p. Keep each candidate only for the S1 that gives it the highest p."""
    return df.sort_values(["p", "s1"], ascending=[False, True]).drop_duplicates("cand")


def select(df, params):
    """df: s1, cand, p (after compete). Returns the predicted pairs."""
    if params["strategy"] == "threshold":
        return df[df["p"] >= params["t"]]
    d = df.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
    if d.empty:
        return d
    s1, p = d["s1"].to_numpy(), d["p"].to_numpy(np.float64)
    start = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1]          # first row of each S1
    gid = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(d)]))
    m = np.arange(len(d)) - start[gid] + 1                             # 1, 2, 3 ... within S1
    cum = np.cumsum(p)
    cum_g = cum - np.r_[0.0, cum][start][gid]                          # p1 + ... + pm
    total = np.add.reduceat(p, start)[gid]
    e_m = 1.25 * cum_g / (m + 0.25 * (total + params["miss"]))
    e_0 = params["lam"] * np.exp(np.add.reduceat(np.log1p(-np.clip(p, 0, 1 - 1e-9)), start))
    best = np.maximum.reduceat(e_m, start)
    m_star = np.minimum.reduceat(np.where(e_m >= best[gid], m, 10**9), start)
    keep = (best > e_0)[gid] & (m <= m_star[gid])
    return d[keep]


def decision_options():
    opts = [{"strategy": "threshold", "t": float(t)} for t in np.round(np.arange(0.05, 0.96, 0.025), 3)]
    opts += [{"strategy": "expected", "miss": float(m), "lam": float(l)}
             for m, l in itertools.product([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
                                           [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])]
    return opts


def tune_decision(d, ids, n_true):
    """d: s1, cand, label, p (after compete). Returns (best val F0.5, best params)."""
    best = (-1.0, None)
    for o in decision_options():
        sel = select(d, o)
        s = f05_macro_pairs(ids, n_true, sel["s1"], sel["label"])
        if s > best[0]:
            best = (s, o)
    return best


def load_params(run):
    return json.loads((model_dir(run) / "decision.json").read_text())


def predict_proba(booster, f):
    """Uses the feature list stored in the model, so older models keep working."""
    return booster.predict(f[booster.feature_name()].to_numpy(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cache = args.cache_dir
    runs = f"{cache}/runs/{args.run}"
    booster = load_booster(args.run)
    split = pd.read_parquet(f"{cache}/split.parquet")
    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet")
    n_true_all = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(lambda s: len(parse_ids(s)))))

    def prepare(name):
        cached = Path(f"{runs}/pred_{name}.parquet")   # probabilities are cached per run
        if cached.exists() and cached.stat().st_mtime > (model_dir(args.run) / "lgbm.txt").stat().st_mtime:
            d = pd.read_parquet(cached)
        else:
            d = pd.read_parquet(f"{runs}/pairs_{name}.parquet",
                                columns=["s1", "cand", "label"] + booster.feature_name())
            d = d.assign(p=predict_proba(booster, d))[["s1", "cand", "label", "p"]]
            d.to_parquet(cached, index=False)
        ids = split.loc[split["split"] == name, "source1_entity_id"].to_numpy()
        return compete(d), ids, {s: n_true_all[s] for s in ids}

    def score(d, ids, n_true, params):
        sel = select(d, params)
        return f05_macro_pairs(ids, n_true, sel["s1"], sel["label"]), sel

    val = prepare("val")
    options = [{"strategy": "threshold", "t": float(t)} for t in np.round(np.arange(0.05, 0.96, 0.025), 3)]
    options += [{"strategy": "expected", "miss": float(m), "lam": float(l)}
                for m, l in itertools.product([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
                                              [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])]
    results = [(score(*val, o)[0], o) for o in options]
    for strategy in ["threshold", "expected"]:
        s, o = max((r for r in results if r[1]["strategy"] == strategy), key=lambda r: r[0])
        print(f"best {strategy:9s} on val: F0.5 {s:.4f}  {o}")
    best_score, best = max(results, key=lambda r: r[0])
    (model_dir(args.run) / "decision.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"\nSaved model/{args.run}/decision.json: {best}")

    for name, data in [("val", val), ("holdout", prepare("holdout"))]:
        s, sel = score(*data, best)
        ids, n_true = data[1], data[2]
        n_pred = sel.groupby("s1").size().reindex(ids).fillna(0).to_numpy()
        t = np.array([n_true[i] for i in ids])
        print(f"{name:8s} F0.5 {s:.4f} | pair precision {sel['label'].mean():.4f} | "
              f"pair recall {sel['label'].sum() / t.sum():.4f} | singletons empty "
              f"{(n_pred[t == 0] == 0).mean():.4f} | matched-entities empty {(n_pred[t > 0] == 0).mean():.4f}")


if __name__ == "__main__":
    main()
