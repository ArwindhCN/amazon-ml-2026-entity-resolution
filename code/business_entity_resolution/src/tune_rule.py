"""
Tune the rule's thresholds on val to maximise macro F0.5. Writes rule_params.json.

Run from the project root (after eval_blocking.py has written cache/val_candidates.parquet):
    .venv/bin/python code/business_entity_resolution/src/tune_rule.py [--k 50]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import rule
from features import pair_features
from metrics import f05_macro_pairs, parse_ids
from normalize import load_clean

GRID = np.round(np.arange(0.40, 1.001, 0.05), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--k", type=int, default=50, help="candidates per S1 fed to the matcher")
    args = ap.parse_args()
    cache = args.cache_dir
    t0 = time.time()

    # Val candidates (from eval_blocking.py) + cleaned text of both sides.
    split = pd.read_parquet(f"{cache}/split.parquet")
    val_ids = split.loc[split["split"] == "val", "source1_entity_id"]
    cand = pd.read_parquet(f"{cache}/val_candidates.parquet")
    cand = cand[cand["rank"] <= args.k].reset_index(drop=True)
    s1 = load_clean(cache, "train_source1").set_index("entity_id")
    pool = pd.concat([load_clean(cache, n) for n in ["train_source2", "train_source3"]])
    pool = pool[pool["entity_id"].isin(set(cand["cand"]))].set_index("entity_id")
    f = pair_features(s1["name_c"].reindex(cand["s1"]).reset_index(drop=True),
                      s1["addr_c"].reindex(cand["s1"]).reset_index(drop=True),
                      pool["name_c"].reindex(cand["cand"]).reset_index(drop=True),
                      pool["addr_c"].reindex(cand["cand"]).reset_index(drop=True))
    del s1, pool
    print(f"{len(cand):,} val pairs, features in {time.time() - t0:.0f}s")

    # Labels.
    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet")
    gt = gt[gt["source1_entity_id"].isin(set(val_ids))]
    truth = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(parse_ids)))
    n_true = {s: len(t) for s, t in truth.items()}
    label = np.array([c in truth[s] for s, c in zip(cand["s1"], cand["cand"])])
    st = rule.strength(f)

    def score(p):
        m = rule.predict(f, p)
        pairs = pd.DataFrame({"s1": cand["s1"].to_numpy()[m], "cand": cand["cand"].to_numpy()[m],
                              "strength": st[m], "label": label[m]})
        pairs = rule.best_s1_only(pairs)
        return f05_macro_pairs(val_ids, n_true, pairs["s1"], pairs["label"]), pairs

    # Coordinate search: improve one threshold at a time, repeat until nothing changes.
    p = {"t_name": 0.8, "t_addr": 0.6, "t_indic": 0.7, "t_name_only": 0.95}
    best, _ = score(p)
    print(f"start {p} -> val F0.5 {best:.4f}")
    for rnd in range(3):
        changed = False
        for name in p:
            for v in GRID:
                trial = {**p, name: float(v)}
                s, _ = score(trial)
                if s > best + 1e-6:
                    best, p, changed = s, trial, True
        print(f"round {rnd + 1}: {p} -> val F0.5 {best:.4f}")
        if not changed:
            break

    rule.PARAMS_FILE.write_text(json.dumps(p, indent=2) + "\n")
    _, pairs = score(p)
    n_pred = pairs.groupby("s1").size().reindex(val_ids).fillna(0)
    singles = [s for s in val_ids if n_true[s] == 0]
    print(f"\nSaved {rule.PARAMS_FILE.name}: {p}")
    print(f"val F0.5 {best:.4f} | pair precision {pairs['label'].mean():.4f} | "
          f"pair recall {pairs['label'].sum() / sum(n_true.values()):.4f}")
    print(f"singletons predicted empty: {(n_pred[singles] == 0).mean():.4f} | "
          f"entities with matches predicted empty: "
          f"{(n_pred[[s for s in val_ids if n_true[s] > 0]] == 0).mean():.4f}")


if __name__ == "__main__":
    main()
