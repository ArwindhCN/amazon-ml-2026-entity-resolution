"""
Final-stage experiments on top of an existing cascade run (same blocking, same filter):
character 3-gram features and an ensemble of final models.

    .venv/bin/python -u code/business_entity_resolution/src/train_final.py --from-run v5 --run v6

Steps
1. Survivors exactly as in the cascade: out-of-fold filter probabilities on train (2 folds by S1,
   cached in cache/runs/<from>/pfilter_oof_train.npy), the saved filter on val/holdout, cutoff t.
2. Group features on the survivors; character 3-gram features (features.char_features) on the
   survivors only, from the cleaned (translated) texts.
3. Compare variants on val (decision tuned there) and holdout (--variants):
     A             final model with the from-run's features (reproduces the from-run)
     char          A + character 3-gram features
     rev           A + reverse-competition features (reverse.py)
     char_rev      A + both
     *_ens         the same as an ensemble of 3 LightGBMs with different settings (averaged)
4. Save the best variant (by val; only if it beats A on holdout by >= --min-gain) to src/model/<run>/.
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
from features import CHAR_FEATURES, GROUP_FEATURES, PAIR_FEATURES, char_features, group_features, read_pairs
from learn_translit import TRANSLIT_FILE
from metrics import f05_macro_pairs, parse_ids
from normalize import clean_address, clean_name, transliterate
from reverse import REV_FEATURES, reverse_candidates, reverse_features
from s1_context import NameIndex, _core
from train_cascade import FILTER_PARAMS, FINAL_PARAMS, fit

MODEL_ROOT = Path(__file__).with_name("model")
ENSEMBLE = [dict(FINAL_PARAMS, seed=1),
            dict(FINAL_PARAMS, seed=2, num_leaves=63, feature_fraction=0.7),
            dict(FINAL_PARAMS, seed=3, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.9)]


def filter_oof(src, filt, train, val, log=print):
    """Out-of-fold filter probabilities on train (2 folds by S1), cached in <src>/pfilter_oof_train.npy."""
    oof_file = Path(f"{src}/pfilter_oof_train.npy")
    if oof_file.exists():
        return np.load(oof_file)
    fcols = filt.feature_name()
    X, y = train[fcols].to_numpy(np.float32), train["label"].to_numpy()
    Xv, yv = val[fcols].to_numpy(np.float32), val["label"].to_numpy()
    fold = (pd.util.hash_array(train["s1"].to_numpy(dtype=object)) % 2).astype(bool)
    p_tr = np.zeros(len(y))
    for f in [False, True]:
        m = fit(FILTER_PARAMS, X[fold != f], y[fold != f], Xv, yv, fcols, filt.num_trees())
        p_tr[fold == f] = m.predict(X[fold == f])
        log(f"OOF filter fold {int(f)} done")
    np.save(oof_file, p_tr)
    return p_tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--variants", default="A,char,char_ens")
    ap.add_argument("--ce-run", default=None, help="run whose train_ce.py scores to use for *_ce variants")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    C, src = args.cache_dir, f"{args.cache_dir}/runs/{args.from_run}"
    cfg = decide.load_config(args.from_run)
    t = cfg["filter_t"]
    filt = lgb.Booster(model_file=str(decide.model_dir(args.from_run) / "filter.txt"))
    fcols = filt.feature_name()

    # ---- 1. survivors
    cols = sorted(set(["s1", "cand", "label"] + PAIR_FEATURES + fcols))
    data = {s: read_pairs(f"{src}/pairs_{s}.parquet", cols) for s in ["train", "val", "holdout"]}
    p = {"train": filter_oof(src, filt, data["train"], data["val"], log)}
    for s in ["val", "holdout"]:
        p[s] = filt.predict(data[s][fcols].to_numpy(np.float32))
    surv = {}
    for s in data:
        k = data[s][p[s] >= t].reset_index(drop=True)
        surv[s] = group_features(k, k["s1"].to_numpy())
    del data
    log("survivors: " + ", ".join(f"{s} {len(d):,}" for s, d in surv.items()))

    # ---- 2. character features from cleaned, translated texts
    tr = json.loads(TRANSLIT_FILE.read_text()) if cfg.get("translit") else None
    s1 = pd.read_parquet(f"{C}/train_source1.parquet")
    s1 = pd.DataFrame({"entity_id": s1["entity_id"], "country": s1["country"],
                       "name_c": clean_name(s1["business_name"]), "addr_c": clean_address(s1["business_address"])})
    index = NameIndex(s1)
    s1 = s1.set_index("entity_id")
    ids = set().union(*[set(d["cand"]) for d in surv.values()])
    variants = args.variants.split(",")
    pool = []
    for n in ["train_source2", "train_source3"]:
        r = pd.read_parquet(f"{C}/{n}.parquet")
        r = r[r["entity_id"].isin(ids)]
        nm = clean_name(r["business_name"])
        pool.append(pd.DataFrame({"name_c": (transliterate(nm, tr) if tr else nm).to_numpy(),
                                  "addr_c": clean_address(r["business_address"]).to_numpy(),
                                  "country": r["country"].to_numpy()},
                                 index=r["entity_id"].to_numpy()))
    pool = pd.concat(pool)
    if any("rev" in v for v in variants):
        # search backwards: every survivor record against the FULL train S1 file
        recs = pool.rename_axis("entity_id").reset_index()[["entity_id", "country", "name_c", "addr_c"]]
        s1_all = s1.reset_index()[["entity_id", "country", "name_c", "addr_c"]]
        rev = reverse_candidates(recs, s1_all, log=lambda m: log("reverse " + m.strip()))
        for sname, d in surv.items():
            rf = reverse_features(d["s1"].to_numpy(), d["cand"].to_numpy(), rev)
            for col in REV_FEATURES:
                d[col] = rf[col].to_numpy()
        del rev
        log("reverse-competition features added")
    for s, d in surv.items():
        sn, sa = s1["name_c"].reindex(d["s1"]).tolist(), s1["addr_c"].reindex(d["s1"]).tolist()
        cn, ca = pool["name_c"].reindex(d["cand"]).tolist(), pool["addr_c"].reindex(d["cand"]).tolist()
        country = s1["country"].reindex(d["s1"]).tolist()
        snc = [_core(x, index.generic[c]) for x, c in zip(sn, country)]
        cnc = [_core(x, index.generic[c]) for x, c in zip(cn, country)]
        cf = char_features(sn, cn, sa, ca, snc, cnc)
        for col in CHAR_FEATURES:
            d[col] = cf[col].to_numpy()
    log("character features added")
    if any("ce" in v for v in variants):   # transformer cross-encoder score (train: out-of-fold)
        for sname, d in surv.items():
            ce = pd.read_parquet(f"{C}/runs/{args.ce_run}/ce_{sname}.parquet")
            m = d[["s1", "cand"]].merge(ce, on=["s1", "cand"], how="left")
            assert m["ce"].notna().all(), f"{sname}: survivors without a cross-encoder score"
            d["ce"] = m["ce"].to_numpy(np.float32)
        log("cross-encoder scores added")

    # ---- 3. compare A / B / C
    split = pd.read_parquet(f"{C}/split.parquet")
    gt = pd.read_parquet(f"{C}/train_ground_truth.parquet")
    n_all = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"].map(lambda x: len(parse_ids(x)))))
    ev_ids = {s: split.loc[split["split"] == s, "source1_entity_id"].to_numpy() for s in ["val", "holdout"]}
    n_true = {s: {i: n_all[i] for i in ev_ids[s]} for s in ev_ids}
    base = decide.load_booster(args.from_run).feature_name()

    def evaluate(name, feats, params_list):
        models = [fit(pp, surv["train"][feats].to_numpy(np.float32), surv["train"]["label"].to_numpy(),
                      surv["val"][feats].to_numpy(np.float32), surv["val"]["label"].to_numpy(), feats)
                  for pp in params_list]
        sc = {}
        for s in ["val", "holdout"]:
            d = surv[s]
            prob = np.mean([m.predict(d[feats].to_numpy(np.float32)) for m in models], axis=0)
            sc[s] = decide.compete(pd.DataFrame({"s1": d["s1"], "cand": d["cand"], "label": d["label"], "p": prob}))
        vf, params = decide.tune_decision(sc["val"], ev_ids["val"], n_true["val"])
        sel = decide.select(sc["holdout"], params)
        hf = f05_macro_pairs(ev_ids["holdout"], n_true["holdout"], sel["s1"], sel["label"])
        log(f"{name:28s} val {vf:.4f} | holdout {hf:.4f} | trees {[m.best_iteration for m in models]} | {params}")
        return dict(name=name, val=vf, holdout=hf, params=params, models=models, feats=feats)

    def feats_of(v):
        return (base + (CHAR_FEATURES if "char" in v else []) + (REV_FEATURES if "rev" in v else [])
                + (["ce"] if "ce" in v else []))

    results = {v: evaluate(v, feats_of(v), ENSEMBLE if v.endswith("_ens") else [FINAL_PARAMS])
               for v in variants}
    A = results["A"]

    # ---- 4. save the best if it clearly beats A
    best = max([r for v, r in results.items() if v != "A"], key=lambda r: r["val"])
    if best["holdout"] < A["holdout"] + args.min_gain:
        log(f"no gain over A (best {best['name']} holdout {best['holdout']:.4f} vs {A['holdout']:.4f}): nothing saved")
        return
    out = MODEL_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(decide.model_dir(args.from_run) / "filter.txt", out / "filter.txt")
    files = []
    for i, m in enumerate(best["models"]):
        m.save_model(str(out / f"final_{i}.txt"))
        files.append(f"final_{i}.txt")
    shutil.copy(out / files[0], out / "lgbm.txt")   # decide.load_booster() compatibility
    (out / "decision.json").write_text(json.dumps(best["params"], indent=2) + "\n")
    new_cfg = dict(cfg, final_models=files, char_features="char" in best["name"],
                   reverse="rev" in best["name"], from_run=args.from_run)
    if "ce" in best["name"]:   # the fine-tuned cross-encoders travel with the model
        for k in ["ce_fold0", "ce_fold1"]:
            src_dir = decide.model_dir(args.ce_run) / k
            if src_dir.resolve() != (out / k).resolve():
                shutil.copytree(src_dir, out / k, dirs_exist_ok=True)
        new_cfg["ce_models"] = ["ce_fold0", "ce_fold1"]
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    log(f"SAVED {best['name']} -> {out}: val {best['val']:.4f}, holdout {best['holdout']:.4f}")


if __name__ == "__main__":
    main()
