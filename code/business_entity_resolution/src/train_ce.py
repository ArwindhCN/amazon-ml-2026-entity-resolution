"""
Fine-tune the transformer cross-encoder (cross_encoder.py) on the filter survivors and write its
scores for train (out-of-fold), val and holdout, to be used as a feature by train_final.py.

    .venv/bin/python -u code/business_entity_resolution/src/train_ce.py --from-run v7c --run v8

- Survivors are exactly those of train_final.py: filter probability >= cutoff, with out-of-fold filter
  probabilities on train (cache/runs/<from-run>/pfilter_oof_train.npy).
- Two folds of the TRAIN-split survivors (by S1 entity). Model k is fine-tuned on a sample of fold k and
  scores the OTHER fold, so every train survivor gets a score from a model that never saw it.
  Val / holdout (and later test) are scored by the average of both models. Val/holdout labels are never
  used for training.
- --reuse <run>: take that run's fine-tuned ce_fold0/1 instead of fine-tuning again (e.g. new blocking,
  same train entities). Still out-of-fold: folds are by S1 hash, the same in every run.
- Saves src/model/<run>/ce_fold0, ce_fold1 and cache/runs/<run>/ce_<split>.parquet (s1, cand, ce).
"""
import argparse
import json
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import cross_encoder as CE
import decide
from features import PAIR_FEATURES, read_pairs
from learn_translit import TRANSLIT_FILE
from normalize import clean_address, clean_name, transliterate
from train_final import filter_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--from-run", required=True, help="cascade run with filter + OOF filter probabilities")
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-train", type=int, default=300_000, help="fine-tuning pairs per fold")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--reuse", default=None,
                    help="run whose fine-tuned ce_fold0/1 to reuse instead of fine-tuning (folds are by S1 hash, "
                         "so model k still only ever scores S1 entities of the other fold)")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    C, src = args.cache_dir, f"{args.cache_dir}/runs/{args.from_run}"
    out_cache, out_model = f"{C}/runs/{args.run}", decide.model_dir(args.run)
    os.makedirs(out_cache, exist_ok=True); out_model.mkdir(parents=True, exist_ok=True)
    cfg = decide.load_config(args.from_run)
    filt = lgb.Booster(model_file=str(decide.model_dir(args.from_run) / "filter.txt"))

    data = {s: read_pairs(f"{src}/pairs_{s}.parquet", ["s1", "cand", "label"] + filt.feature_name())
            for s in ["train", "val", "holdout"]}
    surv = {}
    for s, d in data.items():
        p = filter_oof(src, filt, data["train"], data["val"], log) if s == "train" else filt.predict(d[filt.feature_name()].to_numpy(np.float32))
        surv[s] = d[p >= cfg["filter_t"]][["s1", "cand", "label"]].reset_index(drop=True)
    del data
    log("survivors: " + ", ".join(f"{s} {len(d):,}" for s, d in surv.items()))

    tr = json.loads(TRANSLIT_FILE.read_text()) if cfg.get("translit") else None
    s1 = pd.read_parquet(f"{C}/train_source1.parquet")
    s1 = pd.DataFrame({"name_c": clean_name(s1.business_name).to_numpy(),
                       "addr_c": clean_address(s1.business_address).to_numpy()}, index=s1.entity_id.to_numpy())
    ids = list(set().union(*[set(d.cand) for d in surv.values()]))
    pool = []
    for i in (2, 3):
        r = pd.read_parquet(f"{C}/train_source{i}.parquet", filters=[("entity_id", "in", ids)])
        nm = clean_name(r.business_name)
        pool.append(pd.DataFrame({"name_c": (transliterate(nm, tr) if tr else nm).to_numpy(),
                                  "addr_c": clean_address(r.business_address).to_numpy()}, index=r.entity_id.to_numpy()))
    pool = pd.concat(pool)
    text = {s: (CE.pair_text(s1.name_c.reindex(d.s1), s1.addr_c.reindex(d.s1)),
                CE.pair_text(pool.name_c.reindex(d.cand), pool.addr_c.reindex(d.cand))) for s, d in surv.items()}
    log("texts ready")

    fold = (pd.util.hash_array(surv["train"]["s1"].to_numpy(dtype=object)) % 2).astype(int)
    y = surv["train"]["label"].to_numpy().astype(np.float32)
    ta, tb = text["train"]
    oof = np.full(len(y), np.nan, np.float32)
    avg = {s: np.zeros(len(surv[s]), np.float32) for s in ["val", "holdout"]}
    rng = np.random.default_rng(0)
    for k in [0, 1]:
        mine, other = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        if args.reuse:
            src_dir = decide.model_dir(args.reuse) / f"ce_fold{k}"
            tok, model = CE.load(src_dir)
            if src_dir.resolve() != (out_model / f"ce_fold{k}").resolve():
                shutil.copytree(src_dir, out_model / f"ce_fold{k}", dirs_exist_ok=True)
            log(f"fold {k}: reusing {src_dir} (fine-tuned on fold {k} only)")
        else:
            sample = rng.choice(mine, min(args.n_train, len(mine)), replace=False)
            tok, model = CE.load()
            log(f"fold {k}: fine-tuning on {len(sample):,} pairs ({args.epochs} epoch)")
            CE.fine_tune(tok, model, [ta[i] for i in sample], [tb[i] for i in sample], y[sample],
                         epochs=args.epochs, seed=k, log=log)
            model.save_pretrained(str(out_model / f"ce_fold{k}")); tok.save_pretrained(str(out_model / f"ce_fold{k}"))
        oof[other] = CE.score(tok, model, [ta[i] for i in other], [tb[i] for i in other], log=log)
        for s in avg:
            avg[s] += CE.score(tok, model, *text[s]) / 2
        log(f"fold {k}: scored the other fold ({len(other):,}) + val + holdout")
    assert not np.isnan(oof).any()
    for s, d in surv.items():
        d.assign(ce=oof if s == "train" else avg[s])[["s1", "cand", "ce"]].to_parquet(f"{out_cache}/ce_{s}.parquet", index=False)
    ce_cfg = ({**json.loads((decide.model_dir(args.reuse) / "ce_config.json").read_text()), "reused_from": args.reuse}
              if args.reuse else {"n_train": args.n_train, "epochs": args.epochs})
    json.dump({**ce_cfg, "ce_models": ["ce_fold0", "ce_fold1"], "from_run": args.from_run},
              open(out_model / "ce_config.json", "w"))
    log(f"saved scores to {out_cache} and models to {out_model}")


if __name__ == "__main__":
    main()
