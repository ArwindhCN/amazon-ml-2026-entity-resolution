"""
End-to-end prediction on the TEST set: blocking -> features -> matcher -> the two output files.

Run from the project root (after convert_to_parquet.py, and train.py + decide.py for the model):
    .venv/bin/python -u code/business_entity_resolution/src/predict.py --run v2   # LightGBM
    .venv/bin/python -u code/business_entity_resolution/src/predict.py --matcher rule
Then validate:
    python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test

Writes:
  output/candidate_pairs.tsv   every candidate the matcher scored (top-k per S1 from blocking)
  output/matching_results.tsv  predicted matches (always a subset of the candidates)
Both have exactly one row per test S1 entity, in the order of test_source1.tsv.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import decide
import rule
from blocking import iter_candidates
from features import CHAR_FEATURES, char_features, group_features, pair_features, pair_level_features
from learn_translit import TRANSLIT_FILE
from normalize import load_clean
from reverse import REV_FEATURES, ReverseSearcher, reverse_features
from s1_context import NameIndex, _core, add_s1_context

KEEP_P = 0.02   # pairs below this probability can never be selected: don't keep them in memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--matcher", choices=["lgbm", "rule"], default="lgbm")
    ap.add_argument("--run", default=None, help="model run to use (src/model/<run>/), e.g. v2")
    ap.add_argument("--k", type=int, default=50, help="candidates per S1 fed to the matcher")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N S1 records")
    args = ap.parse_args()
    cache, out = args.cache_dir, args.out_dir
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    translit, cascade = None, False
    if args.matcher == "lgbm":
        assert args.run, "--run is required for the lgbm matcher"
        booster = decide.load_booster(args.run)
        params = decide.load_params(args.run)
        config = decide.load_config(args.run)
        # an ensemble of final models (probabilities averaged), or the single lgbm.txt
        finals = [lgb.Booster(model_file=str(decide.model_dir(args.run) / f))
                  for f in config.get("final_models", ["lgbm.txt"])]
        ce_models = []
        if config.get("ce_models"):   # fine-tuned transformer cross-encoders (averaged)
            import cross_encoder as CE
            ce_models = [CE.load(decide.model_dir(args.run) / m) for m in config["ce_models"]]
            print(f"cross-encoders loaded on {ce_models[0][1].device}", flush=True)
        if config.get("translit"):
            translit = json.loads(TRANSLIT_FILE.read_text())
        if config.get("cascade"):   # learned candidate filter before the final model
            cascade = True
            filt = lgb.Booster(model_file=str(decide.model_dir(args.run) / "filter.txt"))
            filter_t = config["filter_t"]
            filter_trees = config.get("filter_trees")   # None = all trees
        args.k = config.get("k", args.k)   # candidates per S1 the model was trained with
        print(f"run {args.run}: config {config} -> k={args.k}", flush=True)
    else:
        params = rule.load_params()
    print(f"matcher {args.matcher}, params {params}", flush=True)

    query = add_s1_context(load_clean(cache, "test_source1", translit=translit))  # full S1 file
    name_index, rev_searcher = None, None
    if args.matcher == "lgbm" and config.get("name_index"):   # S1 name uniqueness (full S1 file)
        name_index = NameIndex(query)
        query["name_share"] = name_index.count(query["name_c"].tolist(), query["country"].tolist())
    if args.matcher == "lgbm" and config.get("reverse"):   # reverse competition vs the FULL S1 file
        rev_searcher = ReverseSearcher(query[["entity_id", "country", "name_c", "addr_c"]])
    if args.limit:
        query = query.head(args.limit)
    s1_order = query["entity_id"].tolist()

    def load_pool(country):  # all test S2+S3 records of one country
        return pd.concat([load_clean(cache, "test_source2", country, translit),
                          load_clean(cache, "test_source3", country, translit)], ignore_index=True)

    cand_lists, scored, n_pairs, n_cands = {}, [], 0, 0
    bidir = args.matcher == "lgbm" and config.get("bidir")   # v7: bidirectional blocking
    batches = iter_candidates(query, load_pool, k=args.k, with_text=True,
                              reverse_s1=query[["entity_id", "country", "name_c", "addr_c"]] if bidir else None,
                              rev_add=config.get("rev_add", 3) if bidir else 3,
                              **({"rev_cap": config["rev_cap"]} if bidir and "rev_cap" in config else {}),
                              q_extra=("addr_share", "street_share", "country")
                                      + (("name_share",) if name_index else ()),
                              log=lambda m: print(m, flush=True))
    for bi, b in enumerate(batches):
        n_pairs += len(b)   # blocking output (before the filter)
        if name_index:
            b["cand_name_s1_count"] = name_index.count(b["cand_name"].tolist(), b["s1_country"].tolist())
        if cascade:
            f = pair_level_features(b)
            keep = filt.predict(f[filt.feature_name()].to_numpy(np.float32),
                                num_iteration=filter_trees) >= filter_t
            b, f = b[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
            group_features(f, b["s1"].to_numpy())   # the survivors ARE candidate_pairs.tsv
            if config.get("char_features"):
                ctry = b["s1_country"].tolist()
                cf = char_features(b["s1_name"].tolist(), b["cand_name"].tolist(),
                                   b["s1_addr"].tolist(), b["cand_addr"].tolist(),
                                   [_core(x, name_index.generic[c]) for x, c in zip(b["s1_name"].tolist(), ctry)],
                                   [_core(x, name_index.generic[c]) for x, c in zip(b["cand_name"].tolist(), ctry)])
                for col in CHAR_FEATURES:
                    f[col] = cf[col].to_numpy()
            if ce_models:
                ta = CE.pair_text(b["s1_name"].tolist(), b["s1_addr"].tolist())
                tb = CE.pair_text(b["cand_name"].tolist(), b["cand_addr"].tolist())
                f["ce"] = np.mean([CE.score(tok, m, ta, tb) for tok, m in ce_models], axis=0)
            if rev_searcher is not None:
                recs = pd.DataFrame({"entity_id": b["cand"].to_numpy(), "country": b["s1_country"].to_numpy(),
                                     "name_c": b["cand_name"].to_numpy(), "addr_c": b["cand_addr"].to_numpy()})
                recs = recs.drop_duplicates("entity_id").reset_index(drop=True)
                rf = reverse_features(b["s1"].to_numpy(), b["cand"].to_numpy(), rev_searcher.query(recs))
                for col in REV_FEATURES:
                    f[col] = rf[col].to_numpy()
        else:
            f = pair_features(b)
        if args.matcher == "lgbm":
            p = np.mean([decide.predict_proba(m, f) for m in finals], axis=0)
            keep = p >= KEEP_P
        else:
            keep = rule.predict(f, params)
            p = rule.strength(f)
        scored.append(pd.DataFrame({"s1": b["s1"].to_numpy()[keep],
                                    "cand": b["cand"].to_numpy()[keep], "p": p[keep]}))
        cand_lists.update(b.groupby("s1", sort=False)["cand"].agg(",".join).to_dict())
        n_cands += len(b)
        if bi % 25 == 0:
            print(f"  batch {bi}: {n_pairs:,} blocking pairs, {n_cands:,} candidates, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"blocking: {n_pairs:,} pairs -> candidate_pairs: {n_cands:,} "
          f"({n_cands / len(s1_order):.2f} per S1) in {time.time() - t0:.0f}s", flush=True)

    scored = pd.concat(scored, ignore_index=True)
    if args.matcher == "lgbm":  # keep the probabilities: the decision can be re-tuned without re-running
        scored.to_parquet(f"{out}/test_scored_pairs.parquet", index=False)
    scored = decide.compete(scored)
    matches = decide.select(scored, params) if args.matcher == "lgbm" else scored
    match_lists = matches.groupby("s1")["cand"].agg(",".join).to_dict()

    # One row per S1, in file order; empty list when there is nothing.
    with open(f"{out}/candidate_pairs.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        fh.writelines(f"{s}\t{cand_lists.get(s, '')}\n" for s in s1_order)
    with open(f"{out}/matching_results.tsv", "w", encoding="utf-8") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        fh.writelines(f"{s}\t{match_lists.get(s, '')}\n" for s in s1_order)

    n_matched = sum(1 for s in s1_order if s in match_lists)
    print(f"S1 rows: {len(s1_order):,} | with candidates: {len(cand_lists):,} | "
          f"with >=1 match: {n_matched:,} ({n_matched / len(s1_order):.1%}) | "
          f"matched pairs: {len(matches):,} ({len(matches) / len(s1_order):.2f} per S1)")
    by_country = query.assign(matched=query["entity_id"].isin(match_lists))
    print("share of S1 with >=1 match, by country:",
          by_country.groupby("country")["matched"].mean().round(3).to_dict())
    print(f"total {time.time() - t0:.0f}s -> {out}/", flush=True)


if __name__ == "__main__":
    main()
