"""
Rule-based matcher: the baseline for submission #1 (a LightGBM model replaces it later).

A candidate pair is predicted a match when one of three rules fires:
  (a) Latin-script name, address present:
        name_tset >= t_name  AND  addr_tset >= t_addr  AND  house numbers don't conflict
  (b) Indian-script name (can't compare with the Latin S1 name):
        addr_tset >= t_indic  AND  house numbers agree
  (c) Latin-script name, candidate address empty (only the name can decide):
        name_tset >= t_name_only
Then every S2/S3 record is kept only for the S1 record it matches best
(verified on train: each S2/S3 record belongs to at most one S1 entity).

Thresholds are tuned on val by tune_rule.py and stored in rule_params.json next to this file.
"""
import json
from pathlib import Path

PARAMS_FILE = Path(__file__).with_name("rule_params.json")


def load_params():
    return json.loads(PARAMS_FILE.read_text())


def predict(f, p):
    """f: DataFrame from features.pair_features; p: thresholds. Returns a boolean array."""
    latin = ~f["indic"]
    a = (latin & ~f["addr_empty"] & (f["num"] != 0)
         & (f["name_tset"] >= p["t_name"]) & (f["addr_tset"] >= p["t_addr"]))
    b = f["indic"] & (f["num"] == 1) & (f["addr_tset"] >= p["t_indic"])
    c = latin & f["addr_empty"] & (f["name_tset"] >= p["t_name_only"])
    return (a | b | c).to_numpy()


def strength(f):
    """How strong a predicted match is (used to pick the best S1 for a record)."""
    return (f["name_tset"] + f["addr_tset"]).to_numpy()


def best_s1_only(pairs):
    """pairs: DataFrame(s1, cand, strength) of predicted matches -> one S1 per candidate record."""
    return pairs.sort_values(["strength", "s1"], ascending=[False, True]).drop_duplicates("cand")
