"""
Pair features v2: how similar is an S1 record to one of its candidates?

Input: one batch from blocking.iter_candidates(..., with_text=True): columns s1, cand, score, rank,
s1_name, s1_addr, cand_name, cand_addr and their "core" versions (generic words removed, e.g.
"private", "limited", "llc", "sarl", "street"; generic = very frequent in that country's pool,
learned from the data, so it adapts to France without a hand-made list).

Feature groups
  name_*     similarity of names (several measures: each is fooled by different noise)
  addr_*     similarity of addresses
  num_*      house-number agreement (the planted look-alikes differ mainly here)
  *_core     similarity after generic words are removed
  record     lengths, empty address, Indian-script name, source S2/S3
  blocking   blocking score and rank
  group      how this candidate compares with the OTHER candidates of the same S1
             (is it the best? how far behind? how many strong candidates?) and how similar it is to
             the group's best candidate (true matches are copies of one record, so they resemble
             each other; a planted look-alike often does not fit).
Country is deliberately NOT a feature (open set: France is unseen in training).
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

INDIC = r"[ऀ-෿]"
NUM = re.compile(r"\b(\d+)")   # digits at the start of a word: "1151a" -> 1151, "b33" -> none

FEATURES = [
    "name_tset", "name_tsort", "name_ratio", "name_partial", "name_jw", "name_core_tset",
    "name_core_ratio",
    "addr_tset", "addr_tsort", "addr_ratio", "addr_partial", "addr_core_tset",
    "num", "num_first_eq", "num_jacc", "num_first_lev", "n_num_s1", "n_num_cand",
    "indic", "addr_empty", "cand_s3", "len_name_s1", "len_name_cand", "len_addr_s1",
    "len_addr_cand",
    "bscore", "brank",
    "n_cands", "combo", "combo_gap", "combo_rank", "combo_top2_gap", "n_strong",
    "name_gap", "addr_gap", "bscore_ratio", "ref_name_tset", "ref_addr_tset",
    # v3 additions
    "indic_src", "num_best_lev", "num_sub", "s1_addr_share", "s1_street_share",
    # v5 additions (s1_context.NameIndex)
    "s1_name_share", "cand_name_s1_count",
    # v7 additions (bidirectional blocking: blocking.reverse_table)
    "from_fwd", "rev_rank", "rev_score_ratio", "rev_n_close",
]


def _sim(a, b, scorer, scale=100.0):
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / scale).astype(np.float32)


def _numbers(texts):
    """Numbers in order of appearance, leading zeros stripped, no repeats."""
    out = []
    for t in texts:
        seen = []
        for n in NUM.findall(t):
            n = n.lstrip("0")
            if n and n not in seen:
                seen.append(n)
        out.append(seen)
    return out


def read_pairs(path, cols):
    """Read these columns from a pairs parquet; columns an older run does not have become NaN."""
    import pyarrow.parquet as pq
    have = set(pq.read_schema(path).names)
    d = pd.read_parquet(path, columns=[c for c in cols if c in have])
    for c in cols:
        if c not in have:
            d[c] = np.float32(np.nan)
    return d[cols]


# Features that compare a candidate with the OTHER candidates of the same S1 (depend on the set).
GROUP_FEATURES = ["n_cands", "combo", "combo_gap", "combo_rank", "combo_top2_gap", "n_strong",
                  "name_gap", "addr_gap", "bscore_ratio"]
REF_FEATURES = ["ref_name_tset", "ref_addr_tset"]          # need the text of the group's best
PAIR_FEATURES = [c for c in FEATURES if c not in GROUP_FEATURES + REF_FEATURES]


def pair_features(b):
    """All v3 features: per-pair, group, and similarity to the group's best candidate."""
    f = pair_level_features(b)
    grp = b["s1"].to_numpy()
    group_features(f, grp)
    cn, ca = b["cand_name"].tolist(), b["cand_addr"].tolist()
    rank = f["combo_rank"].to_numpy()

    # Similarity to the group's best candidate (for the best one itself: to the second best).
    pos = np.arange(len(b))
    first = pd.Series(pos[rank == 1], index=grp[rank == 1])
    second = pd.Series(pos[rank == 2], index=grp[rank == 2])
    ref = np.where(rank == 1, second.reindex(grp).to_numpy(), first.reindex(grp).to_numpy())
    has = ~np.isnan(ref)
    r = np.where(has, ref, 0).astype(np.int64)
    ref_name = np.where(has, np.asarray(cn, dtype=object)[r], "")
    ref_addr = np.where(has, np.asarray(ca, dtype=object)[r], "")
    f["ref_name_tset"] = np.where(has, _sim(cn, list(ref_name), fuzz.token_set_ratio), np.nan)
    f["ref_addr_tset"] = np.where(has, _sim(ca, list(ref_addr), fuzz.token_set_ratio), np.nan)

    return f.astype({c: np.float32 for c in FEATURES if c not in ("indic", "addr_empty")})


def group_features(f, grp):
    """Add GROUP_FEATURES to f (in place), comparing each row with the other rows of its S1 (grp).

    Computed only from per-pair columns, so it can be recomputed on any subset of candidates
    (e.g. after the candidate filter) without the text.
    """
    combo = f["name_tset"] + f["addr_tset"] + 0.5 * (f["num"] == 1) - 0.5 * (f["num"] == 0)
    f["combo"] = combo.astype(np.float32)
    g = f.groupby(grp)
    f["n_cands"] = g["combo"].transform("size").astype(np.float32)
    f["combo_gap"] = (f["combo"] - g["combo"].transform("max")).astype(np.float32)
    f["combo_rank"] = g["combo"].rank(ascending=False, method="first").astype(np.float32)
    rank = f["combo_rank"].to_numpy()
    second_best = pd.Series(f["combo"].to_numpy()[rank == 2], index=grp[rank == 2]).reindex(grp)
    f["combo_top2_gap"] = (g["combo"].transform("max").to_numpy() - second_best.to_numpy()).astype(np.float32)
    f["n_strong"] = (f["combo"] >= 1.6).groupby(grp).transform("sum").astype(np.float32)
    f["name_gap"] = (f["name_tset"] - g["name_tset"].transform("max")).astype(np.float32)
    f["addr_gap"] = (f["addr_tset"] - g["addr_tset"].transform("max")).astype(np.float32)
    f["bscore_ratio"] = (f["bscore"] / g["bscore"].transform("max")).astype(np.float32)
    return f


def pair_level_features(b):
    """Features of one (S1, candidate) pair on its own: similarities, numbers, record facts."""
    sn, sa = b["s1_name"].tolist(), b["s1_addr"].tolist()
    cn, ca = b["cand_name"].tolist(), b["cand_addr"].tolist()
    f = pd.DataFrame({
        "name_tset": _sim(sn, cn, fuzz.token_set_ratio),
        "name_tsort": _sim(sn, cn, fuzz.token_sort_ratio),
        "name_ratio": _sim(sn, cn, fuzz.ratio),
        "name_partial": _sim(sn, cn, fuzz.partial_ratio),
        "name_jw": _sim(sn, cn, JaroWinkler.normalized_similarity, 1.0),
        "addr_tset": _sim(sa, ca, fuzz.token_set_ratio),
        "addr_tsort": _sim(sa, ca, fuzz.token_sort_ratio),
        "addr_ratio": _sim(sa, ca, fuzz.ratio),
        "addr_partial": _sim(sa, ca, fuzz.partial_ratio),
    })
    snc, cnc = b["s1_name_core"].tolist(), b["cand_name_core"].tolist()
    f["name_core_tset"] = _sim(snc, cnc, fuzz.token_set_ratio)
    f["name_core_ratio"] = _sim(snc, cnc, fuzz.ratio)
    f["addr_core_tset"] = _sim(b["s1_addr_core"].tolist(), b["cand_addr_core"].tolist(),
                               fuzz.token_set_ratio)

    # House numbers. -1 = unknown (a side has no number).
    ns, nc = _numbers(sa), _numbers(ca)
    num, first_eq, jacc, first_lev, best_lev, sub = (np.full(len(b), -1, np.float32) for _ in range(6))
    for i, (x, y) in enumerate(zip(ns, nc)):
        if x and y:
            sx, sy = set(x), set(y)
            inter = len(sx & sy)
            num[i] = 1.0 if inter else 0.0
            first_eq[i] = 1.0 if x[0] == y[0] else 0.0
            jacc[i] = inter / len(sx | sy)
            first_lev[i] = Levenshtein.normalized_similarity(x[0], y[0])   # 405 vs 40 -> high
            # best match between any two of the first 3 numbers; is one inside the other (123 / 7123)?
            best_lev[i] = max(Levenshtein.normalized_similarity(u, v) for u in x[:3] for v in y[:3])
            sub[i] = float(any(len(u) >= 2 and len(v) >= 2 and (u in v or v in u)
                               for u in x[:3] for v in y[:3]))
    f["num"], f["num_first_eq"], f["num_jacc"], f["num_first_lev"] = num, first_eq, jacc, first_lev
    f["num_best_lev"], f["num_sub"] = best_lev, sub
    f["n_num_s1"] = np.array([len(x) for x in ns], np.float32)
    f["n_num_cand"] = np.array([len(x) for x in nc], np.float32)

    # Record-level facts.
    f["indic"] = b["cand_name"].str.contains(INDIC).to_numpy()
    # was the candidate name in Indian script before translation? (same as indic if not translated)
    f["indic_src"] = (b["cand_indic_src"].to_numpy(np.float32) if "cand_indic_src" in b
                      else f["indic"].to_numpy(np.float32))
    f["addr_empty"] = (b["cand_addr"] == "").to_numpy()
    f["cand_s3"] = (b["cand"].str[:2] == "S3").to_numpy(np.float32)
    f["len_name_s1"] = b["s1_name"].str.len().to_numpy(np.float32)
    f["len_name_cand"] = b["cand_name"].str.len().to_numpy(np.float32)
    f["len_addr_s1"] = b["s1_addr"].str.len().to_numpy(np.float32)
    f["len_addr_cand"] = b["cand_addr"].str.len().to_numpy(np.float32)
    f["bscore"] = b["score"].to_numpy(np.float32)
    # how many S1 businesses share this S1's address / street (s1_context.py); NaN if not provided
    for col in ["s1_addr_share", "s1_street_share", "s1_name_share", "cand_name_s1_count",
                "from_fwd", "rev_rank", "rev_score_ratio", "rev_n_close"]:
        f[col] = b[col].to_numpy(np.float32) if col in b else np.float32(np.nan)
    f["brank"] = b["rank"].to_numpy(np.float32)
    return f


# ---------------------------------------------------------------------------------------------
# v6: character 3-gram similarity, for the final stage (survivors of the filter) only.
# Each text becomes a vector of its 3-letter chunks ("mil", "ill", "lli" ...); similarity is the
# cosine between the two vectors. Robust to typos, glued words ("hermantotal") and word order.
# Hashing (no vocabulary to learn or store) keeps it stateless: identical on train and test.
# ---------------------------------------------------------------------------------------------
CHAR_FEATURES = ["name_c3", "addr_c3", "name_core_c3"]
_HASHER = None


def _char_cos(a, b):
    global _HASHER
    if _HASHER is None:
        from sklearn.feature_extraction.text import HashingVectorizer
        _HASHER = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**20,
                                    alternate_sign=False, norm="l2")
    A, B = _HASHER.transform(a), _HASHER.transform(b)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def char_features(s1_name, cand_name, s1_addr, cand_addr, s1_name_core, cand_name_core):
    """All arguments: equal-length lists of cleaned strings. Returns a DataFrame of CHAR_FEATURES."""
    return pd.DataFrame({"name_c3": _char_cos(s1_name, cand_name),
                         "addr_c3": _char_cos(s1_addr, cand_addr),
                         "name_core_c3": _char_cos(s1_name_core, cand_name_core)})
