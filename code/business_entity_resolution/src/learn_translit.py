"""
Learn an Indian-script -> Latin word dictionary from TRAINING-split ground-truth pairs.

Why: ~24% of India S2 names and ~13% of India S3 names are written in Indian scripts, while S1
names never are. They are phonetic spellings ("ड्रीम बिजनेस प्राइवेट लिमिटेड" = "dream business
private limited"), so string similarity with the S1 name is zero. The training pairs are
ready-made examples: when a matched record's name has the same number of words as the S1 name,
word i usually spells word i. Counting these alignments over ~100k pairs gives a dictionary.

Leakage: only entities of the TRAIN split are used, so val/holdout scores stay honest.
No external data: everything comes from the provided training files.

Run from the project root:
    .venv/bin/python code/business_entity_resolution/src/learn_translit.py
Output: src/model/translit.json  {indian_word: latin_word}
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from metrics import parse_ids
from normalize import clean_name

TRANSLIT_FILE = Path(__file__).with_name("model") / "translit.json"
INDIC = r"[ऀ-෿]"
MIN_COUNT = 3      # an alignment must be seen at least this often
MIN_SHARE = 0.5    # ... and be the spelling in at least half of that word's alignments


def is_indic(w):
    return any("ऀ" <= ch <= "෿" for ch in w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache")
    cache = ap.parse_args().cache_dir

    split = pd.read_parquet(f"{cache}/split.parquet")
    train_ids = set(split.loc[split["split"] == "train", "source1_entity_id"])
    gt = pd.read_parquet(f"{cache}/train_ground_truth.parquet")
    gt = gt[gt["source1_entity_id"].isin(train_ids)]
    pairs = gt.assign(cand=gt["matched_entity_ids"].map(parse_ids)).explode("cand").dropna(subset=["cand"])

    s1 = pd.read_parquet(f"{cache}/train_source1.parquet", columns=["entity_id", "business_name"])
    s1 = s1[s1["entity_id"].isin(train_ids)]
    pool = pd.concat([pd.read_parquet(f"{cache}/train_source{i}.parquet",
                                      columns=["entity_id", "business_name"]) for i in (2, 3)])
    pool = pool[pool["entity_id"].isin(set(pairs["cand"])) & pool["business_name"].str.contains(INDIC)]
    s1_name = dict(zip(s1["entity_id"], clean_name(s1["business_name"])))
    c_name = dict(zip(pool["entity_id"], clean_name(pool["business_name"])))
    pairs = pairs[pairs["cand"].isin(c_name)]
    print(f"{len(pairs):,} training pairs with an Indian-script candidate name")

    counts = defaultdict(Counter)
    aligned = 0
    for s, c in zip(pairs["source1_entity_id"], pairs["cand"]):
        lw, cw = s1_name[s].split(), c_name[c].split()
        if len(lw) != len(cw):
            continue
        aligned += 1
        for a, b in zip(cw, lw):
            if is_indic(a) and not is_indic(b):
                counts[a][b] += 1
    mapping = {}
    for w, c in counts.items():
        best, n = c.most_common(1)[0]
        if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE:
            mapping[w] = best
    TRANSLIT_FILE.parent.mkdir(exist_ok=True)
    TRANSLIT_FILE.write_text(json.dumps(mapping, ensure_ascii=False, sort_keys=True))
    print(f"{aligned:,} same-length pairs aligned | {len(counts):,} Indian-script words seen | "
          f"{len(mapping):,} kept in the dictionary -> {TRANSLIT_FILE}")
    top = sorted(mapping.items(), key=lambda kv: -sum(counts[kv[0]].values()))[:15]
    print("most frequent:", ", ".join(f"{a}={b}" for a, b in top))


if __name__ == "__main__":
    main()
