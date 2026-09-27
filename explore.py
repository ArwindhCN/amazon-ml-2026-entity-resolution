"""
explore.py
Run this from inside the student_resource/ folder:
    python3 explore.py

It does two things:
  1. Prints basic stats (row counts, countries, how many S1 entities have no match, etc.)
  2. Writes small sample files into samples/ that are small enough to upload to Claude.
"""
import csv
import os
import pandas as pd

TRAIN = "dataset/train"
TEST = "dataset/test"
OUT = "samples"
os.makedirs(OUT, exist_ok=True)


def load(path):
    # sep="\t"              -> files are tab-separated
    # dtype=str             -> read everything as text (IDs like S1-00001 stay text)
    # keep_default_na=False -> empty cells stay "" instead of becoming NaN
    # quoting=QUOTE_NONE    -> a " inside an address is treated as a normal character
    return pd.read_csv(path, sep="\t", dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE)


# ---------- 1. Test files: count rows + countries, grab France examples ----------
print("=== TEST files ===")
france_rows = []
for s in [1, 2, 3]:
    df = load(f"{TEST}/test_source{s}.tsv")
    print(f"test_source{s}: {len(df):,} rows | columns: {list(df.columns)}")
    print("   countries:", df["country"].value_counts().head(10).to_dict())
    fr = df[df["country"].str.lower().str.contains("fr")]
    france_rows.append(fr.sample(min(20, len(fr)), random_state=0))
    del df  # free memory before loading the next big file

pd.concat(france_rows).to_csv(f"{OUT}/france_test_sample.tsv", sep="\t", index=False)

# ---------- 2. Train files ----------
print("\n=== TRAIN files ===")
train = {}
for s in [1, 2, 3]:
    df = load(f"{TRAIN}/train_source{s}.tsv")
    print(f"train_source{s}: {len(df):,} rows")
    print("   countries:", df["country"].value_counts().head(10).to_dict())
    train[s] = df

# ---------- 3. Ground truth stats ----------
gt = load(f"{TRAIN}/train_ground_truth.tsv")
gt["matches"] = gt["matched_entity_ids"].apply(lambda x: [m for m in x.split(",") if m])
gt["n"] = gt["matches"].apply(len)

print("\n=== GROUND TRUTH ===")
print(f"S1 entities in ground truth: {len(gt):,}")
print(f"Singletons (no match): {(gt['n'] == 0).mean():.1%}")
print("Matches per entity (count -> how many entities):",
      gt["n"].value_counts().sort_index().to_dict())

all_matches = gt["matches"].explode().dropna()
print("Matched IDs from S2:", all_matches.str.startswith("S2-").sum(),
      "| from S3:", all_matches.str.startswith("S3-").sum())

# Does any S2/S3 record belong to more than one S1 entity?
repeat = (all_matches.value_counts() > 1).sum()
print("S2/S3 records matched to more than one S1 entity:", repeat)

# ---------- 4. Readable examples: 100 matched + 50 singletons ----------
has_match = gt[gt["n"] > 0]
no_match = gt[gt["n"] == 0]
sample_gt = pd.concat([
    has_match.sample(min(100, len(has_match)), random_state=0),
    no_match.sample(min(50, len(no_match)), random_state=0),
])

# Build a small lookup table with only the records we need (saves memory)
needed = set(sample_gt["source1_entity_id"]) | set(sample_gt["matches"].explode().dropna())
lookup = {}
for df in train.values():
    part = df[df["entity_id"].isin(needed)]
    for _, r in part.iterrows():
        lookup[r["entity_id"]] = f"{r['business_name']} | {r['business_address']} | {r['country']}"

with open(f"{OUT}/train_examples.txt", "w", encoding="utf-8") as f:
    for _, row in sample_gt.iterrows():
        sid = row["source1_entity_id"]
        f.write(f"{sid} | {lookup.get(sid, '??')}\n")
        if not row["matches"]:
            f.write("    (no matches)\n")
        for m in row["matches"]:
            f.write(f"    -> {m} | {lookup.get(m, '??')}\n")
        f.write("\n")

print(f"\nDone. Upload these to Claude:\n  {OUT}/train_examples.txt\n  {OUT}/france_test_sample.tsv")
