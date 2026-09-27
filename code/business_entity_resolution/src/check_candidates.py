"""
Memory-light check of candidate_pairs.tsv, applying the same rules as utils/validate_submission.py.

Why: the official validator keeps every candidate list in memory (86M IDs -> ~10 GB), which does
not fit on an 8 GB laptop. Its own docs say to drop --candidate in that case; this script checks
the candidate file line by line instead, holding only the S1 ids and the (small) match lists.

    .venv/bin/python code/business_entity_resolution/src/check_candidates.py \
        --candidate output/candidate_pairs.tsv --matching output/matching_results.tsv --test-dir dataset/test
Rules checked: exact header; one row per test S1 (none missing, none extra, no duplicates);
no repeated ID inside a list; only S2-/S3- IDs; with --check-ids every ID exists in test S2/S3;
every matched ID is among that S1's candidates.
"""
import argparse
import sys

HEADER = "source1_entity_id\tcandidate_entity_ids"


def first_col(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--check-ids", action="store_true")
    args = ap.parse_args()

    required = first_col(f"{args.test_dir}/test_source1.tsv")
    valid = None
    if args.check_ids:
        valid = first_col(f"{args.test_dir}/test_source2.tsv") | first_col(f"{args.test_dir}/test_source3.tsv")
    matches = {}
    with open(args.matching, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            matches[s1] = set(rest.split(",")) if rest else set()

    problems = {k: [] for k in ["dup_row", "dup_id", "prefix", "unknown", "not_subset", "malformed"]}
    seen, n_rows, n_empty, n_ids = set(), 0, 0, 0
    with open(args.candidate, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        if header != HEADER:
            print(f"FAIL: header is {header!r}, expected {HEADER!r}")
            return 1
        for line in f:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                problems["malformed"].append(s1)
                continue
            n_rows += 1
            if s1 in seen:
                problems["dup_row"].append(s1)
            seen.add(s1)
            ids = rest.split(",") if rest else []
            n_empty += not ids
            n_ids += len(ids)
            ids_set = set(ids)
            if len(ids) != len(ids_set):
                problems["dup_id"].append(s1)
            if any(not i.startswith(("S2-", "S3-")) for i in ids):
                problems["prefix"].append(s1)
            if valid is not None and any(i not in valid for i in ids):
                problems["unknown"].append(s1)
            if matches.get(s1, set()) - ids_set:
                problems["not_subset"].append(s1)
    problems["missing"] = list(required - seen)
    problems["extra"] = list(seen - required)

    print(f"{args.candidate}: {n_rows:,} rows ({n_empty:,} empty), {n_ids:,} candidate IDs, "
          f"{n_ids / max(n_rows, 1):.1f} per S1")
    bad = {k: v for k, v in problems.items() if v}
    for k, v in bad.items():
        print(f"  PROBLEM {k}: {len(v)} rows, e.g. {v[:3]}")
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
