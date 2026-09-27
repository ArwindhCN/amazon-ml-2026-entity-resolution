"""
Reverse competition (idea from Sridharan's notebook, built so training matches test).

Question: for a candidate RECORD (S2/S3), is this S1 its best fit, or does another S1 fit it better?
Each S2/S3 record belongs to at most one S1, so "another S1 fits it clearly better" is strong
evidence against a match (shared addresses, common names, planted look-alikes).

How: search BACKWARDS with the same blocking machinery: candidate records are the queries, and
the pool is the FULL S1 file of the split (all 2.2M train S1, or all 1.73M test S1). Because the
full S1 file is used in both cases, the feature means the same thing in training and on test
(a naive version using only the queried S1s would see most owners missing during training).

Features for a pair (s1, record):
  rev_rank         rank of this S1 among the record's reverse candidates (1 = best fit);
                   REV_K + 1 if this S1 is not among them
  rev_score_ratio  this S1's reverse blocking score / the record's best reverse score (0 if absent)
  rev_n_close      how many S1 records score >= 90% of the record's best (ambiguity)
"""
import numpy as np
import pandas as pd

from blocking import Vocab, build_index, record_keys, score_queries

REV_K = 10
REV_CAP = 300
REV_FEATURES = ["rev_rank", "rev_score_ratio", "rev_n_close"]


class ReverseSearcher:
    """Index of the FULL S1 file, one country at a time (built once, queried in batches)."""

    def __init__(self, s1_all):
        self.s1 = {c: g.reset_index(drop=True) for c, g in s1_all.groupby("country")}
        self.country, self.index = None, None

    def _load(self, country):
        if country != self.country:
            self.index = None                      # keep one country in memory
            pc = self.s1[country]
            nv, av = Vocab(pc["name_c"]), Vocab(pc["addr_c"])
            self.index = (pc["entity_id"].to_numpy(), nv, av,
                          build_index(pc, nv, av, pool_side="query"), len(pc))
            self.country = country
        return self.index

    def query(self, records):
        """records: DataFrame(entity_id, country, name_c, addr_c) -> DataFrame(record, s1, score, rank)."""
        out = []
        for country, r in records.groupby("country"):
            if country not in self.s1:
                continue
            r = r.reset_index(drop=True)
            s1_ids, nv, av, packed, n = self._load(country)
            for i in range(0, len(r), 20_000):
                part = r.iloc[i:i + 20_000]
                rec, key = record_keys(part["name_c"], part["addr_c"], nv, av, "query")
                sc = score_queries(rec + i, key, packed, n, REV_CAP)
                sc = sc.sort_values(["q", "score", "p"], ascending=[True, False, True])
                sc["rank"] = sc.groupby("q").cumcount() + 1
                sc = sc[sc["rank"] <= REV_K]
                out.append(pd.DataFrame({"record": r["entity_id"].to_numpy()[sc["q"].to_numpy()],
                                         "s1": s1_ids[sc["p"].to_numpy()],
                                         "score": sc["score"].to_numpy(), "rank": sc["rank"].to_numpy()}))
        cols = ["record", "s1", "score", "rank"]
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def reverse_candidates(records, s1_all, log=lambda m: None):
    """One-off reverse search (training): records vs the full S1 file."""
    return ReverseSearcher(s1_all).query(records)


def reverse_features(s1_ids, cand_ids, rev):
    """Features for pairs (s1_ids[i], cand_ids[i]) from reverse candidates `rev`."""
    best = rev.groupby("record")["score"].transform("max")
    rev = rev.assign(ratio=(rev["score"] / best).astype(np.float32),
                     close=(rev["score"] >= 0.9 * best))
    n_close = rev.groupby("record")["close"].sum()
    look = rev.set_index(["record", "s1"])[["rank", "ratio"]]
    key = pd.MultiIndex.from_arrays([np.asarray(cand_ids), np.asarray(s1_ids)])
    hit = look.reindex(key)
    return pd.DataFrame({
        "rev_rank": hit["rank"].fillna(REV_K + 1).to_numpy(np.float32),
        "rev_score_ratio": hit["ratio"].fillna(0).to_numpy(np.float32),
        "rev_n_close": pd.Series(np.asarray(cand_ids)).map(n_close).fillna(0).to_numpy(np.float32),
    })
