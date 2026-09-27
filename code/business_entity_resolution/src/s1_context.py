"""
Context about each S1 record that comes from the WHOLE Source 1 file, not from one pair.

Why: several different S1 businesses can share one address (a building, a school hosting several
associations). Measured: ~7% of S1 records in US/India train share their exact address with
another S1 record; in the France test data it is 14.9%, with up to 120 businesses on one address.
When an address is shared, an exact address match cannot tell those businesses apart and the name
must decide. The model can only learn that if it is told how shared the address is.

  addr_share    S1 records (same country) with the same address words, in any order
  street_share  S1 records (same country) with the same address words ignoring numbers (same street)

Always compute on the FULL S1 file of the split (train or test), then select the query records.
"""
import numpy as np
import pandas as pd


def add_s1_context(s1):
    """s1: DataFrame(entity_id, country, name_c, addr_c, ...). Adds addr_share and street_share."""
    words = s1["addr_c"].str.split()
    addr_key = words.map(lambda w: " ".join(sorted(set(w))))
    street_key = words.map(lambda w: " ".join(sorted({x for x in w if not x[0].isdigit()})))
    out = s1.copy()
    for col, key in [("addr_share", addr_key), ("street_share", street_key)]:
        n = pd.Series(np.ones(len(s1)), index=s1.index).groupby([s1["country"], key]).transform("size")
        out[col] = np.where(key == "", 0, n).astype(np.float32)   # empty address: unknown -> 0
    return out


# ---------------------------------------------------------------------------------------------
# Name uniqueness among S1 businesses.
#
# Why: when a candidate has no address (~1/3 of the remaining errors), only the name can decide.
# "Korry's Manufacturing" is probably the only S1 business with that core name, so an empty-address
# "Korry's Mánufacturing" most likely belongs to it; but if 40 S1 businesses are called "Vision ...",
# an empty-address "Vision" is a coin flip. The model can only weigh this if it is told the counts.
#
#   s1_name_share       S1 records (same country) with the same core name as this S1 record
#   cand_name_s1_count  S1 records (same country) whose core name equals the CANDIDATE's core name
#                       (0 = no S1 business has exactly that name)
# Core name = name words minus words frequent among that country's S1 names (llc, private, sarl ...),
# sorted. Learned per country from the S1 file itself: no hand-made lists, works for France.
# ---------------------------------------------------------------------------------------------
from collections import Counter

NAME_GENERIC_SHARE = 0.002


def _core(name, generic):
    return " ".join(sorted({w for w in name.split() if w not in generic}))


class NameIndex:
    """Per country: generic name words and counts of core names, from the FULL S1 file."""

    def __init__(self, s1):
        self.generic, self.counts = {}, {}
        for country, g in s1.groupby("country"):
            df = Counter(w for t in g["name_c"] for w in set(t.split()))
            thr = max(30, NAME_GENERIC_SHARE * len(g))
            self.generic[country] = {w for w, c in df.items() if c >= thr}
            self.counts[country] = Counter(_core(t, self.generic[country]) for t in g["name_c"])

    def count(self, names, countries):
        """How many S1 records (of that country) have the same core name as each given name."""
        out = np.zeros(len(names), np.float32)
        for i, (t, c) in enumerate(zip(names, countries)):
            if c in self.counts:
                core = _core(t, self.generic[c])
                out[i] = self.counts[c].get(core, 0) if core else np.nan
            else:
                out[i] = np.nan
        return out
