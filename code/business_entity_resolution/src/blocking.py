"""
Blocking v2: find a short list of plausible S2/S3 candidates for each S1 record.

How it works
------------
1. Word rarity. Per country, count how many pool (S2+S3) records contain each word
   (its "document frequency", df). "street" or "limited" have a huge df; "milliken" a tiny one.
   This word list is also the vocabulary: each word gets an integer id.
2. Keys. Every record emits keys built from its RAREST words, plus its address numbers:
     N1  one name word                                 nexify
     N2  two name words, sorted                        golden+pilates  (word order doesn't matter)
     NP  first 6 letters of the name, spaces removed   "hermantotal com" and "herman total" -> herman
     A   address number + address word                 74+milliken, 74+bridgewater
     A2  two address words, sorted                     cedar+cortland  (addresses with no number)
   S1 records use more words than pool records: S1 addresses are often long while S2/S3 copies
   are short, so whichever words a short copy kept are still among the S1 record's keys.
   A key is a 40-bit hash of (kind, integer, integer): pure numpy, no Python strings.
3. Index. Per country, every pool key is stored as one 64-bit number (key in the high 40 bits,
   pool record number in the low 24 bits) in one sorted array. Finding all pool records with a key
   is a binary search, so S1 records are processed in small batches with flat memory use.
4. Scoring. score(S1, candidate) = sum over shared keys of log(N / block size): sharing a rare key
   counts more. Keys shared by more than `cap` pool records are ignored. Keep the top k per S1.

Countries are processed one at a time and compared as plain strings (France etc. just work).
Records must already be cleaned (normalize.clean_name / clean_address).
"""
import numpy as np
import pandas as pd

K_NAME = {"query": 6, "pool": 4}   # rarest name words used per record
K_ADDR = {"query": 8, "pool": 4}   # rarest address words used per record
MAX_NUMBERS = 3                    # address numbers used per record
PREFIX_LEN = 6
KINDS = {"N1": 1, "N2": 2, "NP": 3, "A": 4, "A2": 5, "NC": 6}

REC_BITS = 24                      # up to 16.7M pool records per country
SHIFT = np.uint64(REC_BITS)
REC_MASK = np.uint64((1 << REC_BITS) - 1)
_GOLDEN = 0x9E3779B97F4A7C15


def _mix(x):
    """splitmix64 finaliser: scrambles uint64 values so similar inputs give unrelated outputs."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _key(kind, a, b):
    """40-bit key from a kind and two integer arrays."""
    salt = np.uint64((KINDS[kind] * _GOLDEN) % 2**64)
    a = np.asarray(a).astype(np.uint64)
    b = np.asarray(b).astype(np.uint64)
    return _mix(_mix(a ^ salt) ^ b) >> SHIFT


class Vocab:
    """Words of one country's pool, with their document frequency (df) and integer ids."""

    def __init__(self, texts, chunk=500_000):
        counts = [_words(texts.iloc[i:i + chunk])["word"].value_counts()
                  for i in range(0, len(texts), chunk)]
        df = pd.concat(counts).groupby(level=0).sum()
        self.index = pd.Index(df.index)
        self.df = df.to_numpy()

    def rarest(self, words, k):
        """words: DataFrame(rec, word). Keep each record's k rarest known words -> (rec, id)."""
        ids = self.index.get_indexer(words["word"])
        known = ids >= 0                               # unseen words can never match: dropped
        w = pd.DataFrame({"rec": words["rec"].to_numpy()[known], "id": ids[known],
                          "df": self.df[ids[known]]})
        return w.sort_values(["rec", "df"], kind="stable").groupby("rec").head(k)[["rec", "id"]]


def _words(s):
    """Series of cleaned strings -> DataFrame(rec, word), one row per distinct word, in order."""
    w = s.reset_index(drop=True).str.split().explode().dropna()
    return pd.DataFrame({"rec": w.index.to_numpy(np.int64), "word": w.to_numpy()}).drop_duplicates()


def _pairs(w):
    p = w.merge(w, on="rec")
    return p[p["id_x"] < p["id_y"]]


def record_keys(name_c, addr_c, name_vocab, addr_vocab, side, generic_name=None):
    """Keys for a batch of records. Returns (rec, key) arrays; rec = row position in the batch.

    generic_name: if given, also emit NC = the whole core name (generic words removed, sorted),
    e.g. "mumbai services ltd" and "mumbai services-ltd" -> "mumbai services". Finds records whose
    only usable information is the name (empty address), where single words are too common.
    """
    recs, keys = [], []

    def add(kind, rec, a, b=0):
        recs.append(np.asarray(rec, dtype=np.int64))
        keys.append(_key(kind, a, np.broadcast_to(b, len(recs[-1]))))

    # Name keys
    nw = _words(name_c)
    nw = name_vocab.rarest(nw[nw["word"].str.len() >= 2], K_NAME[side])
    add("N1", nw["rec"], nw["id"])
    p = _pairs(nw)
    add("N2", p["rec"], p["id_x"], p["id_y"])
    pre = name_c.reset_index(drop=True).str.replace(" ", "", regex=False).str.slice(0, PREFIX_LEN)
    pre = pre[pre.str.len() >= 4]
    add("NP", pre.index, pd.util.hash_array(pre.to_numpy(dtype=object)))
    if generic_name is not None:
        core = pd.Series([" ".join(sorted({w for w in t.split() if w not in generic_name}))
                          for t in name_c], dtype=object)
        core = core[core.str.len() >= 3]
        add("NC", core.index, pd.util.hash_array(core.to_numpy(dtype=object)))

    # Address keys
    aw = _words(addr_c)
    lead = aw["word"].str.extract(r"^(\d+)", expand=False)            # "1151a" -> "1151"
    num = pd.to_numeric(lead.str.slice(-15), errors="coerce")          # "003810" -> 3810
    nums = pd.DataFrame({"rec": aw["rec"], "num": num})
    nums = nums[nums["num"] > 0].drop_duplicates().groupby("rec").head(MAX_NUMBERS)
    alpha = aw[lead.isna() & (aw["word"].str.len() >= 3)]
    alpha = addr_vocab.rarest(alpha, K_ADDR[side])
    combo = nums.merge(alpha, on="rec")                                # every number x rare word
    add("A", combo["rec"], combo["num"], combo["id"])
    p = _pairs(alpha)
    add("A2", p["rec"], p["id_x"], p["id_y"])

    out = pd.DataFrame({"rec": np.concatenate(recs), "key": np.concatenate(keys)}).drop_duplicates()
    return out["rec"].to_numpy(), out["key"].to_numpy()


def build_index(pool, name_vocab, addr_vocab, chunk=250_000, generic_name=None, pool_side="pool"):
    """Sorted array of (key << 24 | pool record number) for every key of every pool record."""
    assert len(pool) < (1 << REC_BITS), "too many pool records for one country"
    packed = []
    for i in range(0, len(pool), chunk):
        part = pool.iloc[i:i + chunk]
        rec, key = record_keys(part["name_c"], part["addr_c"], name_vocab, addr_vocab, pool_side,
                               generic_name)
        packed.append((key << SHIFT) | (rec + i).astype(np.uint64))
    packed = np.concatenate(packed)
    packed.sort()
    return packed


def score_queries(qrec, qkey, packed, n_pool, cap):
    """All (query, pool record, score) pairs sharing at least one usable key."""
    uq, inv = np.unique(qkey, return_inverse=True)
    lo = np.searchsorted(packed, uq << SHIFT, "left")
    hi = np.searchsorted(packed, (uq << SHIFT) | REC_MASK, "right")
    size = hi - lo                                   # block size: pool records with this key
    use = (size > 0) & (size <= cap)
    w = np.zeros(len(uq))
    w[use] = np.log(n_pool / size[use])              # rarity weight

    # Expand each query key into the pool records of its block (compact dtypes: this is the
    # biggest temporary array in blocking, and its size grows with the query batch).
    n = np.where(use, size, 0)[inv]
    wt = np.repeat(w.astype(np.float32)[inv], n)
    idx = np.repeat(lo[inv] - (np.cumsum(n) - n), n) + np.arange(n.sum())
    p = (packed[idx] & REC_MASK).astype(np.int64)
    del idx
    code = (np.repeat(qrec.astype(np.int64), n) << REC_BITS) | p   # (query, pool record) as one int
    s = pd.Series(wt).groupby(code).sum()
    return pd.DataFrame({"q": s.index.to_numpy() >> REC_BITS,
                         "p": s.index.to_numpy() & ((1 << REC_BITS) - 1),
                         "score": s.to_numpy(np.float64)})


GENERIC_SHARE = 0.002   # a word in >= 0.2% of a country's pool records counts as generic


def generic_words(vocab, n_pool):
    """Very frequent words of one country's pool (llc, private, limited, sarl, street, road ...)."""
    return set(vocab.index[vocab.df >= max(30, GENERIC_SHARE * n_pool)])


def core_text(texts, generic):
    """Remove generic words: 'rkk products private limited' -> 'rkk products'."""
    return pd.array([" ".join(w for w in t.split() if w not in generic) for t in texts], dtype="str")


# ---------------------------------------------------------------------------------------------
# Bidirectional blocking (v7). Forward: for each S1, the best records among ~10M. Reverse: for each
# record, the best S1 businesses among the FULL S1 file (~2M, deduplicated). A true match lost among
# 10M records often ranks at the top from the other side: on val, reverse top-10 recovers 49% of the
# pairs forward top-50 misses (pair recall 0.953 -> 0.976).
# Kept as compact integer arrays (~0.6 GB per country), not strings.
# ---------------------------------------------------------------------------------------------
REV_K_BLOCK = 10


def reverse_table(pc, s1c, qc, rev_k=REV_K_BLOCK, cap=300, chunk=20_000, log=print):
    """Search every pool record of one country against that country's FULL S1 file.

    pc: pool records, s1c: full S1 file, qc: the queried S1 records (subset of s1c, or all of it).
    Returns lookup arrays keyed by (query position << 32 | pool position), sorted:
      keys, rank (1 = the record's best S1), ratio (score / record's best score),
      and n_close[pool position] = S1 records within 90% of that record's best score.
    """
    nv, av = Vocab(s1c["name_c"]), Vocab(s1c["addr_c"])
    packed = build_index(s1c, nv, av, pool_side="query")
    recs, s1s, scores, ranks = [], [], [], []
    for j in range(0, len(pc), chunk):
        part = pc.iloc[j:j + chunk]
        rec, key = record_keys(part["name_c"], part["addr_c"], nv, av, "query")
        sc = score_queries(rec + j, key, packed, len(s1c), cap)
        sc = sc.sort_values(["q", "score", "p"], ascending=[True, False, True])
        sc["rank"] = sc.groupby("q").cumcount() + 1
        sc = sc[sc["rank"] <= rev_k]
        recs.append(sc["q"].to_numpy(np.int32))      # 32-bit: pool < 16.7M, S1 < 2^31
        s1s.append(sc["p"].to_numpy(np.int32))
        scores.append(sc["score"].to_numpy(np.float32))
        ranks.append(sc["rank"].to_numpy(np.int8))
    del packed
    rec, s1pos = np.concatenate(recs), np.concatenate(s1s)
    score, rank = np.concatenate(scores), np.concatenate(ranks)
    del recs, s1s, scores, ranks
    best = np.zeros(len(pc), np.float32)
    best[rec[rank == 1]] = score[rank == 1]
    b = best[rec]
    ratio = np.where(b > 0, score / np.where(b > 0, b, 1), 1.0).astype(np.float32)
    n_close = np.bincount(rec, weights=(score >= 0.9 * b), minlength=len(pc)).astype(np.float32)
    del b, score
    qpos_of = (pd.Series(np.arange(len(qc)), index=qc["entity_id"].to_numpy())
                 .reindex(s1c["entity_id"].to_numpy()).to_numpy())       # NaN = S1 not queried
    qpos = qpos_of[s1pos]
    del s1pos
    keep = ~np.isnan(qpos)
    keys = (qpos[keep].astype(np.int64) << 32) | rec[keep].astype(np.int64)
    n_links = len(rec)
    del qpos, rec
    order = np.argsort(keys, kind="stable")
    out = {"keys": keys[order], "rank": rank[keep][order], "ratio": ratio[keep][order], "n_close": n_close}
    log(f"    reverse: {len(pc):,} records searched against {len(s1c):,} S1 -> {n_links:,} links")
    return out


def _rev_lookup(rev, q, p):
    """Reverse rank / score ratio for pairs (query position q, pool position p)."""
    key = (np.asarray(q, np.int64) << 32) | np.asarray(p, np.int64)
    if len(rev["keys"]) == 0:
        return np.full(len(key), REV_K_BLOCK + 1, np.float32), np.zeros(len(key), np.float32)
    idx = np.searchsorted(rev["keys"], key)
    idx_c = np.minimum(idx, len(rev["keys"]) - 1)
    found = (idx < len(rev["keys"])) & (rev["keys"][idx_c] == key)
    rank = np.where(found, rev["rank"][idx_c], REV_K_BLOCK + 1).astype(np.float32)
    ratio = np.where(found, rev["ratio"][idx_c], 0).astype(np.float32)
    return rank, ratio


REV_CAP_PER_S1 = 20   # max reverse-found candidates per S1: stops "hub" S1s (a word rare among S1
                      # but common among records, e.g. "com") from collecting 100k+ junk candidates;
                      # keeps 99.6% of the true pairs the reverse search finds (val/holdout/train)


def cap_reverse(df, cap):
    """Per S1 (column q), keep the `cap` strongest reverse-only candidates:
    best reverse rank, then best score ratio, then candidate id. Same rule in blocking and on pairs."""
    df = df.sort_values(["q", "rev_rank", "rev_score_ratio", "cid"], ascending=[True, True, False, True],
                        kind="stable")
    return df[df.groupby("q").cumcount() < cap]


def _add_reverse(s, rev, i, n, k, rev_add, pool_ids, cap=REV_CAP_PER_S1):
    """Add each record's top-`rev_add` reverse S1s (queries i..i+n) that forward blocking missed."""
    lo = np.searchsorted(rev["keys"], np.int64(i) << 32)
    hi = np.searchsorted(rev["keys"], np.int64(i + n) << 32)
    keys, rank, ratio = rev["keys"][lo:hi], rev["rank"][lo:hi], rev["ratio"][lo:hi]
    m = rank <= rev_add
    keys, rank, ratio = keys[m], rank[m], ratio[m]
    fwd = (s["q"].to_numpy(np.int64) << 32) | s["p"].to_numpy(np.int64)
    m = ~np.isin(keys, fwd)
    keys, rank, ratio = keys[m], rank[m], ratio[m]
    ext = pd.DataFrame({"q": keys >> 32, "p": keys & 0xFFFFFFFF, "rev_rank": rank.astype(np.float32),
                        "rev_score_ratio": ratio, "cid": pool_ids[keys & 0xFFFFFFFF]})
    ext = cap_reverse(ext, cap)
    extra = pd.DataFrame({"q": ext["q"].to_numpy(), "p": ext["p"].to_numpy(), "score": 0.0,
                          "rank": k + 1, "from_fwd": 0})
    s = pd.concat([s.assign(from_fwd=1), extra], ignore_index=True)
    return s.sort_values(["q", "from_fwd", "rank"], ascending=[True, False, True], kind="stable")


def iter_candidates(query, load_pool, cap=300, k=100, q_chunk=5_000, with_text=False, q_extra=(),
                    core_name_key=False, pool_side="pool", reverse_s1=None, rev_add=3,
                    rev_cap=REV_CAP_PER_S1, log=print):
    """Yield top-k candidates per S1 record, one batch at a time.

    query:     DataFrame(entity_id, country, name_c, addr_c) of S1 records.
    load_pool: function(country) -> DataFrame(entity_id, name_c, addr_c) of that country's S2+S3.
    Yields DataFrame(s1, cand, score, rank) with entity ids; with_text=True also adds the
    cleaned s1_name, s1_addr, cand_name, cand_addr of every pair, and their *_core versions
    (generic words removed), for the matching step. q_extra: query columns to copy as s1_<col>.
    reverse_s1: the FULL S1 file -> bidirectional blocking: each record's top-`rev_add` S1s are added
    as candidates (score 0, rank k+1, from_fwd=0), and every pair gets rev_rank, rev_score_ratio,
    rev_n_close. Without it, behaviour is exactly as before.
    """
    for country in query["country"].unique():        # whatever countries the data contains
        qc = query[query["country"] == country].reset_index(drop=True)
        pc = load_pool(country).reset_index(drop=True)
        if len(pc) == 0:
            continue
        name_vocab, addr_vocab = Vocab(pc["name_c"]), Vocab(pc["addr_c"])
        gen_nc = generic_words(name_vocab, len(pc)) if core_name_key else None
        packed = build_index(pc, name_vocab, addr_vocab, generic_name=gen_nc, pool_side=pool_side)
        pool_ids = pc["entity_id"].to_numpy()
        n_pool = len(pc)
        pool_text = None
        if with_text:
            gen_n, gen_a = generic_words(name_vocab, n_pool), generic_words(addr_vocab, n_pool)
            pool_text = (pc["name_c"].array, pc["addr_c"].array,
                         core_text(pc["name_c"], gen_n), core_text(pc["addr_c"], gen_a),
                         pc["indic_src"].to_numpy() if "indic_src" in pc else None)
            log(f"  {country}: {len(gen_n)} generic name words, {len(gen_a)} generic address words")
        rev = None
        if reverse_s1 is not None:
            s1c = reverse_s1[reverse_s1["country"] == country].reset_index(drop=True)
            if len(s1c):
                rev = reverse_table(pc, s1c, qc, log=log)
        del pc
        log(f"  {country}: {len(qc):,} S1 | {n_pool:,} pool | index {len(packed):,} keys "
            f"({packed.nbytes / 1e9:.2f} GB)")
        for i in range(0, len(qc), q_chunk):
            part = qc.iloc[i:i + q_chunk]
            rec, key = record_keys(part["name_c"], part["addr_c"], name_vocab, addr_vocab, "query",
                                   gen_nc)
            s = score_queries(rec + i, key, packed, n_pool, cap)
            s = s.sort_values(["q", "score", "p"], ascending=[True, False, True])
            s["rank"] = s.groupby("q").cumcount() + 1
            s = s[s["rank"] <= k]
            if rev is not None:
                s = _add_reverse(s, rev, i, len(part), k, rev_add, pool_ids, rev_cap)
            q, p = s["q"].to_numpy(), s["p"].to_numpy()
            out = pd.DataFrame({"s1": qc["entity_id"].to_numpy()[q], "cand": pool_ids[p],
                                "score": s["score"].to_numpy(), "rank": s["rank"].to_numpy()})
            if rev is not None:
                out["from_fwd"] = s["from_fwd"].to_numpy(np.float32)
                out["rev_rank"], out["rev_score_ratio"] = _rev_lookup(rev, q, p)
                out["rev_n_close"] = rev["n_close"][p]
            if with_text:  # .take keeps the compact Arrow strings (no Python string objects)
                q_local = q - i
                out["s1_name"] = part["name_c"].array.take(q_local)
                out["s1_addr"] = part["addr_c"].array.take(q_local)
                out["s1_name_core"] = core_text(part["name_c"], gen_n).take(q_local)
                out["s1_addr_core"] = core_text(part["addr_c"], gen_a).take(q_local)
                out["cand_name"] = pool_text[0].take(p)
                out["cand_addr"] = pool_text[1].take(p)
                out["cand_name_core"] = pool_text[2].take(p)
                out["cand_addr_core"] = pool_text[3].take(p)
                if pool_text[4] is not None:
                    out["cand_indic_src"] = pool_text[4][p]
                for col in q_extra:
                    out[f"s1_{col}"] = part[col].to_numpy()[q_local]
            yield out
        del packed, name_vocab, addr_vocab, pool_ids, pool_text, rev


def candidates(query, load_pool, **kw):
    """All candidates at once (fine for val-sized queries; stream with iter_candidates for test)."""
    return pd.concat(iter_candidates(query, load_pool, **kw), ignore_index=True)
