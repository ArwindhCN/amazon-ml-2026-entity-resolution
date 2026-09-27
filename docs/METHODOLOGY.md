# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

**Team Name:** Runtime Terrors  
**Team Members:** Sridharan Kannan (team lead), Arwindh C N, Raghav V S  
**Submission Date:** 27 September 2026

> Final model: **v8** (v7 + fine-tuned transformer cross-encoder). This is the methodology write-up prepared for our final submission package.

---

## 1. Executive Summary
We resolve every Source 1 business against Sources 2 and 3 with a five-stage **cascade**:
1. **Bidirectional blocking:** rarity-weighted multi-key search keeps the top 50 of 10M records per
   S1. A *reverse* search of every record against the full S1 file then adds each record's best 3 S1
   businesses. Pair recall on validation rises from 95.3% to 97.1%.
2. **A learned candidate filter** cuts that to **~4 candidates per S1**, and these survivors are
   exactly `candidate_pairs.tsv`.
3. **A small transformer cross-encoder** (MiniLM, 22.7M parameters, Apache 2.0) fine-tuned on our
   training pairs reads both records of each surviving pair together and scores them.
4. **A LightGBM matcher** scores the survivors using string, house-number, *group-comparison* and
   *whole-file context* features, plus the transformer's score. The context features record how many
   S1 businesses share an address or a name.
5. **A decision layer** gives each S2/S3 record to at most one S1, then picks the subset of
   candidates that maximises *expected* F0.5 per entity.

Indian-script names are translated with a word dictionary learned from the training pairs.
Macro F0.5 on a held-out split of the training data: **0.979** (v7 without the transformer: 0.974,
public leaderboard 0.967). The rule baseline scored 0.799. No external data is used; the
only pretrained component is the Apache-2.0 MiniLM cross-encoder, fine-tuned on the training split only.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the training data (2.21M S1, 5.03M S2, 5.29M S3 records):
- **Few singletons:** only 5.6% of S1 entities have no match; on average they have 3.46 (max 11).
  Predicting nothing for everyone scores 0.056, so recall matters as well as precision.
- **One owner per record:** no S2/S3 record matches more than one S1 entity (verified on all 7.6M
  true pairs). This justifies a "competition" constraint at decision time.
- **Decoys:** ~1.34M S2 and ~1.34M S3 records match nothing. Singletons often have planted
  look-alikes with the same name and street but a different house number, or the same name and no address.
- **Name noise:** case, legal suffixes added/dropped/swapped, word reordering, character typos,
  look-alike digits (`N0rthern`, `5ecurity`), junk prefixes (`***`, `--`), honorifics, filler words
  (Services, Center), domain-style names (`rkkproducts.com`), trade-name wrappers ("X trading as Y",
  "f/k/a"), and some names replaced by an unrelated word.
- **Indian scripts:** 23.5% of India S2 and 13.2% of India S3 names are in Indian scripts
  (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam); S1 names never are. They are
  phonetic spellings of English words ("ड्रीम बिजनेस प्राइवेट लिमिटेड" = "dream business private
  limited"), drawn from a small vocabulary: 1,309 distinct words cover all training pairs.
- **Address noise:** abbreviations, state names vs codes, native-script state names, reordered
  components, `NULL`/`N/A` placeholders, empty addresses, and neighbouring city names. House numbers
  vary in format (`003810`, `##7601`, `1525 1/2`, `97-99`, `B-8-03`) and are sometimes changed even
  in true matches (`123` vs `7123`), so a number disagreement is strong but not absolute evidence.
- **Shared addresses:** several different S1 businesses can share one address (a building, or a
  school hosting several associations). This is ~7% of S1 records in train and **14.9% in France**,
  with up to 120 businesses on one French address. We found this from test-set diagnostics after a
  first leaderboard score (0.944) came in below validation (0.952): French records were claimed by
  several S1 entities 7× more often than US records.
- **France (test only, 15% of test S1):** legal forms SARL/SAS/EURL/SASU/SCI/EI, `Rue`/`R.`, `Bd`,
  `Av.`, `N°`, region vs département. Country is treated as an open set: it is only ever compared
  as a string (inside blocking keys), never used as a feature. Every generic-word list is learned
  per country from the data.

### 2.2 Solution Strategy
**Approach Type:** Blocking → learned candidate filter → classifier → per-entity decision (cascade)  
**Core Innovations:**
1. **Rarity-based blocking** over a compact sorted index. It scales to 1.7M × 10M records on an 8 GB laptop.
2. **A learned candidate filter** that cuts the candidate set **13× (50 → ~4 per S1) with no loss
   of F0.5**, trained out-of-fold so the final model sees realistic candidate sets.
3. **Group features:** each candidate is compared with the other candidates of the same S1, which
   exposes planted look-alikes.
4. **Whole-file context features:** how many S1 businesses share this address, street, or core name.
   These tell the model when an exact address or name match is *not* decisive.
5. **Bidirectional blocking with reverse competition.** For each record, which S1 businesses fit it
   best, searched against the *full* S1 file so that training and test see the same competition?
   These reverse candidates and features go to both the filter and the final model. A cap of 20
   reverse-found candidates per S1 prevents "hub" businesses: "COM Pacific Senior L.L.C." attracted
   142k domain-style records through the word "com".
6. **A fine-tuned transformer cross-encoder as one feature** of the final LightGBM, trained
   out-of-fold so the final model learns how much to trust it (+0.005 holdout F0.5).
7. **A learned Indian-script → Latin word dictionary** built from training pairs only.
8. **An expected-F0.5 decision** per entity, after the one-owner competition.

**Validation:** training S1 entities are split by entity (seed 42, stratified by country) into
train 300k / val 50k / holdout 50k. Leakage checks assert that no S1 or S2/S3 record is shared
between splits. Blocking always searches the full train S2+S3 pool (10.3M), so decoy density is
realistic. Val is used for tuning and early stopping; holdout is scored once per model. We
implemented macro F0.5 exactly as specified and checked it against the README example (0.714).

---

## 3. Candidate Generation (Blocking + learned filter)
- **Cleaning:** NFKD normalisation, accent stripping, lower-casing, punctuation → space (letters of
  every script kept), removal of `NULL`/`N/A`, look-alike digit repair inside words, and Indian-script
  words replaced by their learned Latin spelling.
- **Blocking keys.** Every key includes the country. Keys are built from each record's *rarest*
  words by pool document frequency. S1 uses 6 name / 8 address words and S2/S3 records use 4 / 4,
  because S1 addresses are long and S2/S3 copies are often truncated.
  - N1: one rare name word
  - N2: pair of rare name words (order-free)
  - NP: first 6 letters of the name with spaces removed (for `hermantotal.com`)
  - A: house number × rare address word
  - A2: pair of rare address words (order-free), for addresses without a number
- **Index:** per country, each key is a 40-bit hash packed with a 24-bit record number into one
  sorted uint64 array, so a lookup is a binary search. Keys shared by more than 300 pool records are
  ignored. A candidate's score is Σ log(N / block size) over shared keys, and we keep the top 50 per
  S1. S1 records are processed in batches of 5k, so peak memory stays around 4 GB.
- **Reverse search (v7):** each S2/S3 record is searched against the full S1 file of its country with
  the same key machinery, and its top 10 S1 businesses are kept. For every pair the model gets
  `rev_rank`, `rev_score_ratio`, `rev_n_close` and `from_fwd`. Each record's top 3 are also *added* as
  candidates (at most 20 per S1). On val, forward top-50 misses 8,118 true pairs; the reverse search
  finds the true S1 as #1 for 30% of them, top-3 for 37.5%, top-10 for 49%.
- **Learned filter:** a LightGBM model (≤600 trees) on ~40 per-pair features (no group features)
  gives each of the 50 candidates a probability. Candidates with p ≥ 0.03 survive. The filter's
  probabilities on training pairs are out-of-fold (2 folds by S1 entity).
- **Candidate pairs generated (test, v7):** bidirectional blocking 93.4M (53.9 per S1) → **filter
  survivors 7.23M (4.17 per S1) = `candidate_pairs.tsv`**.
- **How true matches are kept:** we measured blocking recall on val against the full pool,
  examined each category of miss, and added keys for each failure mode (no house number → A2;
  glued/domain names → NP; `1151a` → 1151; look-alike digits; translation). Pair recall at k=50 is
  95.3% (ceiling F0.5 with a perfect matcher: 0.982). The filter keeps 95.1% of true pairs, i.e.
  it drops only 0.2%, and val F0.5 is unchanged at cutoffs 0.001–0.03 (5.0 → 3.7 candidates per S1).

---

## 4. Matching Model

**Features used** (all on cleaned text, computed with RapidFuzz):
- **Name:** token-set, token-sort, Indel ratio, partial ratio, Jaro-Winkler. Also token-set and
  ratio on the *core* name, i.e. with words frequent in the country's pool (private, limited, llc,
  sarl…) removed; the list is learned, not hand-written.
- **Address:** token-set, token-sort, ratio, partial ratio, core-address token-set.
- **House numbers:** shared number (yes/no/unknown), first-number equality, number-set Jaccard,
  edit similarity of the first numbers (405 vs 40), best edit similarity over the first three
  numbers, one number contained in the other (123 / 7123), and number counts.
- **Record:** lengths, empty address, name in Indian script (before/after translation), source (S2/S3).
- **Blocking:** blocking score and rank.
- **Whole-file context:** number of S1 businesses with this S1's address (words in any order), street
  (numbers ignored) and core name, and number of S1 businesses with the *candidate's* core name.
  Example: among empty-address candidates with near-identical names, 70% are true matches when
  exactly one S1 business has that core name, versus 27% when more than 20 do.
- **Group (recomputed on the filter survivors):** combined similarity, its gap to the best candidate
  of the same S1, its rank, the gap between the top two, the number of strong candidates,
  name/address gaps, and blocking-score ratio.
- **Character 3-grams:** cosine similarity of hashed character 3-gram vectors of name, address and
  core name (catches spelling variants that word-level measures miss).
- **Transformer cross-encoder score (`ce`):** `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0,
  22.7M parameters) reads `"name | address" [SEP] "name | address"` of both records (cleaned,
  Indian script translated) and outputs a match probability. It is fine-tuned (1 epoch, AdamW lr 3e-5,
  batch 64, max 96 tokens) on the filter survivors of the train split in **two folds by S1 entity**:
  model k is fine-tuned on 300k pairs of fold k and scores the other fold, so the training feature
  is out-of-fold; val, holdout and test get the average of both models. On its own it scores 0.961
  holdout F0.5; as a LightGBM feature it lifts holdout from 0.9742 to 0.9794.
- Country is deliberately **not** a feature.

**Model type:** LightGBM binary classifiers (MIT licence; text models of a few MB each) plus the
fine-tuned MiniLM cross-encoder (Apache 2.0, 22.7M parameters, 2 fold copies of 87 MB), far below
the 8B-parameter limit. The final model has 127 leaves, learning rate 0.05, and is early-stopped
on val log-loss (655 trees with the transformer feature vs 1,822 without).

**Threshold selection method:** a two-step decision layer tuned on val macro F0.5:
1. *Competition:* each S2/S3 record keeps only the S1 that gives it the highest probability.
2. *Expected-F0.5 subset per S1:* with p1 ≥ p2 ≥ …, predict the top m candidates, where m
   maximises E[F0.5] ≈ 1.25·(p1+…+pm) / (m + 0.25·(Σp + miss)). Predicting nothing (m = 0) is an
   option with value λ·Π(1−pi); miss and λ are tuned on val.

---

## 5. Results & Error Analysis

| Version | Main change | Candidates per S1 | Val F0.5 | Holdout F0.5 | Public LB |
|---|---|---|---|---|---|
| – | predict nothing | 0 | 0.056 | – | – |
| v1 | rule: name ≥ 0.8, address ≥ 0.6, numbers agree | 49.8 | 0.799 | – | – |
| v2 | LightGBM, 38 features, group features, expected-F0.5 decision | 49.8 | 0.952 | 0.953 | 0.944 |
| v3 | + Indian-script translation, number containment, shared-address context | 49.8 | 0.961 | 0.961 | – |
| v4 | + learned candidate filter (cascade) | **3.9** | 0.962 | 0.9615 | – |
| v5 | + S1 name-uniqueness context | 3.8 | 0.9644 | 0.9637 | – |
| v6 | + character 3-gram similarity + reverse competition in the final model | 3.8 | 0.9688 | 0.9677 | 0.956 |
| v7 | bidirectional blocking (reverse candidates + reverse features for filter and final model), hub cap | 3.8 | 0.9747 | 0.9742 | 0.967 |
| **v8** | **+ fine-tuned transformer cross-encoder score (out-of-fold) as a LightGBM feature** | **3.8** | **0.9792** | **0.9794** | – |

- **F_0.5 Score (macro):** 0.9792 val / 0.9794 holdout (public leaderboard of v7: 0.967)
- **Where v7 still loses (val, total loss 0.0253):** true matches that never become candidates
  41.5% (5,414 pairs), candidates the model rejects 31.8% (4,733 pairs), wrong merges 12.1% (888
  pairs), mixed 10.0%, wrongly matched singletons 4.6% (58 entities).
- **Leaderboard vs holdout:** v2 −0.009, v6 −0.012, v7 −0.007. Test is somewhat harder than our
  validation. France (unseen) is the most likely reason: a manual review of French borderline pairs
  found category words swapped in names (Club ↔ Institut), small house-number shifts, and many
  businesses sharing one address (14.9% of French S1 vs ~7% in train).
- **Where the loss is (v3 val error analysis):** 93% of the lost F0.5 is in entities *with*
  matches (India 49%, US 43%), and only 7% in singletons.
- **Common false positives (wrong merges):** candidates with an empty address and a near-identical
  name that belong to another business, or are planted decoys (32% of wrong merges); records at a
  shared address; look-alikes differing only in the house number.
- **Common false negatives (missed matches):** true matches with an empty address (37%), with
  changed house numbers (`110` vs `11`, `7123` vs `123`), or with heavily corrupted names. A further
  4.7% of true pairs never reach the candidate set: mostly empty-address records with common names,
  and short addresses built from common words. Recall at 200 candidates would be 97.1%.

---

## 6. Conclusion
Measuring each stage separately on a realistic validation split, and reading the
errors, drove every improvement. Blocking set the ceiling, group features and context counts
("how many businesses share this address/name?") fixed look-alikes and shared addresses, and a
learned filter cut the candidate set 13× at no cost. The result is 0.80 → 0.96 macro F0.5 with
~4 candidates per entity. A small fine-tuned transformer, used as one feature rather than as the
matcher, added the last half point (0.974 → 0.979), using only the provided data, small MIT-licensed
LightGBM models and one Apache-2.0 MiniLM.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains `src/` (all code), `README.md` (exact run steps) and
`requirements.txt` (pinned versions). The entry point for the outputs is `src/predict.py --run <RUN>`,
which runs blocking → filter → features → final model → decision and writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. Training is reproduced by
`convert_to_parquet.py` → `make_split.py` → `learn_translit.py` → `make_pairs.py` →
`add_name_features.py` → `train_cascade.py` → `train_ce.py` (fine-tunes the cross-encoder) →
`train_final.py`. The pretrained MiniLM is stored in `src/model/pretrained/ms-marco-MiniLM-L-6-v2`
(Apache-2.0; in this public repository it is downloaded, see `code/business_entity_resolution/src/model/README.md`) and the fine-tuned fold models in `src/model/v8/` (not published here; regenerated by the pipeline). The README has a
table of all modules.

### B. Additional Results
Blocking recall vs candidates kept per S1 (val):

| k | 10 | 20 | 30 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Pair recall | 0.902 | 0.933 | 0.944 | 0.954 | 0.964 | 0.971 |
| Ceiling F0.5 | 0.963 | 0.975 | 0.979 | 0.983 | 0.986 | 0.989 |

Candidate filter cutoff vs size and score (v4, val):

| filter cutoff | 0.001 | 0.003 | 0.01 | 0.03 |
|---|---|---|---|---|
| Candidates per S1 | 5.00 | 4.46 | 4.02 | 3.72 |
| Val F0.5 | 0.9620 | 0.9621 | 0.9617 | 0.9620 |

