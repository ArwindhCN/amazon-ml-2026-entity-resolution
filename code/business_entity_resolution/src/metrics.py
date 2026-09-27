"""
Macro F0.5 scorer, following the official README formula:

    F0.5 = 1.25 * P * R / (0.25 * P + R)   per Source 1 entity, then averaged over ALL entities.

Run this file directly to self-test the scorer:
    .venv/bin/python code/business_entity_resolution/src/metrics.py
"""


def parse_ids(s):
    """'S2-1,S3-2' -> {'S2-1', 'S3-2'};  '' -> set()"""
    return {x for x in s.split(",") if x}


def f05_entity(pred, truth):
    """F0.5 for one S1 entity. pred and truth are sets of S2/S3 ids."""
    if not truth:              # singleton (README): empty prediction -> 1.0, anything else -> 0.0
        return 1.0 if not pred else 0.0
    if not pred:               # has matches but we predicted none: recall = 0 -> 0.0 (our inference)
        return 0.0
    tp = len(pred & truth)     # true positives: predicted ids that really match
    if tp == 0:
        return 0.0
    p = tp / len(pred)         # precision: share of our predictions that are right
    r = tp / len(truth)        # recall: share of true matches we found
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(pred, truth):
    """Average F0.5 over every entity in `truth`.

    pred, truth: dict  S1 id -> set of S2/S3 ids.
    An entity missing from `pred` counts as an empty prediction.
    """
    extra = set(pred) - set(truth)
    if extra:
        raise ValueError(f"{len(extra)} predicted entities are not in the evaluation set")
    return sum(f05_entity(pred.get(k, set()), t) for k, t in truth.items()) / len(truth)


def f05_macro_pairs(s1_all, n_true, pred_s1, pred_correct):
    """Fast (vectorized) macro F0.5 for pair-level predictions; same result as f05_macro.

    s1_all:       every S1 id in the evaluation set
    n_true:       dict or Series  S1 id -> number of true matches (0 for singletons)
    pred_s1:      S1 id of each PREDICTED pair
    pred_correct: for each predicted pair, is it a true match?
    """
    import numpy as np
    import pandas as pd
    g = pd.DataFrame({"s1": pred_s1, "tp": pred_correct}).groupby("s1")["tp"].agg(["size", "sum"])
    d = pd.DataFrame(index=pd.Index(s1_all, name="s1"))
    d["t"] = pd.Series(n_true).reindex(d.index).fillna(0).to_numpy()
    d = d.join(g).fillna(0)
    t, n, tp = d["t"].to_numpy(), d["size"].to_numpy(), d["sum"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n, tp / t
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where(t == 0, (n == 0).astype(float), f)   # singletons: 1 only if nothing predicted
    return float(f.mean())


if __name__ == "__main__":
    T = {"a", "b", "c", "d"}  # an entity with 4 true matches
    checks = [
        # (description, got, expected)
        ("README example (3 predicted, 2 right)",
         f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 0.714),
        ("singleton, predict empty", f05_entity(set(), set()), 1.0),
        ("singleton, predict anything", f05_entity({"x"}, set()), 0.0),
        ("has matches, predict empty", f05_entity(set(), {"a"}), 0.0),
        ("4 true, predict 1 right", f05_entity({"a"}, T), 0.625),
        ("4 true, predict 2 right", f05_entity({"a", "b"}, T), 0.833),
        ("4 true, predict all 4 + 1 wrong", f05_entity(T | {"x"}, T), 0.833),
        ("4 true, predict only wrong ones", f05_entity({"x", "y"}, T), 0.0),
        ("macro: README example + correct singleton",
         f05_macro({"e1": {"S2-00047", "S2-00193", "S3-00812"}},
                   {"e1": {"S2-00047", "S3-00812"}, "e2": set()}), 0.857),
    ]
    for desc, got, expected in checks:
        ok = abs(got - expected) < 1e-3
        print(f"{'PASS' if ok else 'FAIL'}  {desc:45s} got {got:.3f}  expected {expected:.3f}")
        assert ok

    # The fast vectorized version must give exactly the same answer as the simple one.
    import random
    rng = random.Random(0)
    truth = {f"e{i}": {f"m{i}_{j}" for j in range(rng.randint(0, 4))} for i in range(2000)}
    pred = {}
    for e, t in truth.items():
        guess = {m for m in t if rng.random() < 0.7} | {f"x{e}_{j}" for j in range(rng.randint(0, 2))}
        if guess:
            pred[e] = guess
    pairs = [(e, m in truth[e]) for e, g in pred.items() for m in g]
    fast = f05_macro_pairs(list(truth), {e: len(t) for e, t in truth.items()},
                           [e for e, _ in pairs], [c for _, c in pairs])
    slow = f05_macro(pred, truth)
    ok = abs(fast - slow) < 1e-12
    print(f"{'PASS' if ok else 'FAIL'}  {'vectorized scorer == simple scorer':45s} got {fast:.6f}  expected {slow:.6f}")
    assert ok
    print("All scorer checks passed.")
