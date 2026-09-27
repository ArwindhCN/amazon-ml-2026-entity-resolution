"""
Transformer cross-encoder: reads BOTH records of a pair at once and scores how likely they match.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache 2.0, 22.7M parameters), a small pretrained
MiniLM already trained for text-pair relevance, stored in src/model/pretrained/. We fine-tune it on
our labelled candidate pairs (train split only) and use its score as ONE extra feature for the final
LightGBM, so if the transformer is weak, LightGBM simply gives it little weight.

Input per pair (cleaned, translated text):  "name | address"  [SEP]  "name | address"
Runs on the Apple GPU (MPS) when available, otherwise on the CPU (e.g. Windows laptops).
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PRETRAINED = Path(__file__).with_name("model") / "pretrained" / "ms-marco-MiniLM-L-6-v2"
MAX_LEN = 96


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pair_text(names, addrs):
    return [f"{n} | {a}" for n, a in zip(names, addrs)]


def load(model_dir=PRETRAINED):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=1)
    return tok, model.to(device())


@torch.no_grad()
def score(tok, model, a, b, batch=512, log=None):
    """Match probability for pairs (a[i], b[i]). Sorted by length for speed, returned in input order."""
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
    out = np.empty(len(a), np.float32)
    t0 = time.time()
    for k, i in enumerate(range(0, len(a), batch)):
        idx = order[i:i + batch]
        enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt").to(model.device)
        out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy()
        if log and k % 200 == 0:
            log(f"    scored {i + len(idx):,}/{len(a):,} pairs ({(i + len(idx)) / (time.time() - t0):.0f}/s)")
    return out


def fine_tune(tok, model, a, b, y, epochs=1, batch=64, lr=3e-5, warmup=0.05, seed=0, log=None):
    """Binary fine-tuning with BCE loss, AdamW, linear warm-up/decay."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(a)
    steps = epochs * math.ceil(n / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup * steps)) * max(0.0, (steps - s) / steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(model.device)
            target = torch.tensor(np.asarray(y)[idx], dtype=torch.float32, device=model.device)
            loss = loss_fn(model(**enc).logits.squeeze(-1), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if log and step % 200 == 0:
                log(f"    step {step:,}/{steps:,} loss {loss.item():.4f} ({step / (time.time() - t0):.1f} steps/s)")
    model.eval()
    return model
