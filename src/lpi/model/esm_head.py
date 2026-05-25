"""Frozen ESM-2 + tiny head: does sequence carry chemistry signal beyond gene content?

Small ESM-2 (facebook/esm2_t6_8M_UR50D, 8M params, CPU) is FROZEN; we mean-pool its
residue embeddings of each KR domain and train a tiny linear head. Two bacterial targets,
leave-cluster-out, vs the domain-rule / majority baselines:
  * KR stereochemistry (R vs S) -- domains carry no stereo signal;
  * inactive-domain detection (did the present KR fire) -- gene content cannot determine it.

Honest by design: report whichever way it lands. Beating the baselines = a "beyond the
rule" result (sequence carries the signal); not beating them = a clean null result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lpi.data.mibig import PROCESSED, RAW

_MODEL = "facebook/esm2_t6_8M_UR50D"
EMB_CACHE = RAW / "esm2_emb_cache.json"


def embed_sequences(seqs: dict[str, str]) -> dict[str, list[float]]:
    """Mean-pooled frozen-ESM-2 embedding per id (cached on disk)."""
    cache = json.loads(EMB_CACHE.read_text()) if EMB_CACHE.exists() else {}
    todo = {k: v for k, v in seqs.items() if k not in cache and v}
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModel.from_pretrained(_MODEL)
        model.eval()
        items = list(todo.items())
        with torch.no_grad():
            for i in range(0, len(items), 16):
                batch = items[i:i + 16]
                ids = [b[0] for b in batch]
                enc = tok([b[1][:1022] for b in batch], return_tensors="pt",
                          padding=True, truncation=True)
                out = model(**enc).last_hidden_state  # (B, L, 320)
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                for j, k in enumerate(ids):
                    cache[k] = pooled[j].tolist()
        EMB_CACHE.write_text(json.dumps(cache))
    return {k: cache[k] for k in seqs if k in cache}


@dataclass
class HeadResult:
    target: str
    n: int
    head_metric: float       # balanced accuracy (mean over leave-cluster-out folds)
    head_auc: float          # ROC-AUC where binary (else nan)
    baseline_majority: float
    baseline_domain_rule: float
    metric_name: str


def _train_linear(Xtr, ytr, Xte, n_classes, class_weight=None, epochs=300, seed=0):
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(np.asarray(Xtr, dtype=np.float32)), torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = net(torch.tensor(np.asarray(Xte, dtype=np.float32)))
        return logits.argmax(1).numpy(), torch.softmax(logits, 1).numpy()


def _roc_auc(y, score):
    """ROC-AUC via the rank (Mann-Whitney U) formula; no sklearn dependency."""
    y = np.asarray(y)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within tied groups (proper Mann-Whitney -> ties give 0.5)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _balanced_acc(y, pred):
    accs = []
    for c in np.unique(y):
        m = y == c
        if m.sum():
            accs.append((pred[m] == c).mean())
    return float(np.mean(accs))


def _leave_cluster_out_folds(groups, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_folds)
    for ch in chunks:
        test = np.array([g in set(ch) for g in groups])
        yield ~test, test


def evaluate_head(parquet=PROCESSED / "clustercad_kr_examples.parquet"):
    import pandas as pd

    df = pd.read_parquet(parquet)
    df = df[df["kr_sequence"].notna()].reset_index(drop=True)
    emb = embed_sequences(dict(zip(df["kr_domainid"], df["kr_sequence"])))
    df = df[df["kr_domainid"].isin(emb)].reset_index(drop=True)
    X = np.array([emb[d] for d in df["kr_domainid"]], dtype=np.float32)
    groups = df["cluster"].to_numpy()
    results = []

    # --- inactive-domain detection (did the present KR fire?) ---
    ya = df["kr_active"].to_numpy()
    bals, aucs = [], []
    for tr, te in _leave_cluster_out_folds(groups):
        if len(np.unique(ya[tr])) < 2 or te.sum() == 0:
            continue
        n0, n1 = (ya[tr] == 0).sum(), (ya[tr] == 1).sum()
        cw = [len(ya[tr]) / (2 * max(n0, 1)), len(ya[tr]) / (2 * max(n1, 1))]
        pred, proba = _train_linear(X[tr], ya[tr], X[te], 2, class_weight=cw)
        bals.append(_balanced_acc(ya[te], pred))
        if len(np.unique(ya[te])) == 2:
            aucs.append(_roc_auc(ya[te], proba[:, 1]))
    maj = max((ya == 0).mean(), (ya == 1).mean())
    results.append(HeadResult("inactive-KR detection", len(ya),
                              float(np.mean(bals)) if bals else float("nan"),
                              float(np.mean(aucs)) if aucs else float("nan"),
                              0.5, 0.5, "balanced-accuracy (majority/rule=0.5)"))

    # --- KR stereochemistry (R vs S; only stereocenters) ---
    st = df[df["stereo"].isin(["R", "S"])].reset_index(drop=True)
    if len(st) > 20:
        Xs = np.array([emb[d] for d in st["kr_domainid"]], dtype=np.float32)
        ys = (st["stereo"] == "S").astype(int).to_numpy()
        gs = st["cluster"].to_numpy()
        accs = []
        for tr, te in _leave_cluster_out_folds(gs):
            if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
                continue
            pred, _p = _train_linear(Xs[tr], ys[tr], Xs[te], 2)
            accs.append(float((pred == ys[te]).mean()))
        maj_s = max(ys.mean(), 1 - ys.mean())
        results.append(HeadResult("KR stereochemistry (R/S)", len(ys),
                                  float(np.mean(accs)) if accs else float("nan"),
                                  float("nan"), float(maj_s), float(maj_s),
                                  "accuracy (domain-rule has no stereo signal)"))
    return results
