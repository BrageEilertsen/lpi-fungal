"""Differential PoC model: domain features -> per-step action (reduction state, KR stereo).

The smallest model that tests contribution 5: is the transferable per-step chemistry
learnable from abundant bacterial ClusterCAD modules? A tiny MLP over domain-content
features with two heads (reduction state 4-way; KR stereo 3-way). Compared against a
majority baseline and the antiSMASH-style domain rule, evaluated leave-cluster-out, and
transfer-tested on fungal modules.

Honest expectation: for *colinear* bacterial PKS, reduction state is near-deterministic
from domain content, so the MLP should match the domain rule (that is the point -- per-step
chemistry is computable when the grammar is colinear). KR stereo is sequence-motif-dependent
and NOT recoverable from domains -> near-chance (motivating the ESM-2 stretch).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lpi.model.clustercad_labels import (FEATURE_DOMAINS, REDUCTION_CLASSES,
                                         STEREO_CLASSES, label_module)


def _n_carbons(smiles: str) -> int:
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles or "")
    return sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()) if m else -1


def build_dataset(parquet_path):
    """Clean per-step labels for genuine EXTENSION modules only.

    A module is an extension step iff it has a previous intermediate and added a C2/C3
    ketide unit (ΔC in {2,3}). This excludes loading/starter modules, whose saturated
    starter carbon would otherwise be misread as a reduction event.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    X, y_red, y_st, groups, doms = [], [], [], [], []
    for _, r in df.iterrows():
        smi, prev = r["intermediate_smiles"], r["prev_intermediate_smiles"]
        if not smi or not prev:
            continue
        dc = _n_carbons(smi) - _n_carbons(prev)
        if dc not in (2, 3):  # not a single malonyl/methylmalonyl extension
            continue
        domset = set(str(r["domains"]).split(";")) if r["domains"] else set()
        if "KS" not in domset:  # extension requires a ketosynthase
            continue
        dl = label_module(list(domset), smi)
        if dl is None:
            continue
        X.append([dl.features[d] for d in FEATURE_DOMAINS])
        y_red.append(REDUCTION_CLASSES.index(dl.reduction))
        y_st.append(STEREO_CLASSES.index(dl.stereo))
        groups.append(r["cluster"])
        doms.append(domset)
    return (np.array(X, dtype=np.float32), np.array(y_red), np.array(y_st),
            np.array(groups), doms)


def domain_rule_reduction(domain_set: set[str]) -> int:
    """antiSMASH-style baseline: reduction state = which reductive domains are present."""
    if "ER" in domain_set and "DH" in domain_set and "KR" in domain_set:
        return REDUCTION_CLASSES.index("ER")
    if "DH" in domain_set and "KR" in domain_set:
        return REDUCTION_CLASSES.index("DH")
    if "KR" in domain_set:
        return REDUCTION_CLASSES.index("KR")
    return REDUCTION_CLASSES.index("KETO")


@dataclass
class MLPResult:
    mlp_reduction_acc: float
    domain_rule_acc: float
    majority_acc: float
    stereo_mlp_acc: float
    stereo_majority_acc: float
    n_train: int
    n_test: int
    per_class_reduction: dict = field(default_factory=dict)


def _train_mlp(Xtr, ytr, Xte, n_classes, epochs=300, seed=0):
    import torch

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 16), torch.nn.ReLU(),
        torch.nn.Linear(16, n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    lossf = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(torch.tensor(Xte)).argmax(1).numpy()


def evaluate(parquet_path, holdout_frac=0.3, seed=0) -> MLPResult:
    X, y_red, y_st, groups, doms = build_dataset(parquet_path)
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n_hold = max(1, int(len(uniq) * holdout_frac))
    test_clusters = set(uniq[:n_hold])  # leave-cluster-out (no module leakage)
    te = np.array([g in test_clusters for g in groups])
    tr = ~te

    pred_red = _train_mlp(X[tr], y_red[tr], X[te], len(REDUCTION_CLASSES), seed=seed)
    mlp_acc = float((pred_red == y_red[te]).mean())
    rule_pred = np.array([domain_rule_reduction(doms[i]) for i in np.where(te)[0]])
    rule_acc = float((rule_pred == y_red[te]).mean())
    maj = np.bincount(y_red[tr]).argmax()
    maj_acc = float((y_red[te] == maj).mean())

    pred_st = _train_mlp(X[tr], y_st[tr], X[te], len(STEREO_CLASSES), seed=seed)
    st_acc = float((pred_st == y_st[te]).mean())
    st_maj = np.bincount(y_st[tr]).argmax()
    st_maj_acc = float((y_st[te] == st_maj).mean())

    per_class = {}
    for ci, cname in enumerate(REDUCTION_CLASSES):
        mask = y_red[te] == ci
        if mask.sum():
            per_class[cname] = (int(mask.sum()), float((pred_red[mask] == ci).mean()))

    return MLPResult(mlp_acc, rule_acc, maj_acc, st_acc, st_maj_acc,
                     int(tr.sum()), int(te.sum()), per_class)


def _fungal_cycles():
    """Per-cycle (active reductive domains, label) from the curated fungal HR/PR programs."""
    from lpi.data.curated import load_all

    rows = []
    for e in load_all():
        if e.subclass not in ("HR", "PR"):
            continue
        full = {"AT", "KS"}  # the synthase's constant domain set (its max capability)
        states = [c.reduction.value for c in e.program.cycles]
        if "kr" in states or "dh" in states or "er" in states:
            full |= {"KR"}
        if "dh" in states or "er" in states:
            full |= {"DH"}
        if "er" in states:
            full |= {"ER"}
        for c in e.program.cycles:
            r = c.reduction.value
            active = {"AT", "KS"}
            if r in ("kr", "dh", "er"):
                active |= {"KR"}
            if r in ("dh", "er"):
                active |= {"DH"}
            if r == "er":
                active |= {"ER"}
            label = {"keto": "KETO", "kr": "KR", "dh": "DH", "er": "ER"}[r]
            rows.append((e.name, active, full, REDUCTION_CLASSES.index(label)))
    return rows


@dataclass
class TransferResult:
    n_cycles: int
    active_domains_acc: float   # given the cycle's ACTIVE domains (per-step chemistry)
    constant_domains_acc: float  # given the synthase's CONSTANT domain set (grammar wall)


def transfer_test(parquet_path, seed=0) -> TransferResult:
    """Apply the bacteria-trained reduction head to fungal cycles, two ways:
    (a) active-domains: per-step chemistry should transfer (grammar-independent);
    (b) constant synthase domains: one prediction for all cycles -> the iteration-grammar wall.
    """
    X, y_red, _y, groups, _doms = build_dataset(parquet_path)
    cycles = _fungal_cycles()
    if not cycles:
        return TransferResult(0, 0.0, 0.0)
    Xa = np.array([[int(d in act) for d in FEATURE_DOMAINS] for _n, act, _f, _l in cycles],
                  dtype=np.float32)
    Xc = np.array([[int(d in full) for d in FEATURE_DOMAINS] for _n, _a, full, _l in cycles],
                  dtype=np.float32)
    yf = np.array([lab for *_x, lab in cycles])
    pred_a = _train_mlp(X, y_red, Xa, len(REDUCTION_CLASSES), seed=seed)
    pred_c = _train_mlp(X, y_red, Xc, len(REDUCTION_CLASSES), seed=seed)
    return TransferResult(len(cycles), float((pred_a == yf).mean()),
                          float((pred_c == yf).mean()))
