"""Positional-prior probe: is the 0.567 iteration-grammar wall reducible by chain position?

Decisive, cheap experiment for the "running intermediate is the state pointer" idea. The
differential PoC (model/differential.py) established two anchors on the 30 curated fungal
cycles: 1.00 when the per-cycle ACTIVE domains are known (cheating; the chemistry transfers),
and 0.567 when given only the synthase's CONSTANT domain set (the iteration-grammar wall).

This probe inserts the honest middle term: condition the per-cycle reduction prediction on
chain POSITION (cycle index, distance from terminus, fraction along chain) -- information
available at genome-mining time given a hypothesised chain length, and NOT a leak of which
domains fired. We ask whether position recovers any of the 0.567 -> 1.00 gap, estimated two
ways: leave-one-synthase-out on fungi (does the grammar have positional structure at all?)
and bacteria-trained zero-shot (does ClusterCAD supervise that structure?).

n is tiny (30 cycles, 9 synthases): this is a go/no-go probe, not a validated result. If
position helps here, the next step is a real fungal program-trace corpus (NPAtlas/COCONUT);
if it does not, the grammar needs sequence/structure and we pivot.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from lpi.data.curated import load_all
from lpi.model.clustercad_labels import REDUCTION_CLASSES, label_module
from lpi.model.differential import _n_carbons, domain_rule_reduction

RED = list(REDUCTION_CLASSES)  # ("KETO", "KR", "DH", "ER")
_LBL = {"keto": "KETO", "kr": "KR", "dh": "DH", "er": "ER"}


def _avail_domains(states: list[str]) -> set[str]:
    """The synthase's constant (max-capability) reductive domain set, inferred from its
    most-reduced cycle -- the actual genome-mining situation (one iterative module)."""
    full = {"AT", "KS"}
    if any(s in ("kr", "dh", "er") for s in states):
        full |= {"KR"}
    if any(s in ("dh", "er") for s in states):
        full |= {"DH"}
    if any(s == "er" for s in states):
        full |= {"ER"}
    return full


def fungal_cycles():
    rows = []
    for e in load_all():
        if getattr(e, "subclass", "") not in ("HR", "PR"):
            continue
        cyc = list(e.program.cycles)
        N = len(cyc)
        states = [c.reduction.value for c in cyc]
        full = _avail_domains(states)
        for t, c in enumerate(cyc, start=1):
            rows.append(dict(syn=e.name[:26], sub=e.subclass, t=t, N=N,
                             from_end=N - t, frac=t / N, full=full,
                             label=_LBL[c.reduction.value]))
    return rows


def bacterial_cycles(path="data/processed/clustercad_modules.parquet"):
    """Per extension-module (position, available domains, structure-derived reduction)."""
    df = pd.read_parquet(path)
    by_clu = defaultdict(list)
    for _, r in df.iterrows():
        smi, prev = r["intermediate_smiles"], r["prev_intermediate_smiles"]
        if not smi or not prev:
            continue
        if (_n_carbons(smi) - _n_carbons(prev)) not in (2, 3):
            continue
        domset = set(str(r["domains"]).split(";")) if r["domains"] else set()
        if "KS" not in domset:
            continue
        dl = label_module(list(domset), smi)
        if dl is None:
            continue
        by_clu[r["cluster"]].append((int(r["module_idx"]), domset, dl.reduction))
    rows = []
    for clu, mods in by_clu.items():
        mods.sort()
        N = len(mods)
        for i, (_midx, domset, red) in enumerate(mods, start=1):
            rows.append(dict(syn=clu, t=i, N=N, from_end=N - i, frac=i / N,
                             full=domset, label=red))
    return rows


def _fe_bucket(from_end: int) -> int:
    return min(from_end, 3)  # {0=terminal,1,2,3+}


def _ceiling_state(full: set[str]) -> str:
    """The domain-rule (max reductive) state for an available domain set -- this IS the
    constant-domain wall's single prediction for every cycle."""
    return RED[domain_rule_reduction(full)]


# ---- predictors -------------------------------------------------------------

def pred_wall(rows):
    """Constant-domain wall: one prediction (the ceiling state) for all cycles."""
    return [_ceiling_state(r["full"]) for r in rows]


def pred_majority(train, rows):
    maj = Counter(r["label"] for r in train).most_common(1)[0][0]
    return [maj for _ in rows]


def _fit_pos_table(train):
    """bucket (ceiling_state, from_end_bucket) -> majority reduction label."""
    buckets = defaultdict(Counter)
    for r in train:
        buckets[(_ceiling_state(r["full"]), _fe_bucket(r["from_end"]))][r["label"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in buckets.items()}


def pred_positional(train, rows):
    table = _fit_pos_table(train)
    out = []
    for r in rows:
        key = (_ceiling_state(r["full"]), _fe_bucket(r["from_end"]))
        out.append(table.get(key, _ceiling_state(r["full"])))  # fall back to wall
    return out


def acc(preds, rows):
    return sum(p == r["label"] for p, r in zip(preds, rows)) / len(rows)


def loso(rows, predictor):
    """Leave-one-synthase-out accuracy for a (train, test)->preds predictor."""
    syns = sorted({r["syn"] for r in rows})
    correct = 0
    preds_all = {}
    for s in syns:
        tr = [r for r in rows if r["syn"] != s]
        te = [r for r in rows if r["syn"] == s]
        p = predictor(tr, te)
        preds_all[s] = list(zip([x["t"] for x in te], [x["label"] for x in te], p))
        correct += sum(pi == ri["label"] for pi, ri in zip(p, te))
    return correct / len(rows), preds_all


def main():
    fung = fungal_cycles()
    nsyn = len({r["syn"] for r in fung})
    print(f"FUNGAL cycles: {len(fung)} across {nsyn} synthases "
          f"(HR={sum(r['sub']=='HR' for r in fung)}, PR={sum(r['sub']=='PR' for r in fung)})\n")

    wall = pred_wall(fung)
    print(f"  constant-domain WALL (reproduce paper 0.567): {acc(wall, fung):.3f}")
    print(f"  global majority                             : "
          f"{acc(pred_majority(fung, fung), fung):.3f}")

    pos_ceiling = acc(pred_positional(fung, fung), fung)  # in-sample upper bound
    pos_loso, per = loso(fung, pred_positional)
    print(f"  positional prior (in-sample ceiling)        : {pos_ceiling:.3f}")
    print(f"  positional prior (leave-one-synthase-out)   : {pos_loso:.3f}   <-- honest fungal")

    bact = bacterial_cycles()
    nbc = len({r["syn"] for r in bact})
    table = _fit_pos_table(bact)
    pos_bact = acc([table.get((_ceiling_state(r["full"]), _fe_bucket(r["from_end"])),
                               _ceiling_state(r["full"])) for r in fung], fung)
    print(f"  positional prior (BACTERIA-trained, zero-shot, {len(bact)} mods/{nbc} clu): "
          f"{pos_bact:.3f}")

    # stratify
    for sub in ("HR", "PR"):
        sub_rows = [r for r in fung if r["sub"] == sub]
        if sub_rows:
            w = acc(pred_wall(sub_rows), sub_rows)
            pl, _ = loso(sub_rows, pred_positional)
            print(f"    [{sub}] n={len(sub_rows):2d}  wall={w:.3f}  pos-LOSO={pl:.3f}")

    print("\n  per-synthase (t: true -> pos-LOSO pred):")
    for s, items in per.items():
        flags = "".join("." if t == p else "X" for _, t, p in items)
        print(f"    {s:26s} {flags}  " +
              " ".join(f"{t}:{tr}->{pp}" for t, tr, pp in items))

    # bacterial positional structure: does reduction depend on position there?
    print("\n  BACTERIAL positional table (ceiling_state, from_end_bucket) -> majority:")
    for k in sorted(table):
        n = sum(1 for r in bact if (_ceiling_state(r["full"]), _fe_bucket(r["from_end"])) == k)
        print(f"    {k} -> {table[k]:5s} (n={n})")


if __name__ == "__main__":
    main()
