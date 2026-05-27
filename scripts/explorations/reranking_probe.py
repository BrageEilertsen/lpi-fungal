"""Verifier-constrained reranking probe: does a learned prior + mass rank the TRUE iteration
program above chance within the sound legal set? (A bounded test of the "turn the 0.567 wall
into a probabilistic ranking while preserving soundness" idea.)

The honest setup. Accurate mass pins the MULTISET of per-cycle reduction levels (the molecular
formula depends only on how many KETO/KR/DH/ER, not on their order). The residual ambiguity is
the ORDER -- which cycle gets which reduction -- and that residual IS the 0.567 iteration-grammar
wall, here expressed as a ranking problem. So:

    Z*(mass) = the distinct orderings of the true reduction multiset
             = real executor programs, same formula -> same mass -> mass cannot separate them.

We rank the TRUE ordering inside Z* by a leave-one-synthase-out position prior (the only kind of
"learned policy" 9 synthases can support), against a uniform baseline. Soundness is preserved by
construction: the prior only REORDERS a set whose every member is a legal executor program.

n is tiny (curated HR/PR synthases with >1 ordering). This is a go/no-go probe, not a validated
model -- report whatever it says, including null. The ESM/whole-sequence policy is the data-gated
upgrade; this probe asks whether the *architecture* (rerank within soundness) buys anything at all
on the residual that mass leaves.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState
from lpi.data.curated import load_all
from lpi.executor import core
from lpi.observe import exact_mass

LEVELS = ("keto", "kr", "dh", "er")


def tercile(t: int, n: int) -> str:
    f = (t - 0.5) / n
    return "early" if f < 1 / 3 else "late" if f > 2 / 3 else "mid"


def records():
    """(subclass, tercile, level) over all curated entries -- the prior's training pool."""
    rows = []
    for e in load_all():
        cyc = e.program.cycles
        n = len(cyc)
        for t, c in enumerate(cyc, start=1):
            rows.append((e.subclass, tercile(t, n), c.reduction.value))
    return rows


def fit_prior(rows):
    """P(level | subclass, tercile) with Laplace smoothing (transparent; tiny-n honest)."""
    cnt: dict = defaultdict(lambda: {lv: 0 for lv in LEVELS})
    for sub, ter, lv in rows:
        cnt[(sub, ter)][lv] += 1
    prior: dict = {}
    for key, c in cnt.items():
        tot = sum(c.values()) + len(LEVELS)
        prior[key] = {lv: (c[lv] + 1) / tot for lv in LEVELS}
    # global fallback
    g = {lv: 0 for lv in LEVELS}
    for _, _, lv in rows:
        g[lv] += 1
    gt = sum(g.values()) + len(LEVELS)
    prior["_global"] = {lv: (g[lv] + 1) / gt for lv in LEVELS}
    return prior


def score(order, sub, prior):
    n = len(order)
    s = 0.0
    for t, lv in enumerate(order, start=1):
        p = prior.get((sub, tercile(t, n)), prior["_global"])
        s += math.log(p[lv])
    return s


def _program(entry, order):
    cyc = entry.program.cycles
    ncyc = [Cycle(ReductionState(order[i]), cyc[i].c_methyl, cyc[i].extender)
            for i in range(len(cyc))]
    return Program(entry.program.starter, tuple(ncyc), entry.program.release)


def _mass_of(entry, order):
    try:
        return exact_mass(M.canonical_smiles(core.exec(_program(entry, order))))
    except Exception:  # noqa: BLE001 - an ordering whose cyclization cannot fire
        return None


def zstar(entry):
    """The SOUND mass-legal set: reduction-orderings whose EXECUTED product has the same mass as
    the true product. This is literally generate -> execute -> filter by mass (the verifier step).
    Orderings that cyclize differently land at a different mass and are pruned by mass ALONE -- so
    Z* is the genuine residual the iteration-program prior would have to rank within."""
    cyc = entry.program.cycles
    true = tuple(c.reduction.value for c in cyc)
    tm = _mass_of(entry, true)
    if tm is None:
        return [], true, None
    legal = [p for p in {pp for pp in itertools.permutations(true)}
             if (m := _mass_of(entry, p)) is not None and abs(m - tm) < 1e-4]
    return sorted(legal), true, tm


def true_rank(orders, true, sub, prior):
    scored = sorted(orders, key=lambda o: score(o, sub, prior), reverse=True)
    s_true = score(true, sub, prior)
    higher = sum(1 for o in orders if score(o, sub, prior) > s_true)
    ties = sum(1 for o in orders if abs(score(o, sub, prior) - s_true) < 1e-9)
    return higher + (ties + 1) / 2  # average rank over the tie block


def soundness_check(entry):
    """Show the residual: among the mass-LEGAL orderings (same executed mass as the true product),
    the structures are genuinely distinct -- mass cannot separate them; that is what a prior must."""
    orders, _true, tm = zstar(entry)
    smis = [M.canonical_smiles(core.exec(_program(entry, o))) for o in orders[:4]]
    return tm, len(orders), smis


def main():
    rows = records()
    print("Verifier-constrained reranking probe -- does a LOSO prior rank the TRUE iteration")
    print("program above chance within the mass-legal set Z*?  (soundness preserved: Z* = real")
    print("executor programs; the prior only reorders them.)\n")
    hdr = f"  {'synthase':34s} {'sub':3s} {'N':>2s} {'|Z*|':>5s} {'rank':>6s} {'rank/|Z*|':>9s} {'uniform':>8s}"
    print(hdr)
    agg = []
    for e in load_all():
        orders, true, tm = zstar(e)
        if len(orders) <= 1:
            continue  # mass already pins it: |Z*|=1, nothing to rerank
        # leave-one-synthase-out: refit the prior excluding exactly this entry's own cycles
        own = [(e.subclass, tercile(t, len(true)), lv) for t, lv in enumerate(true, 1)]
        train = list(rows)
        for r in own:
            train.remove(r)
        loso = fit_prior(train)
        rank = true_rank(orders, true, e.subclass, loso)
        uniform = (len(orders) + 1) / 2
        agg.append((e.name, e.subclass, rank, len(orders), uniform))
        print(f"  {e.name[:34]:34s} {e.subclass:3s} {len(true):>2d} {len(orders):>5d} "
              f"{rank:>6.1f} {rank/len(orders):>9.2f} {uniform:>8.1f}")
    print()
    if agg:
        mrr_p = sum(1 / r for _, _, r, _, _ in agg) / len(agg)
        mrr_u = sum(1 / ((z + 1) / 2) for _, _, _, z, _ in agg) / len(agg)
        norm_p = sum(r / z for _, _, r, z, _ in agg) / len(agg)
        top1 = sum(1 for _, _, r, _, _ in agg if r <= 1.5)
        print(f"  prior:   mean rank/|Z*| = {norm_p:.3f}   MRR = {mrr_p:.3f}   top-1 = {top1}/{len(agg)}")
        print(f"  uniform: mean rank/|Z*| = {0.5:.3f}   MRR = {mrr_u:.3f}")
        print(f"  (rank/|Z*| < 0.5 means the prior beats chance; n={len(agg)} synthases -- a probe.)")
    print()
    ten = next(e for e in load_all() if "tenellin" in e.name)
    tm, nz, smis = soundness_check(ten)
    print(f"Soundness check (tenellin, HR): the mass-legal set Z* = {nz} orderings, all executed to")
    print(f"mass {tm:.4f}; e.g.")
    for smi in smis:
        print(f"   {smi}")
    print("  -> same executed mass, distinct structures: mass cannot separate them (the residual a")
    print("     prior must rank). Cyclizing cores instead get pruned by mass to a smaller Z*.")


if __name__ == "__main__":
    main()
