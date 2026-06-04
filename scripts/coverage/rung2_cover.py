"""Rung-2 sound set-cover over the Completion-1 certified gaps -> the next operator to build.

Completion 1 produced a SOUND, typed new-chemistry frontier: 117 cores provably beyond Theta_0's reach up
to (N=9, C<=20) -- 83 sound-structural-gap + 34 formula-infeasible. This script answers the build-queue
question the rung ladder never had cleanly: **which single soundly-expressible (rung-2) operator unlocks
the most provable gaps?** -- by tagging each gap with the operator-CLASS it needs (the census structural
proxy `cho_classes`), restricting to RUNG-2 classes, and ranking by gaps unlocked (per-class clean yield +
the greedy set-cover).

WHAT "RUNG-2" MEANS HERE (earned PER CLASS, not read off structure -- the ground rule). A class is rung-2
(soundly expressible) iff its operator's degree of freedom is FORMULA-INVARIANT and recoverable from
cluster features phi(B) -- the executor never guesses it (returns None on ambiguity). The macrolactone
ring-size (curated locant) is the canonical rung-2 DOF already in Theta_0. rung-3 splits into rung-3-MASS
(formula-CHANGING: oxygenase / prenyl / skeleton -- the 34 mass gaps) and rung-3-OPEN (formula-invariant
but fold/dynamics-gated, e.g. non-aromatic carbocyclization -- not yet soundly writable).

HONESTY (carried from the census + Completion 1):
  * The tag is a STRUCTURAL PROXY -- necessary, NOT sufficient. A class tag does NOT guarantee the operator
    can be written soundly, nor that ONE new operator covers every core carrying that tag (they may need
    distinct registers). So per-class counts are an UPPER BOUND on a single operator's yield.
  * The SOUND yield is confirmed only by BUILDING the operator + re-running `make forward-index`
    (R_term non-decreasing in Theta_0; kappa-certified; assert zero gap->? demotions). This script RANKS
    the build; it does not certify the gain.
  * rung-3 classes are reported but EXCLUDED from the rung-2 queue: build what is soundly buildable, not
    merely most-demanded.

Reproduction: `make rung2-cover` (reads results/forward_index.csv + the parquet; deterministic).
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "coverage"))

from lpi.chem import mol as M  # noqa: E402
from oog_census import cho_classes, greedy_curve  # noqa: E402  (reuse the census tagger + set-cover)

FWD = ROOT / "results" / "forward_index.csv"
PAIRS = ROOT / "data" / "processed" / "fungal_pks_pairs.parquet"
GAP_VERDICTS = ("sound-structural-gap", "formula-infeasible")

# Rung earned PER CLASS via the sound-expressibility criterion (formula-invariant DOF, recoverable from
# phi(B); executor returns None on ambiguity). NOT read off structure.
CLASS_RUNG = {
    "aromatic_cyclization":    "rung-2",       # aromatic-cyclization regiochemistry: formula-invariant curated
    #   register (PT/product-template clade, Li 2010) -- same kind as RESORCYLIC_MACROLACTONE / ALDOL_AROMATIC
    #   already in Theta_0.
    "macrolactonization":      "rung-2",       # cis-TE macrolactone, ring-size = curated locant -- a rung-2 DOF
    #   already in Theta_0 (2 modes); the gaps need a NEW register. Split out of the census's lumped tag below.
    "non_aromatic_carbocycle": "rung-3-open",  # non-aromatic CARBOCYCLE / small lactone (Diels-Alder / Michael /
    #   pyranone): ring geometry is FOLD/DYNAMICS-gated -- the chemistry twin of C2's conformational residue.
    "prenylation":             "rung-3-mass",  # prenyl adds C5H8 (extrinsic DMAPP) -> formula-changing tailoring
    "oxidative_tailoring":     "rung-3-mass",  # adds O (oxygenase) -> formula-changing tailoring (the mass gaps)
    "other_formula":           "rung-3-mass",  # H/C skeleton mass gap -> formula-changing
    "other_topology":          "unclassified", # release/linear-topology fallback -> cannot earn a rung
}
RUNG2 = frozenset(k for k, v in CLASS_RUNG.items() if v == "rung-2")

# Split the census's lumped `non_aromatic_ring_rearrangement` into rung-2 macrolactone vs rung-3-open other.
_MIN_RING = 8  # cis-TE macrolactone min ring (src/lpi/executor/cyclize.py:_MACROLACTONE_MIN_RING)
_INRING_ESTER = Chem.MolFromSmarts("[#8;R]-[#6;R]=[#8;!R]")  # ring-O - ring-C(=O)exocyclic = lactone linkage


def _nonaromatic_split(mol) -> set:
    """A non-aromatic ring is rung-2 `macrolactonization` if it is a macrocycle (>= _MIN_RING) carrying an
    in-ring lactone ester; every other non-aromatic ring (carbocycle, pyranone, delta-lactone) is rung-3-open
    `non_aromatic_carbocycle` (fold/dynamics-gated geometry)."""
    ri = mol.GetRingInfo()
    ester = mol.GetSubstructMatches(_INRING_ESTER)
    tags: set = set()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
               for i in ring):
            continue  # fully-aromatic carbocycle -> aromatic_cyclization already covers it
        rs = set(ring)
        is_macro = len(ring) >= _MIN_RING and any(m[0] in rs and m[1] in rs for m in ester)
        tags.add("macrolactonization" if is_macro else "non_aromatic_carbocycle")
    return tags


def refined_classes(mol) -> frozenset:
    """Census tagger with its lumped non-aromatic-ring tag split into rung-2 macrolactone vs rung-3-open."""
    cls = set(cho_classes(mol))
    if "non_aromatic_ring_rearrangement" in cls:
        cls.discard("non_aromatic_ring_rearrangement")
        cls |= _nonaromatic_split(mol)
    return frozenset(cls)


def load_gaps() -> list:
    rows = list(csv.DictReader(FWD.open()))
    gaps = [r for r in rows if r["forward_verdict"] in GAP_VERDICTS]
    df = pd.read_parquet(PAIRS)
    smi = {}
    for _, r in df.iterrows():
        s = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        if s:
            smi[r["bgc_id"]] = s
    out = []
    for g in gaps:
        s = smi.get(g["bgc_id"])
        if not s:
            continue
        cls = refined_classes(M.mol_from_smiles(s))
        out.append((g["bgc_id"], g["name"], g["forward_verdict"], int(g["C"]), cls))
    return out


def main() -> None:
    gaps = load_gaps()
    struct = [g for g in gaps if g[2] == "sound-structural-gap"]
    mass = [g for g in gaps if g[2] == "formula-infeasible"]
    print("RUNG-2 SOUND SET-COVER over the Completion-1 certified gaps")
    print(f"  {len(gaps)} gaps = {len(struct)} sound-structural-gap + {len(mass)} formula-infeasible\n")

    # ---- per-class demand (a core may carry several classes), with its earned rung ----
    demand = Counter(c for *_, cs in gaps for c in cs)
    demand_struct = Counter(c for g in struct for c in g[4])
    print("  per-class demand over the 117 (proxy tag; a core may carry several classes):")
    print(f"    {'class':34s} {'rung':12s} {'all-117':>8s} {'of-83-struct':>13s}")
    for cls, _n in demand.most_common():
        print(f"    {cls:34s} {CLASS_RUNG.get(cls, '?'):12s} {demand[cls]:>8d} {demand_struct.get(cls, 0):>13d}")

    # ---- entanglement: how many gaps are PURELY rung-2 (a rung-2 build can fully unlock them) ----
    pure_r2 = [g for g in gaps if g[4] <= RUNG2]
    has_r2 = [g for g in gaps if g[4] & RUNG2]
    entangled = [g for g in has_r2 if not g[4] <= RUNG2]
    print(f"\n  rung-2 entanglement:")
    print(f"    gaps carrying a rung-2 class            : {len(has_r2)}")
    print(f"    of those, PURELY rung-2 (set ⊆ rung-2)  : {len(pure_r2)}  <- soundly unlockable by rung-2 ops alone")
    print(f"    entangled with a rung-3 class           : {len(entangled)}  <- need a rung-3 op too (not yet sound)")

    # ---- the rung-2 build-queue: per rung-2 class, clean (sole-class) yield vs proxy demand ----
    print(f"\n  RUNG-2 BUILD-QUEUE (rank by provable reach a SINGLE new operator buys):")
    print(f"    {'rung-2 operator class':34s} {'clean(sole)':>12s} {'demand(appears)':>16s}")
    ranked = []
    for cls in sorted(RUNG2):
        clean = sum(1 for g in gaps if g[4] == {cls})          # sole class -> one operator fully unlocks
        appears = demand[cls]                                  # upper bound (may share with other needs)
        ranked.append((cls, clean, appears))
    for cls, clean, appears in sorted(ranked, key=lambda x: -x[1]):
        print(f"    {cls:34s} {clean:>12d} {appears:>16d}")

    # ---- greedy set-cover (full, all classes) for context: shows the whole demand sequence ----
    print(f"\n  greedy set-cover over ALL 117 (context -- a core is recovered when ALL its classes covered):")
    for step in greedy_curve([list(g[4]) for g in gaps]):
        print(f"    +{step['op']:34s} (n_ops={step['n_ops']}) -> {step['recovered']:>3d}/{len(gaps)} "
              f"({step['frac']:.0%})  rung={CLASS_RUNG.get(step['op'], '?')}")

    # ---- verdict ----
    top = max(ranked, key=lambda x: x[1]) if ranked else None
    print(f"\n  {'=' * 68}")
    if top and top[1] > 0:
        print(f"  VERDICT -- next operator to build (rung-2, max provable reach):")
        print(f"    >> {top[0]} <<  clean yield {top[1]} gaps (sole-class), {top[2]} demand (upper bound)")
        print(f"  PROXY caveat: {top[1]} is the count of gaps tagged SOLELY '{top[0]}'. A single new register may")
        print(f"  not cover all (distinct regiochemistries); the SOUND yield is whatever `make forward-index`")
        print(f"  reports as gap->reachable after the operator is built + kappa-certified.")
    else:
        print("  VERDICT: no rung-2 class carries a sole-class gap -- the buildable reach is entangled; investigate.")
    print(f"  {'=' * 68}")


if __name__ == "__main__":
    main()
