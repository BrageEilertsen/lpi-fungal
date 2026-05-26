"""Design-by-grammar-inversion: the structural/control realizability geometry.

Every target program gets a coordinate against the natural-cluster manifold, decomposed into two
edit axes that map onto different wet-lab capabilities:

* STRUCTURAL edits change domain content (add a reductive domain, swap starter/extender, reprogram
  the release) -- transferable from bacterial modular-PKS engineering precedent;
* CONTROL edits change the iteration program with the domain set fixed (which cycle a domain fires
  on; the iteration count) -- the iteration-grammar frontier, the 100%->57% wall as a design axis.

The headline experiment is the 1-edit design neighbourhood of the (small, high-value) fungal
manifold: how many designs are structural-only ('engineerable' with documented edits) vs require a
control edit ('frontier', iteration-program editing). For fungal iterative PKS the control axis
dominates -- design here *is* iteration-program editing, which only this architecture can formulate.

Cost basis: edit tiers are literature-curated (data/curation/edit_tiers.csv + edit_precedent.md);
control edits stack super-additively (Cox 2023). The full MIBiG/ClusterCAD manifold is the flagged
data step (bacterial generalization).
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.realizability import design_space, distinct_products, natural_manifold, realize

RDLogger.DisableLog("rdApp.*")

man = natural_manifold()

EXAMPLES = [
    ("6-MSA + KR@cyc3 (reprogram firing)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), Release.ALDOL_AROMATIC)),
    ("6-MSA + ER@cyc2 (add ER domain + fire it)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.ER), Cycle(R.KETO)), Release.ALDOL_AROMATIC)),
    ("6-MSA iteration++ (extra reduced cycle)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO), Cycle(R.KETO)),
             Release.ALDOL_AROMATIC)),
    ("6-MSA -> lactone (release reprogram)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), Release.LACTONIZATION)),
]


def main() -> None:
    print("Design-by-grammar-inversion: structural/control realizability over the fungal manifold\n")

    ds = design_space(man)
    print(f"1-edit design neighbourhood of {len(man)} fungal templates -> {ds['n_designs']} distinct designs")
    print(f"{'':46s} {'edit-specs':>10s} {'distinct products':>18s}")
    for k, note in [("engineerable", "structural-only, documented"),
                    ("frontier", ">=1 control edit (dominates)"),
                    ("speculative", "de-novo C-MeT / shortened chain / too far")]:
        print(f"  {k:13s} ({note:30s}) {len(ds[k]):10d} {distinct_products(ds[k]):18d}")
    print("\n  -> for fungal iterative PKS the *control* axis dominates the design space: biosynthetic")
    print("     design here is mostly iteration-program editing, the axis only this engine can express.\n")

    print(f"{'designed target':40s} {'(S,C)':>7s} {'verdict':12s} nearest natural cluster")
    print("-" * 92)
    for label, prog in EXAMPLES:
        r = realize(prog, man)
        sc = f"({r.structural_cost},{r.control_cost})"
        print(f"{label:40s} {sc:>7s} {r.verdict:12s} {r.nearest_id}")
        s = "; ".join(e.detail for e in r.structural_edits) or "-"
        c = "; ".join(e.detail for e in r.control_edits) or "-"
        print(f"      structural: {s}   |   control: {c}")


if __name__ == "__main__":
    main()
