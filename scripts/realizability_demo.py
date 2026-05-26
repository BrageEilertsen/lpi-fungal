"""Design-by-grammar-inversion demo: a distance-to-realizable coordinate for every target.

For a set of target programs -- natural products and non-natural designed analogs -- the engine
returns the nearest natural cluster, the typed edit path (with class-level feasibility tiers), a
realizability verdict, and the executed structure. This is the third direction: not "reconstruct a
known molecule" but "what is the smallest documented-edit path from an existing cluster to a desired
one." The same sound executor that verifies natural products verifies the designs.

Honest scope (see lpi.realizability): the natural manifold here is the small reachable set, and the
edit tiers are CLASS-level (no fabricated paper citations). The machinery is exact; whether the
manifold is dense enough to give meaningful distances for arbitrary drug targets is the open
question this metric makes cheaply testable.
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.executor import core
from lpi.observe import exact_mass
from lpi.realizability import natural_manifold, realize

RDLogger.DisableLog("rdApp.*")

man = natural_manifold()

# (label, target program). Naturals have distance 0; the rest are non-natural designed analogs.
TARGETS = [
    ("6-MSA (natural)", man["BGC0001275 6-MSA"]),
    ("orsellinic (natural)", man["BGC0001121 orsellinic"]),
    ("design: 6-MSA + KR@cyc3 (reduced analog)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), Release.ALDOL_AROMATIC)),
    ("design: orsellinic + C-MeT@cyc2 (methylated)",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KETO, c_methyl=True), Cycle(R.KETO)),
             Release.ALDOL_AROMATIC)),
    ("design: 6-MSA -> lactone release",
     Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), Release.LACTONIZATION)),
    ("design: propionyl-started reduced hexaketide",
     Program("propionyl", (Cycle(R.ER), Cycle(R.DH), Cycle(R.KR), Cycle(R.KETO), Cycle(R.ER)),
             Release.HYDROLYSIS)),
]


def _structure(prog: Program) -> tuple[str, float]:
    try:
        m = core.exec(prog)
        return M.canonical_smiles(m), exact_mass(M.canonical_smiles(m))
    except Exception:  # noqa: BLE001
        return "(executor: cyclization did not fire)", float("nan")


def main() -> None:
    print("Design-by-grammar-inversion: distance-to-realizable over the natural-cluster manifold\n")
    print(f"{'target':44s} {'verdict':13s} {'edits':>5s}  nearest natural cluster")
    print("-" * 96)
    for label, prog in TARGETS:
        r = realize(prog, man)
        print(f"{label:44s} {r.verdict:13s} {r.cost:5d}  {r.nearest_id}")
        if r.edits:
            print("      path: " + "; ".join(f"{e.detail} [T{int(e.tier)}:{e.tier.name.lower()}]"
                                              for e in r.edits))
        smi, mass = _structure(prog)
        print(f"      -> {smi}" + (f"  (m={mass:.3f})" if mass == mass else ""))
    print("\nReading: every target gets a coordinate -- natural (cost 0), engineerable (a few")
    print("documented/plausible edits from a real cluster), or speculative (an undocumented release")
    print("reprogramming, or too many edits). Edit tiers are class-level; specific directed-evolution")
    print("citations and the full MIBiG/ClusterCAD manifold are the flagged next data step.")


if __name__ == "__main__":
    main()
