"""Prop 16.4c reproduction: kappa-monotonicity as a runnable relative-completeness certificate.

Paper Sec. 16.4b; companion Results row 5. By Lemma 16.3, enriching Theta cannot DECREASE kappa
(programs are only added). The beam verifier can only under-count (kappa_beta <= kappa), so a reported
*decrease* across nested Theta proves the larger search pruned a valid program -- a sound, zero-false-
alarm certificate of beam inadequacy, localized to the larger operator set.

Witness: adding the methylmalonyl extender (Theta_A = {malonyl} subset Theta_B = {malonyl,methylmalonyl})
at beam=8000 inverts kappa on olivetolic (3->2) and BAB (1->0) -- both flagged incomplete; widening to
beam=50000 restores monotonicity, confirming R_terminal=8 under the extended universe (methylmalonyl is
redundant: the methyls are starter/cyclization-derived, not methylmalonyl-derived -- the methyl-source
hypothesis refuted). Reuses the committed reachability scan; the flagged cores are re-searched directly.
"""
from __future__ import annotations

from lpi.chem import mol as M
from lpi.chem.program import Extender, ReductionState, Release
from lpi.data.mibig import PROCESSED
from lpi.search.beam import OperatorSpec, search
from lpi.search.reachability import scan_parquet

PARQUET = PROCESSED / "fungal_pks_pairs.parquet"
ALLRED = tuple(ReductionState)
REL0 = (Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN)
FULL4 = ("acetyl", "propionyl", "butyryl", "hexanoyl")


def uni(extenders) -> OperatorSpec:
    return OperatorSpec(starters=FULL4, reductions=ALLRED, allow_c_methyl=True,
                        releases=REL0, extenders=extenders, max_cycles=None)


THETA_A = uni((Extender.MALONYL,))                          # malonyl only
THETA_B = uni((Extender.MALONYL, Extender.METHYLMALONYL))   # + methylmalonyl  (Theta_A subset Theta_B)


def kmap(rep):
    return {r.bgc_id: (r.z_star_size if r.status == "reachable" else 0, r.name, r.smiles) for r in rep.rows}


def main() -> None:
    print("Prop 16.4c: kappa-monotonicity completeness certificate (methylmalonyl extender)\n")
    print("Scanning all cores under Theta_A (malonyl) and Theta_B (+methylmalonyl) at beam=8000 ...", flush=True)
    a = kmap(scan_parquet(PARQUET, spec=THETA_A, beam_width=8000, max_executions=200000))
    b = kmap(scan_parquet(PARQUET, spec=THETA_B, beam_width=8000, max_executions=200000))

    inversions = []
    print(f"\n  {'core (bgc_id)':18s} {'name':18s} {'kA(mal)':>8s} {'kB(+mmal)':>10s}  Lemma 16.3")
    for bid in a:
        kA, name, _ = a[bid]
        kB = b[bid][0]
        if kA == 0 and kB == 0:
            continue
        violation = kB < kA
        tag = "VIOLATION -> Theta_B beam-incomplete (Prop 16.4c)" if violation else "OK (kB >= kA)"
        print(f"  {bid:18s} {str(name)[:18]:18s} {kA:>8d} {kB:>10d}  {tag}")
        if violation:
            inversions.append((bid, name, a[bid][2], kA, kB))

    print(f"\n  inversions at beam=8000: {len(inversions)}  (expect 2 -- olivetolic 3->2, BAB 1->0)")
    print("\nRe-searching flagged cores at beam=50000 (max_exec=1e6) -- expect monotonicity restored:")
    restored = 0
    for bid, name, smi, kA, kB8 in inversions:
        kB50 = search(M.canonical_smiles(M.mol_from_smiles(smi)), THETA_B,
                      beam_width=50000, max_executions=1_000_000).size
        ok = kB50 >= kA
        restored += ok
        print(f"  [{'OK' if ok else 'XX'}] {bid} ({name}): kA={kA}  kB@8000={kB8}  kB@50000={kB50}  "
              f"-> {'RESTORED (kB>=kA)' if ok else 'STILL LOW'}")

    ok_all = len(inversions) == 2 and restored == len(inversions)
    print(f"\n{'='*60}\nProp 16.4c REPRODUCES (2 inversions flagged @8000, both restored @50000): "
          f"{'YES' if ok_all else 'NO -- see rows above'}")


if __name__ == "__main__":
    main()
