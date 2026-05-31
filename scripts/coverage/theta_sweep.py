"""Theta-sweep reproduction: coverage-flow transition matrices + R_terminal climb.

Paper Sec. 16.2a / 16.4a; companion Results rows 2-4. Reuses the committed reachability scan
(``lpi.search.reachability.scan_parquet``) under varied operator libraries Theta, so the per-core
partition logic is byte-identical to ``make reachability`` -- only Theta changes.

Reproduces (each headline printed with its expected value):
  * starter sweep {acetyl} -> {acetyl,propionyl,hexanoyl}:   |R| 9->8, reach 9->9, one R->U demotion (olivetolic)
  * release sweep aromatic-only -> all-four cyclizations:    |R| 6->9, reach 6->9, X->R=3, 0 demotions
  * R_terminal, 4-starter universe REL0 -> REL0+PT closure:  R_term 8->11, reach 9->12, X->R=3, 0 demotions

In every transition the Lemma-16.3 forbidden cells (R->X, U->R, U->X) must be empty: a nonzero entry
there is not a counterexample to the lemmas but an enumeration/beam artifact (the consistency check of
Sec. 16.2a). The R_terminal scan runs the heavier beam (8000/200000) of the climb; the two reach-level
sweeps run the base-scan beam (4000/60000).
"""
from __future__ import annotations

import collections

from lpi.chem.program import Extender, ReductionState, Release
from lpi.data.mibig import PROCESSED
from lpi.search.beam import OperatorSpec
from lpi.search.reachability import scan_parquet

PARQUET = PROCESSED / "fungal_pks_pairs.parquet"
ALLRED = tuple(ReductionState)
REL0 = (Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN)
RELP = REL0 + (Release.PT_NAPHTHALENE,)
BASE3 = ("acetyl", "propionyl", "hexanoyl")            # base reachability starters (Theta_0)
FULL4 = ("acetyl", "propionyl", "butyryl", "hexanoyl")  # R_terminal universe (adds butyryl)


def uni(starters, releases, extenders=(Extender.MALONYL,)) -> OperatorSpec:
    return OperatorSpec(starters=starters, reductions=ALLRED, allow_c_methyl=True,
                        releases=releases, extenders=extenders, max_cycles=None)


def _state(k: int) -> str:
    return "R" if k == 1 else ("U" if k >= 2 else "X")


def kappa_map(rep) -> dict[str, tuple[int, str]]:
    """bgc_id -> (kappa, name); kappa = z_star_size if reachable else 0 (the partition input)."""
    return {r.bgc_id: (r.z_star_size if r.status == "reachable" else 0, r.name) for r in rep.rows}


def partition(km) -> tuple[int, int, int, int]:
    c = collections.Counter(_state(k) for k, _ in km.values())
    return c["R"], c["U"], c["X"], c["R"] + c["U"]      # R, U, X, reach


def transition(kmA, kmB):
    T = collections.Counter()
    moved = collections.defaultdict(list)
    for b in kmA:
        sA, sB = _state(kmA[b][0]), _state(kmB[b][0])
        T[(sA, sB)] += 1
        if sA != sB:
            moved[(sA, sB)].append((b, kmA[b][1]))
    return T, moved


def report(title, kmA, kmB, expect):
    RA, UA, XA, reachA = partition(kmA)
    RB, UB, XB, reachB = partition(kmB)
    T, moved = transition(kmA, kmB)
    print(f"\n=== {title} ===")
    print(f"  A: R={RA} U={UA} X={XA} reach={reachA}     B: R={RB} U={UB} X={XB} reach={reachB}")
    print("  transition matrix (row=state under A, col=state under B):")
    for sA in "RUX":
        print("    " + sA + " -> " + "   ".join(f"{sB}:{T[(sA, sB)]}" for sB in "RUX"))
    forbidden = {k: T[k] for k in [("R", "X"), ("U", "R"), ("U", "X")] if T[k]}
    print(f"  Lemma-16.3 forbidden cells empty? {'YES' if not forbidden else f'NO -> {forbidden} (enum/beam artifact)'}")
    for cell, items in sorted(moved.items()):
        print(f"    {cell[0]}->{cell[1]} ({len(items)}): " + ", ".join(f"{b}:{n}" for b, n in items))
    ok_all = True
    for label, got, exp in expect:
        ok = got == exp
        ok_all &= ok
        print(f"  [{'OK' if ok else 'XX'}] {label}: got {got}, expect {exp}")
    return ok_all and not forbidden


def main() -> None:
    print("Theta-sweep reproduction (Sec 16.2a / 16.4a) -- reusing lpi.search.reachability.scan_parquet")
    results = []

    print("\n[1/3] starter sweep (base beam 4000/60000)...", flush=True)
    a = kappa_map(scan_parquet(PARQUET, spec=uni(("acetyl",), REL0)))
    b = kappa_map(scan_parquet(PARQUET, spec=uni(BASE3, REL0)))
    results.append(report(
        "Starter sweep {acetyl} -> {acetyl,propionyl,hexanoyl}  (releases = REL0)", a, b,
        [("|R| A", partition(a)[0], 9), ("|R| B", partition(b)[0], 8),
         ("reach A", partition(a)[3], 9), ("reach B", partition(b)[3], 9),
         ("R->U demotions", transition(a, b)[0][("R", "U")], 1)]))

    print("\n[2/3] release sweep (base beam 4000/60000)...", flush=True)
    a = kappa_map(scan_parquet(PARQUET, spec=uni(("acetyl",), (Release.ALDOL_AROMATIC,))))
    b = kappa_map(scan_parquet(PARQUET, spec=uni(("acetyl",), REL0)))
    results.append(report(
        "Release sweep aromatic-only -> all-four  (starter = acetyl)", a, b,
        [("|R| A", partition(a)[0], 6), ("|R| B", partition(b)[0], 9),
         ("reach A", partition(a)[3], 6), ("reach B", partition(b)[3], 9),
         ("X->R gains", transition(a, b)[0][("X", "R")], 3),
         ("R->U demotions", transition(a, b)[0][("R", "U")], 0)]))

    print("\n[3/3] R_terminal climb, 4-starter universe REL0 -> REL0+PT (heavy beam 8000/200000)...", flush=True)
    a = kappa_map(scan_parquet(PARQUET, spec=uni(FULL4, REL0), beam_width=8000, max_executions=200000))
    b = kappa_map(scan_parquet(PARQUET, spec=uni(FULL4, RELP), beam_width=8000, max_executions=200000))
    # R_terminal under universe U == #{kappa==1 under U} == the R cell of the partition under U.
    results.append(report(
        "R_terminal: 4-starter universe  REL0 -> REL0 + PT_NAPHTHALENE", a, b,
        [("R_term A (REL0)", partition(a)[0], 8), ("R_term B (+PT)", partition(b)[0], 11),
         ("reach A", partition(a)[3], 9), ("reach B", partition(b)[3], 12),
         ("X->R gains", transition(a, b)[0][("X", "R")], 3),
         ("R->U demotions", transition(a, b)[0][("R", "U")], 0)]))

    print(f"\n{'='*60}\nALL SWEEPS REPRODUCE: {'YES' if all(results) else 'NO -- see XX rows above'}")


if __name__ == "__main__":
    main()
