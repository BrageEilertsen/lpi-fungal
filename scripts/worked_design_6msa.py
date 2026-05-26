"""Worked design example: the 1-edit neighbourhood of 6-MSA, end to end.

One concrete walk through the whole machine for a single, clean fungal template (6-MSA synthase,
the cluster with a verified blind real-mass match). For each design in 6-MSA's 1-edit neighbourhood
we report, ranked most-realizable first:

  * the (structural, control) edit decomposition and realizability verdict (design distance);
  * the executed structure and predicted monoisotopic mass (the executor);
  * the observable that would VERIFY the design if synthesized -- mass alone if the design's own
    grammar + mass pins it to a unique core, otherwise MS/MS (the observability organ).

So one row reads: "this molecule is N documented + M iteration-program edits from 6-MSA, has this
predicted mass, and is verifiable by <observable>." That is a design proposal a wet-lab director can
act on -- design distance and verification plan in the same line.
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.engine import State
from lpi.executor import core
from lpi.grammars import PKS
from lpi.observe import MSObservables, exact_mass, infer_ms
from lpi.realizability import (_target_domains, cost, natural_manifold, one_edit_neighbours,
                               realize)

RDLogger.DisableLog("rdApp.*")

_MAN = natural_manifold()
_6MSA = next(t for t in _MAN if "6-MSA" in t.id)


def _structure(prog):
    try:
        return M.canonical_smiles(core.exec(prog))
    except Exception:  # noqa: BLE001
        return None


def _verifying_observable(prog, smiles) -> str:
    """Observable that would confirm this design if synthesized: mass alone if the design's grammar
    + mass pins a unique core, else MS/MS (run the observability organ on the design's own mass)."""
    domains = {"KS", "AT", "ACP"} | set(_target_domains(prog))
    alpha = PKS.alphabet_from_domains(domains, starters=(prog.starter,))  # the design's own starter
    n = len(prog.cycles)
    cho = exact_mass(smiles)
    res = infer_ms(alpha, MSObservables(neutral_mass=cho, ppm=5), PKS, max(1, n - 1), n + 1,
                   ladder=False)
    if res.state is State.VERIFIED:
        return "accurate mass alone"
    if res.state is State.UNDER_OBSERVED:
        return "+ MS/MS"
    return "out-of-grammar"


_HDR = f"  {'(S,C)':>6s}  {'mass':>9s}  {'verifies by':19s} edit"


def _print_row(r, prog, smi, ver) -> None:
    sc = f"({r.structural_cost},{r.control_cost})"
    edit = "; ".join(e.detail for e in r.edits) or "-"
    print(f"  {sc:>6s}  {exact_mass(smi):9.3f}  {ver:19s} {edit}")
    print(f"  {'':40s}{smi}")


def main() -> None:
    print("Worked design example: the 1-edit neighbourhood of 6-MSA (MIBiG BGC0001275)\n")
    print("Two design classes share one realizability coordinate.  (S,C) = # structural, # control edits.")
    print("  STRUCTURAL = domain-content swap with documented bacterial-PKS precedent      -> buildable.")
    print("  CONTROL    = iteration-program edit (which cycle fires; the chain length)     -> the frontier,")
    print("               the axis only a sound per-cycle executor can even express.")

    eng, fro, spec = [], [], []
    for prog in one_edit_neighbours(_6MSA):
        smi = _structure(prog)
        if smi is None:
            continue
        r = realize(prog, _MAN)
        if r.verdict == "natural":
            continue
        ver = _verifying_observable(prog, smi)
        (eng if r.verdict == "engineerable" else fro if r.verdict == "frontier" else spec).append(
            (r, prog, smi, ver))
    # within a class, lead with designs the executor can actually verify; coverage-limited
    # out-of-grammar designs sink (realizable but not yet executor-reachable -- see caveats).
    for bucket in (eng, fro, spec):
        bucket.sort(key=lambda t: (t[3] == "out-of-grammar", cost(t[0].edits), t[0].control_cost))

    print("\n== ENGINEERABLE -- structural-only, documented PKS engineering (buildable-now analogs) ==")
    print(_HDR)
    for r, prog, smi, ver in eng[:6]:
        _print_row(r, prog, smi, ver)
    print("\n== FRONTIER -- >=1 control edit, iteration-program editing (only this engine can express it) ==")
    print(_HDR)
    for r, prog, smi, ver in fro[:6]:
        _print_row(r, prog, smi, ver)

    print("\nRead the two classes straight off the rows: (1,0) acetyl->hexanoyl is a documented domain swap")
    print("a wet-lab builds today; (0,1) cyc3 keto->kr reprograms which cycle reduces -- the frontier.")
    print(f"Across the neighbourhood: {len(eng)} engineerable, {len(fro)} frontier, {len(spec)} speculative "
          "(de-novo C-MeT / shortened chain). Control dominates.")
    print("\nHonest caveats (realizability != current executor coverage): AT extender-swap designs verify")
    print("'out-of-grammar' (methylmalonyl is a bacterial-AT feature the fungal executor does not enumerate);")
    print("starter-swap analogs render as the linear acid (aromatic closure implemented only for the acetyl")
    print("tetraketide). Both are flagged coverage items, not realizability claims.")


if __name__ == "__main__":
    main()
