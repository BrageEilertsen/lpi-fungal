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


def main() -> None:
    print("Worked design example: the 1-edit neighbourhood of 6-MSA (MIBiG BGC0001275)\n")
    rows = []
    for prog in one_edit_neighbours(_6MSA):
        smi = _structure(prog)
        if smi is None:
            continue
        r = realize(prog, _MAN)
        if r.verdict == "natural":
            continue
        rows.append((r, prog, smi))
    rows.sort(key=lambda t: (cost(t[0].edits), t[0].control_cost))

    print(f"{'(S,C)':>6s} {'verdict':12s} {'mass':>9s}  {'verifies by':19s} edit / structure")
    print("-" * 100)
    for r, prog, smi in rows[:18]:
        sc = f"({r.structural_cost},{r.control_cost})"
        edit = "; ".join(e.detail for e in r.edits) or "-"
        ver = _verifying_observable(prog, smi)
        print(f"{sc:>6s} {r.verdict:12s} {exact_mass(smi):9.3f}  {ver:19s} {edit}")
        print(f"{'':51s}{smi}")
    print(f"\n{len(rows)} non-natural designs in the 1-edit neighbourhood; showing the 18 most realizable.")
    print("Each row is a build proposal: design distance (structural/control edits from a real")
    print("cluster) + predicted mass + the observable that confirms it. 2-edit neighbours extend this.")
    print("\nHonest caveats (realizability != current executor coverage): AT extender-swap designs")
    print("verify 'out-of-grammar' because methylmalonyl is a bacterial-AT feature the fungal executor")
    print("does not enumerate; and starter-swap analogs render as the linear acid where the aromatic")
    print("closure is implemented only for the acetyl tetraketide -- both are flagged coverage items.")


if __name__ == "__main__":
    main()
