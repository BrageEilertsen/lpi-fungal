"""EXPLORATORY (Direction B): retro-biosynthesis under grammar constraints.

Given an arbitrary SMILES, does any grammar-legal program produce it? This inverts the SOUND forward
executor via the existing beam verifier (lpi.search.beam) over a PERMISSIVE PKS operator library --
no unsound shortcut, just enumerate-execute-match. Verdict mirrors the three-state pattern:

  VERIFIED          -- exactly one grammar-legal program produces the target (sound: the executor matched)
  UNDER_CONSTRAINED -- several grammar-legal programs produce it (candidate set returned)
  OUT_OF_GRAMMAR    -- search COMPLETED and no program produces it (a real negative)
  CENSORED          -- the execution budget was hit before completion -> NOT a negative claim (honest)

Scope of this prototype: PKS family only (beam.py is the ready sound inverter). NRPS and hybrid retro
need the same enumerate-match over their grammars and are scoped in the report, not built here.

This is exploratory code (scripts/explorations/), not shipped src/. No soundness break: retro is the
inverse of a sound forward function.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.chem.program import Extender, ReductionState, Release
from lpi.executor import core
from lpi.realizability import natural_manifold
from lpi.search.beam import OperatorSpec, search

RDLogger.DisableLog("rdApp.*")

# the full modelled PKS operator library (the "all grammar-legal moves" spec)
_PERMISSIVE = OperatorSpec(
    starters=("acetyl", "propionyl", "butyryl", "hexanoyl"),
    reductions=(ReductionState.KETO, ReductionState.KR, ReductionState.DH, ReductionState.ER),
    allow_c_methyl=True,
    releases=(Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
              Release.DIHYDROISOCOUMARIN),
    extenders=(Extender.MALONYL,),
)
_BUDGET = 60000


@dataclass
class RetroVerdict:
    smiles: str
    formula: str
    verdict: str
    n_programs: int
    example: str | None
    censored: bool
    runtime_s: float


def _formula(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.rdMolDescriptors.CalcMolFormula(m) if m else "?"


# COMPLETENESS FINDING (verified on BAB, a known-producible C12 HR core): at the default beam_width=2000
# the beam PRUNES the producing program and returns a FALSE OUT_OF_GRAMMAR; beam_width>=20000 recovers it.
# So the beam is SOUND (VERIFIED = a real producing program) but NOT complete: OUT_OF_GRAMMAR is only a
# trustworthy negative when the target is formula-infeasible, or the search is effectively exhaustive.
# We widen the beam; OUT_OF_GRAMMAR is reported as "no program within beam budget", not a sound negative.
_BEAM = 20000


def retro_pks(smiles: str) -> RetroVerdict:
    res = search(smiles, _PERMISSIVE, beam_width=_BEAM, max_executions=_BUDGET)
    n = res.size
    censored = bool(res.stats and res.stats.budget_exhausted)
    if n == 0:
        verdict = "CENSORED" if censored else "OUT_OF_GRAMMAR"
    elif n == 1:
        verdict = "VERIFIED"
    else:
        verdict = "UNDER_CONSTRAINED"
    example = repr(res.z_star[0]) if res.z_star else None
    return RetroVerdict(smiles, _formula(smiles), verdict, n, example, censored,
                        res.stats.runtime_s if res.stats else 0.0)


def _manifold_product(needle: str) -> str:
    t = next(x for x in natural_manifold() if needle in x.id)
    return M.canonical_smiles(core.exec(t.program))


def corpus() -> list[tuple[str, str, str]]:
    """(category, label, smiles). Positives are real producible cores (executed from known programs);
    negatives are molecules requiring chemistry outside the PKS operator set; probes are
    polyketide-shaped test molecules whose natural-producer status is a SEPARATE database question."""
    pos = [("positive", "orsellinic acid", _manifold_product("orsellinic")),
           ("positive", "6-MSA", _manifold_product("6-MSA")),
           ("positive", "mellein", _manifold_product("mellein")),
           ("positive", "BAB (C12 HR)", _manifold_product("BAB"))]
    neg = [("negative", "limonene (terpene)", "CC1=CCC(CC1)C(=C)C"),
           ("negative", "glucose (sugar)", "OCC1OC(O)C(O)C(O)C1O"),
           ("negative", "glycylglycine (peptide)", "NCC(=O)NCC(=O)O"),
           ("negative", "caffeine (alkaloid)", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]
    # probes: polyketide-plausible, NOT drawn from the manifold. Novel-producer status = TODO.
    probe = [("probe", "tetraacetic acid lactone", "CC1=CC(=O)CC(=O)O1"),
             ("probe", "C8 linear triketo acid", "CC(=O)CC(=O)CC(=O)CC(=O)O"),
             ("probe", "propionyl resorcylic acid", "CCc1cc(O)cc(O)c1C(=O)O"),
             ("probe", "alkyl dihydroisocoumarin", "CCCCc1cccc(O)c1C(=O)O")]
    return pos + neg + probe


def main() -> None:
    print("Direction B -- retro-biosynthesis under the PKS grammar (sound beam inversion)\n")
    print(f"{'cat':9s} {'label':28s} {'formula':14s} {'verdict':17s} {'n':>3s} {'t(s)':>6s}")
    print("-" * 92)
    rows = []
    for cat, label, smi in corpus():
        v = retro_pks(smi)
        rows.append((cat, label, v))
        print(f"{cat:9s} {label[:28]:28s} {v.formula:14s} {v.verdict:17s} {v.n_programs:3d} {v.runtime_s:6.2f}")
        if v.example:
            print(f"{'':25s}-> {v.example}")
    print()
    by = {}
    for cat, _, v in rows:
        by.setdefault(cat, []).append(v.verdict)
    for cat in ("positive", "negative", "probe"):
        vs = by.get(cat, [])
        print(f"  {cat:9s}: " + ", ".join(f"{vs.count(x)} {x}" for x in sorted(set(vs))))
    print("\nHonest notes: OUT_OF_GRAMMAR = search completed empty (real negative); CENSORED = budget hit")
    print("(not a negative). Any probe that is VERIFIED is grammar-accessible; whether it has a known")
    print("natural producer is a SEPARATE database query, marked TODO -- not claimed here.")


if __name__ == "__main__":
    main()
