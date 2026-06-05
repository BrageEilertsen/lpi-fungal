"""Position-sound (atom-placed) post-PKS tailoring -- the refinement beyond the formula-budget layer
(``tailoring.py`` / ``search/joint.py``, which place no atoms; cf. ``joint.py`` "positions are the next
refinement"). Each operator is CURATION-CONDITIONAL in the sense of S4.1: a deposited witness pins the edit
POSITION, encoded as a curated SMARTS register. The executor places the core atoms; this places the
decoration, by a deterministic RDKit graph op at the register-matched site -- no formula-budget shortcut.
The decorated product is checked against the deposit by the SAME flat-canonical ~= as the core map, so a
match is position-sound, not exact-by-budget.

Honesty typing (kept apart, deliberately):
  * The verdict is "verified RELATIVE TO the methylation register" -- the position is curated (witness-
    pinned), NOT observation-determined. Same honesty category as the curated PT aromatic folds (S4.1).
  * Whether the register is PROSPECTIVELY predictable -- substrate-directed (rung-2) vs enzyme-gated
    (rung-3-curated) -- is a SEPARATE tag, settled by a second witness: the tailoring enzyme's
    characterization literature. It LABELS the operator; it does NOT gate the build.

Soundness guards: the register must pin EXACTLY ONE site (ambiguity is refused, not guessed); and the
applied edit's formula delta must equal the declared ``tailoring.EDIT_TYPES`` delta (a cross-check that the
graph op did what its type claims).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import RWMol

from lpi.executor.tailoring import BY_NAME


@dataclass(frozen=True)
class CuratedEdit:
    name: str
    edit_type: str          # key into tailoring.BY_NAME (its formula delta + enzyme family)
    register_smarts: str    # curated site selector; the edit acts on the FIRST matched atom
    witness_bgc: str        # the deposited structure that pins the position
    rung: str               # "rung-3-curated" until the enzyme literature promotes it to "rung-2"
    note: str = ""


def _formula(mol: Chem.Mol) -> Counter:
    c: Counter = Counter()
    for a in mol.GetAtoms():
        c[a.GetSymbol()] += 1
        c["H"] += a.GetTotalNumHs()
    return c


def _site_atom(mol: Chem.Mol, register_smarts: str) -> int:
    """Index of the unique register-matched edit site. Raises if the register is absent OR ambiguous --
    ambiguity means the curated register does not pin one position, so we refuse rather than guess
    (the guardrail against smuggling a fold-determined position into a position-sound operator)."""
    patt = Chem.MolFromSmarts(register_smarts)
    if patt is None:
        raise ValueError(f"bad register SMARTS: {register_smarts!r}")
    sites = {m[0] for m in mol.GetSubstructMatches(patt)}
    if not sites:
        raise ValueError("curated register did not match -- no site to edit")
    if len(sites) > 1:
        raise ValueError(f"curated register is AMBIGUOUS ({len(sites)} sites) -- it does not pin one "
                         "position; refuse rather than guess")
    return next(iter(sites))


def apply_o_methylation(mol: Chem.Mol, register_smarts: str) -> Chem.Mol:
    """Position-sound O-methylation (O-H -> O-CH3) at the unique register-matched oxygen. Deterministic."""
    o_idx = _site_atom(mol, register_smarts)
    if mol.GetAtomWithIdx(o_idx).GetSymbol() != "O":
        raise ValueError("register site is not an oxygen")
    rw = RWMol(mol)
    c_idx = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(o_idx, c_idx, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


_APPLY = {"o_methylation": apply_o_methylation}


def apply_curated_edit(mol: Chem.Mol, edit: CuratedEdit) -> Chem.Mol:
    """Apply a curated, position-sound edit and cross-check its formula delta against the declared type."""
    if edit.edit_type not in _APPLY:
        raise NotImplementedError(f"no position-sound applier for edit_type {edit.edit_type!r}")
    before = _formula(mol)
    out = _APPLY[edit.edit_type](mol, edit.register_smarts)
    after = _formula(out)
    got = {el: after.get(el, 0) - before.get(el, 0)
           for el in set(before) | set(after) if after.get(el, 0) - before.get(el, 0)}
    expect = {el: n for el, n in BY_NAME[edit.edit_type].delta.items() if n}
    if got != expect:
        raise ValueError(f"{edit.name}: applied formula delta {got} != declared {expect} "
                         f"for edit_type {edit.edit_type!r}")
    return out


# ---- curated registers (witness-pinned) -------------------------------------------------------------

LASIODIPLODIN_OME = CuratedEdit(
    name="resorcylic_ortho_ester_O_methylation",
    edit_type="o_methylation",
    register_smarts="[OX2H1]-c:c-[CX3](=O)[OX2]",   # resorcinol OH ORTHO to the macrolactone ester carbonyl
    witness_bgc="BGC0001245",                         # deposited lasiodiplodin pins this position
    rung="rung-3-curated",
    note="Position curated from deposited lasiodiplodin, NOT observation-determined. The ortho-OH is "
         "H-bonded (chelated) to the ester carbonyl, so naive substrate electronics would favour "
         "methylating the FREE para-OH; the deposited ortho-methylation is therefore at least as "
         "consistent with the O-MT overriding substrate preference. Stays rung-3-curated until the "
         "lasiodiplodin O-MT characterization literature settles substrate-directed (-> rung-2) vs "
         "enzyme-gated/silent (stays rung-3-curated).",
)
