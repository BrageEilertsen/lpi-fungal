"""Single-mode terminal cyclization (lactonization + non-PT aromatic aldol).

Scope (per the approved Phase 0 plan): one unambiguous cyclization mode per release
type, no regiochemistry choice. This covers the small aromatic/lactonizing fungal
PKS (orsellinic acid, 6-MSA, mellein, triacetic-acid lactone). The five-class
PT-domain regioselectivity problem of the large NR-PKS (norsolorinic acid / aflatoxin
class) is deliberately deferred to :mod:`lpi.executor.aromatic` / Phase 0b.

These operators act on the *released linear acid* (post TE-hydrolysis), so the linear
checkpoint is always available upstream regardless of whether cyclization succeeds.
A best-effort design: if the linear chain does not match a known cyclization motif,
:func:`release` returns ``None`` rather than guessing.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from lpi.chem.program import Release

# Orsellinic-type C2->C7 aldol condensation + aromatization on a non-reduced
# tetraketide acid: HOOC-CH2-CO-CH2-CO-CH2-CO-CH3 -> 2,4-dihydroxy-6-methylbenzoic
# acid (orsellinic acid). One net dehydration; C7 keto-oxygen is the leaving water.
_ORSELLINIC_ALDOL = (
    "[OH][C:1](=[O:10])[CH2:2][C:3](=[O:11])[CH2:4][C:5](=[O:12])[CH2:6][C:7](=O)[CH3:8]"
    ">>[OH][C:1](=[O:10])[c:2]1[c:3]([OH:11])[cH][c:5]([OH:12])[cH][c:7]1[CH3:8]"
)

# 6-MSA-type aldol on the singly-reduced tetraketide acid (one KR+DH leaves a C4=C5
# enoyl): HOOC-CH2-CO-CH=CH-CH2-CO-CH3 -> 2-hydroxy-6-methylbenzoic acid (6-MSA).
# The reduced position is why 6-MSA lacks orsellinic's 4-OH.
_6MSA_ALDOL = (
    "[OH][C:1](=[O:9])[CH2:2][C:3](=[O:10])[CH:4]=[CH:5][CH2:6][C:7](=O)[CH3:8]"
    ">>[OH][C:1](=[O:9])[c:2]1[c:3]([OH:10])[cH:4][cH:5][cH:6][c:7]1[CH3:8]"
)

# Aromatic single-mode templates tried in order; they are substrate-disjoint (the
# orsellinic motif requires three intact keto groups, the 6-MSA motif requires the
# enoyl), so at most one fires for a given linear chain.
_AROMATIC_TEMPLATES = (_ORSELLINIC_ALDOL, _6MSA_ALDOL)

# Triacetic-acid-lactone (TAL) lactonization on a non-reduced triketide acid:
# HOOC-CH2-CO-CH2-CO-CH3 -> 4-hydroxy-6-methyl-2H-pyran-2-one. The C5 enol oxygen
# attacks the acid carbonyl forming the 6-membered lactone; one dehydration.
_TAL_LACTONE = (
    "[OH][C:1](=[O:9])[CH2:2][C:3](=[O:10])[CH2:4][C:5](=O)[CH3:6]"
    ">>[O:9]=[C:1]1[CH:2]=[C:3]([OH:10])[CH:4]=[C:5]([CH3:6])[O:11]1"
)


def _run_single(smarts: str, mol: Chem.Mol) -> Chem.Mol | None:
    rxn = AllChem.ReactionFromSmarts(smarts)
    rxn.Initialize()
    outcomes = rxn.RunReactants((mol,))
    products: dict[str, Chem.Mol] = {}
    for (prod,) in outcomes:
        for atom in prod.GetAtoms():
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        try:
            Chem.SanitizeMol(prod)
        except Exception:  # noqa: BLE001
            continue
        products[Chem.MolToSmiles(prod)] = prod
    if len(products) == 1:
        return next(iter(products.values()))
    return None  # no match, or ambiguous -> do not guess


def lactonize(linear_acid: Chem.Mol) -> Chem.Mol | None:
    return _run_single(_TAL_LACTONE, linear_acid)


def aldol_aromatic(linear_acid: Chem.Mol) -> Chem.Mol | None:
    products: dict[str, Chem.Mol] = {}
    for smarts in _AROMATIC_TEMPLATES:
        prod = _run_single(smarts, linear_acid)
        if prod is not None:
            products[Chem.MolToSmiles(prod)] = prod
    if len(products) == 1:
        return next(iter(products.values()))
    return None  # no template matched, or conflicting matches


def release(linear_acid: Chem.Mol, mode: Release) -> Chem.Mol | None:
    if mode is Release.LACTONIZATION:
        return lactonize(linear_acid)
    if mode is Release.ALDOL_AROMATIC:
        return aldol_aromatic(linear_acid)
    return None
