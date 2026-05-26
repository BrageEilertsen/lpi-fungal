"""Single-mode terminal cyclization (lactonization + single-mode aromatic aldol).

Scope (per the approved Phase 0 plan): one unambiguous cyclization mode per release
type, no regiochemistry choice. This covers the small aromatic/lactonizing fungal
PKS (orsellinic acid, 6-MSA, mellein, triacetic-acid lactone).

Note on PT domains (Brage's correction): OrsA *does* carry a PT domain -- the point is
that at tetraketide size the PT has only ONE productive cyclization mode, so there is no
regiochemical ambiguity to model. The five-class PT regioselectivity problem is real only
for the larger NR-PKS (hexa-/hepta-/octaketides, e.g. norsolorinic acid / aflatoxin),
which is deferred to :mod:`lpi.executor.aromatic` / Phase 0b.

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
# Generalised to any starter-derived terminal alkyl: R = methyl gives orsellinic acid,
# R = pentyl gives olivetolic acid, etc. The C2->C7 aldol/aromatization is independent of
# what hangs off C7's neighbour, so [CX4:8] matches a CH3 or a longer chain and carries it
# through unchanged to the ring 6-substituent.
_ORSELLINIC_ALDOL = (
    "[OH][C:1](=[O:10])[CH2:2][C:3](=[O:11])[CH2:4][C:5](=[O:12])[CH2:6][C:7](=O)[CX4:8]"
    ">>[OH][C:1](=[O:10])[c:2]1[c:3]([OH:11])[cH][c:5]([OH:12])[cH][c:7]1[C:8]"
)

# 6-MSA-type aldol on the singly-reduced tetraketide acid. Per Brage: encode the single
# reduction as KR only (a beta-hydroxyl), NOT a programmed DH -- the dehydration en route
# to the ring is coupled to aromatization. So the substrate carries a beta-OH at the
# reduced position; aromatization expels it as water (which is why 6-MSA lacks
# orsellinic's 4-OH). HOOC-CH2-CO-CH2-CH(OH)-CH2-CO-CH3 -> 2-hydroxy-6-methylbenzoic acid.
_6MSA_ALDOL = (
    "[OH][C:1](=[O:9])[CH2:2][C:3](=[O:10])[CH2:4][CH:5]([OH])[CH2:6][C:7](=O)[CH3:8]"
    ">>[OH][C:1](=[O:9])[c:2]1[c:3]([OH:10])[cH:4][cH:5][cH:6][c:7]1[CH3:8]"
)

# Aromatic single-mode templates tried in order; they are substrate-disjoint (the
# orsellinic motif requires three intact keto groups, the 6-MSA motif requires the
# beta-hydroxyl), so at most one fires for a given linear chain.
_AROMATIC_TEMPLATES = (_ORSELLINIC_ALDOL, _6MSA_ALDOL)

# Dihydroisocoumarin composite closure (mellein family): TE-released linear pentaketide
# acid -> a fused benzene + delta-lactone in one operator. Lactonization (C1 carboxyl
# onto the cycle-1 KR hydroxyl) + aromatic aldol on the C2-C7 segment + aromatization.
# Two substrate-disjoint variants:
#   * MELLEIN     : the C5-derived ring position is reduced (CH-OH) -> aromatic CH, no OH.
#                   Requires TWO KRs (cycles 1 and 3) -> (R)-mellein, C10H10O3.
#   * HYDROXYMELLEIN: the C5-derived ring ketone survives -> a second phenol.
#                   Requires ONE KR (cycle 1) -> 6-hydroxymellein, C10H10O4.
# Stereochemistry at C3 is not set here (achiral Phase-0 executor; a Phase-2 head).
_MELLEIN = (
    "[OH][C:1](=[O:11])[CH2:2][C:3](=[O:12])[CH2:4][CH:5]([OH])[CH2:6][C:7](=O)"
    "[CH2:8][CH:9]([OH])[CH3:10]"
    ">>[CH3:10][CH:9]1[CH2:8][c:7]2[cH:6][cH:5][cH:4][c:3]([OH:12])[c:2]2[C:1](=[O:11])O1"
)
_HYDROXYMELLEIN = (
    "[OH][C:1](=[O:11])[CH2:2][C:3](=[O:12])[CH2:4][C:5](=[O:13])[CH2:6][C:7](=O)"
    "[CH2:8][CH:9]([OH])[CH3:10]"
    ">>[CH3:10][CH:9]1[CH2:8][c:7]2[cH:6][c:5]([OH:13])[cH:4][c:3]([OH:12])[c:2]2"
    "[C:1](=[O:11])O1"
)
_DIHYDROISOCOUMARIN_TEMPLATES = (_MELLEIN, _HYDROXYMELLEIN)

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


def _first_unique(templates, linear_acid: Chem.Mol) -> Chem.Mol | None:
    products: dict[str, Chem.Mol] = {}
    for smarts in templates:
        prod = _run_single(smarts, linear_acid)
        if prod is not None:
            products[Chem.MolToSmiles(prod)] = prod
    if len(products) == 1:
        return next(iter(products.values()))
    return None  # no template matched, or conflicting matches


def aldol_aromatic(linear_acid: Chem.Mol) -> Chem.Mol | None:
    return _first_unique(_AROMATIC_TEMPLATES, linear_acid)


def dihydroisocoumarin(linear_acid: Chem.Mol) -> Chem.Mol | None:
    return _first_unique(_DIHYDROISOCOUMARIN_TEMPLATES, linear_acid)


def release(linear_acid: Chem.Mol, mode: Release) -> Chem.Mol | None:
    if mode is Release.LACTONIZATION:
        return lactonize(linear_acid)
    if mode is Release.ALDOL_AROMATIC:
        return aldol_aromatic(linear_acid)
    if mode is Release.DIHYDROISOCOUMARIN:
        return dihydroisocoumarin(linear_acid)
    return None
