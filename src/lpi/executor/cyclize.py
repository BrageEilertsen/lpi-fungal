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
from rdkit.Chem import RWMol

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


# --- Macrolactonization (cis-TE) -------------------------------------------------------
# Unlike the small-ring templates above, a macrolactone is formula-identical to any other
# one-water closure; the distinguishing DOF is RING SIZE -- which chain hydroxyl attacks the
# C1 carboxyl. A macrolactone is therefore NOT a fixed SMARTS: the operator esterifies the
# carboxyl with an aliphatic hydroxyl chosen by the recoverable cis-TE feature (`ring_oh_idx`,
# the curated rung-2 locant). Phenols and the acid's own -OH are not eligible nucleophiles.
# When no locant is supplied it fires ONLY if the closure is unambiguous (one carboxyl, one
# eligible hydroxyl); a non-unique ring size returns None -- the executor does not guess.


def _carboxyl_carbons(mol: Chem.Mol) -> list[tuple[int, int]]:
    """(carbonyl-C idx, acid-OH O idx) for each free carboxylic acid -C(=O)OH."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        dbl_o = oh_o = None
        for b in a.GetBonds():
            o = b.GetOtherAtom(a)
            if o.GetSymbol() != "O":
                continue
            if b.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o = o.GetIdx()
            elif o.GetDegree() == 1 and o.GetTotalNumHs() == 1:
                oh_o = o.GetIdx()
        if dbl_o is not None and oh_o is not None:
            out.append((a.GetIdx(), oh_o))
    return out


def _aliphatic_hydroxyls(mol: Chem.Mol) -> list[int]:
    """O idx of sp3 carbinol -OH groups eligible to close the macrolactone (excludes
    phenols and the carboxylic-acid -OH, whose carbon bears a =O)."""
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "O" or a.GetIsAromatic():
            continue
        if a.GetDegree() != 1 or a.GetTotalNumHs() != 1:
            continue
        c = a.GetNeighbors()[0]
        if c.GetSymbol() != "C" or c.GetIsAromatic():
            continue
        if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(c).GetSymbol() == "O"
               for b in c.GetBonds()):
            continue  # the carboxyl -OH
        out.append(a.GetIdx())
    return out


def macrolactonize(linear_acid: Chem.Mol, ring_oh_idx: int | None = None) -> Chem.Mol | None:
    """Esterify the C1 carboxyl with an aliphatic hydroxyl, losing one water. Ring size is
    fixed by `ring_oh_idx` (the curated cis-TE locant); when omitted, fires only if the
    closure is unambiguous, else returns None (no ring-size guess)."""
    acids = _carboxyl_carbons(linear_acid)
    if len(acids) != 1:
        return None
    carbonyl_c, acid_o = acids[0]
    candidates = _aliphatic_hydroxyls(linear_acid)
    if ring_oh_idx is not None:
        if ring_oh_idx not in candidates:
            return None
        oh_o = ring_oh_idx
    elif len(candidates) == 1:
        oh_o = candidates[0]
    else:
        return None  # ambiguous (or no) ring size -> do not guess
    rw = RWMol(linear_acid)
    rw.RemoveBond(carbonyl_c, acid_o)
    rw.AddBond(carbonyl_c, oh_o, Chem.BondType.SINGLE)
    o = rw.GetAtomWithIdx(oh_o)
    o.SetNumExplicitHs(0)
    o.SetNoImplicit(False)
    rw.RemoveAtom(acid_o)  # the expelled water O (last: RemoveAtom reindexes)
    prod = rw.GetMol()
    try:
        Chem.SanitizeMol(prod)
    except Exception:  # noqa: BLE001
        return None
    return prod


# --- Resorcylic aromatization: a register-parameterized POSITIVE CONTROL (NOT a release) ----
# Deliberately NOT wired into the search release set. It exists to RENDER (not infer) the verdict
# that the (trajectory,trajectory) macrolactone occupant is sound-unreachable from current
# observables: a poly-beta-keto stretch admits >1 aromatizable first-ring fold -- the C2-C7 aldol
# (-> resorcylic acid, e.g. zearalenone) vs the C1-C6 Claisen (-> acylphloroglucinol), the classic
# PT-domain regiocontrol -- so the register is a DOF the chain does not fix. Given the register it
# renders the fold; given none it returns None. Because that register (the PT selection) is
# recoverable from neither the executor's features nor the RAL cohort's MIBiG annotation, a sound
# executor must return None on the cohort -> the cell is empty by soundness, not by missing code.


def _ring_path(mol: Chem.Mol, a: int, c: int) -> list[int]:
    """Atom indices on the shortest a..c path inclusive (a 6-membered closure gives length 6)."""
    import collections
    prev: dict[int, int | None] = {a: None}
    q = collections.deque([a])
    while q:
        x = q.popleft()
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            k = nb.GetIdx()
            if k not in prev:
                prev[k] = x
                q.append(k)
    path, cur = [], c
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path


def aromatize_register(linear_acid: Chem.Mol, register: tuple[int, int] | None) -> Chem.Mol | None:
    """Form a first-ring aldol/Claisen at the (alpha-C, carbonyl-C) `register` and aromatize the
    6-ring: the attacked carbonyl loses its O as water (a carboxyl drops its -OH and keeps =O -> a
    phenol), the other two ring ketones tautomerize to phenols. `register` stands for the PT-domain
    fold selection. Returns None when no register is supplied (the soundness behaviour) or the pair
    is not a 6-ring closure -- the executor does not guess the fold."""
    if register is None:
        return None
    alpha, carbonyl = register
    path = _ring_path(linear_acid, alpha, carbonyl)
    if len(path) != 6:
        return None
    ring = set(path)
    rw = RWMol(linear_acid)
    rw.AddBond(alpha, carbonyl, Chem.BondType.SINGLE)
    to_remove: list[int] = []
    for idx in ring:
        a = rw.GetAtomWithIdx(idx)
        dbl = [b.GetOtherAtom(a).GetIdx() for b in a.GetBonds()
               if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "O"]
        if not dbl:
            continue
        o = dbl[0]
        if idx == carbonyl:
            sgl = [b.GetOtherAtom(a).GetIdx() for b in a.GetBonds()
                   if b.GetBondTypeAsDouble() == 1 and b.GetOtherAtom(a).GetSymbol() == "O"]
            if sgl:  # carboxyl Claisen: drop -OH (water), keep =O -> phenol
                rw.RemoveBond(idx, sgl[0])
                to_remove.append(sgl[0])
                rw.RemoveBond(idx, o)
                rw.AddBond(idx, o, Chem.BondType.SINGLE)
                rw.GetAtomWithIdx(o).SetNoImplicit(False)
                rw.GetAtomWithIdx(o).SetNumExplicitHs(1)
            else:    # aldol: attacked ketone O leaves as water
                rw.RemoveBond(idx, o)
                to_remove.append(o)
        else:        # other ring ketone -> phenol
            rw.RemoveBond(idx, o)
            rw.AddBond(idx, o, Chem.BondType.SINGLE)
            rw.GetAtomWithIdx(o).SetNoImplicit(False)
            rw.GetAtomWithIdx(o).SetNumExplicitHs(1)
    for idx in ring:
        rw.GetAtomWithIdx(idx).SetIsAromatic(True)
        rw.GetAtomWithIdx(idx).SetNoImplicit(False)
    for i in range(6):
        b = rw.GetBondBetweenAtoms(path[i], path[(i + 1) % 6])
        if b:
            b.SetBondType(Chem.BondType.AROMATIC)
            b.SetIsAromatic(True)
    for o in sorted(set(to_remove), reverse=True):
        rw.RemoveAtom(o)
    prod = rw.GetMol()
    try:
        Chem.SanitizeMol(prod)
    except Exception:  # noqa: BLE001
        return None
    return prod


def release(linear_acid: Chem.Mol, mode: Release, *, ring_oh_idx: int | None = None) -> Chem.Mol | None:
    if mode is Release.LACTONIZATION:
        return lactonize(linear_acid)
    if mode is Release.ALDOL_AROMATIC:
        return aldol_aromatic(linear_acid)
    if mode is Release.DIHYDROISOCOUMARIN:
        return dihydroisocoumarin(linear_acid)
    if mode is Release.MACROLACTONIZATION:
        return macrolactonize(linear_acid, ring_oh_idx)
    return None
