"""Macrolactonization operator: sound ring closure, ambiguity discipline, and a
real-target validation against zearalenone's RETRIEVED deposited connectivity.

The macrolactone DOF is RING SIZE (which chain hydroxyl attacks C1) -- formula-invisible,
so the operator must not guess it: it fires only when the closure is unambiguous (one
carboxyl, one eligible aliphatic hydroxyl) or when a curated locant is supplied. Phenols
and the acid's own -OH are not eligible nucleophiles. The executor is achiral, so the
zearalenone check is a flat (stereo-stripped) 2D-connectivity match, not a stereo claim.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import RWMol, rdMolDescriptors

from lpi.chem.program import Release
from lpi.executor.cyclize import (
    _aliphatic_hydroxyls,
    _carboxyl_carbons,
    macrolactonize,
    release,
)

# Zearalenone, RETRIEVED (not recalled): MIBiG BGC0001057 deposited product, stereo-stripped.
# Verified 2026-06-01 against data/processed/fungal_pks_pairs.parquet (product_smiles_canonical)
# and data/raw/mibig_json_4.0/BGC0001057.json (chem_struct). C18H22O5, a 14-membered beta-
# resorcylic-acid lactone.
ZEARALENONE_FLAT = "CC1CCCC(=O)CCCC=Cc2cc(O)cc(O)c2C(=O)O1"


def _formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


def _lost_one_water(substrate: Chem.Mol, product: Chem.Mol) -> bool:
    """Product formula == substrate formula minus exactly one H2O."""
    from collections import Counter
    import re

    def counts(m):
        c = Counter()
        for tok, n in re.findall(r"([A-Z][a-z]?)(\d*)", _formula(m)):
            if tok:
                c[tok] += int(n) if n else 1
        return c

    diff = counts(substrate)
    for el, n in counts(product).items():
        diff[el] -= n
    return diff == Counter({"H": 2, "O": 1})


def _seco_acid_from_macrolactone(mol: Chem.Mol) -> Chem.Mol:
    """Ring-open a deposited macrolactone to its seco-acid by hydrolysing the lactone ester
    in the largest ring -- so the test substrate is provably the ring-opened deposited
    structure, not a hand-drawn one."""
    ri = mol.GetRingInfo()
    macro = max(ri.AtomRings(), key=len)
    ester_o = carbonyl = None
    for idx in macro:
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() == "O" and not a.GetIsAromatic() and a.GetDegree() == 2:
            cyl = [c for c in a.GetNeighbors()
                   if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(c).GetSymbol() == "O"
                          for b in c.GetBonds())]
            if cyl:
                ester_o, carbonyl = idx, cyl[0].GetIdx()
                break
    assert ester_o is not None, "no lactone ester found in macro-ring"
    rw = RWMol(mol)
    rw.RemoveBond(carbonyl, ester_o)
    new_o = rw.AddAtom(Chem.Atom(8))  # acid -OH
    rw.AddBond(carbonyl, new_o, Chem.BondType.SINGLE)
    rw.GetAtomWithIdx(new_o).SetNoImplicit(False)
    rw.GetAtomWithIdx(ester_o).SetNoImplicit(False)
    rw.GetAtomWithIdx(ester_o).SetNumExplicitHs(1)  # freed alcohol
    seco = rw.GetMol()
    Chem.SanitizeMol(seco)
    return seco


# ---- mechanism --------------------------------------------------------------------------

def test_macrolactonize_closes_unambiguous_omega_hydroxy_acid():
    # 12-hydroxydodecanoic acid: one carboxyl, one aliphatic -OH -> one macrolactone.
    seco = Chem.MolFromSmiles("OCCCCCCCCCCCC(=O)O")
    prod = macrolactonize(seco)
    assert prod is not None
    assert _lost_one_water(seco, prod)
    ring_sizes = sorted(len(r) for r in prod.GetRingInfo().AtomRings())
    assert ring_sizes == [13]  # 12 C + 1 ester O


def test_release_dispatches_macrolactonization():
    seco = Chem.MolFromSmiles("OCCCCCCCCCCCC(=O)O")
    via_release = release(seco, Release.MACROLACTONIZATION)
    assert via_release is not None
    assert Chem.MolToSmiles(via_release) == Chem.MolToSmiles(macrolactonize(seco))


# ---- soundness: do not guess a ring size --------------------------------------------------

def test_macrolactonize_returns_none_when_ring_size_ambiguous():
    # Two eligible aliphatic hydroxyls, no curated locant -> ambiguous ring size -> None.
    diol_acid = Chem.MolFromSmiles("OCCCCCC(O)CCCCC(=O)O")
    assert len(_aliphatic_hydroxyls(diol_acid)) == 2
    assert macrolactonize(diol_acid) is None


def test_macrolactonize_uses_curated_locant_to_resolve_ambiguity():
    diol_acid = Chem.MolFromSmiles("OCCCCCC(O)CCCCC(=O)O")
    candidates = _aliphatic_hydroxyls(diol_acid)
    assert len(candidates) == 2
    # supplying either curated locant fires a unique (different) ring
    p0 = macrolactonize(diol_acid, ring_oh_idx=candidates[0])
    p1 = macrolactonize(diol_acid, ring_oh_idx=candidates[1])
    assert p0 is not None and p1 is not None
    assert Chem.MolToSmiles(p0) != Chem.MolToSmiles(p1)
    # a locant that is not an eligible hydroxyl does not fire
    assert macrolactonize(diol_acid, ring_oh_idx=999) is None


def test_macrolactonize_excludes_phenol_and_acid_oh():
    # salicylic acid: only a phenolic -OH (not eligible) + the carboxyl -OH -> no closure.
    salicylic = Chem.MolFromSmiles("Oc1ccccc1C(=O)O")
    assert _aliphatic_hydroxyls(salicylic) == []
    assert len(_carboxyl_carbons(salicylic)) == 1
    assert macrolactonize(salicylic) is None


# ---- real target: zearalenone -----------------------------------------------------------

def test_macrolactonize_reproduces_zearalenone_connectivity():
    zea = Chem.MolFromSmiles(ZEARALENONE_FLAT)
    target = Chem.MolToSmiles(zea)
    seco = _seco_acid_from_macrolactone(zea)
    # the seco-acid presents exactly one carboxyl and one eligible aliphatic -OH
    # (the starter-derived methyl-carbinol; the two resorcinol phenols are excluded)
    assert len(_carboxyl_carbons(seco)) == 1
    assert len(_aliphatic_hydroxyls(seco)) == 1
    prod = macrolactonize(seco)
    assert prod is not None
    assert _lost_one_water(seco, prod)
    assert Chem.MolToSmiles(prod) == target
