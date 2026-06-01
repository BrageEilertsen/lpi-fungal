"""Positive control for the (trajectory,trajectory) sound-unreachability verdict.

The verdict rests on "the poly-beta-keto stretch admits >1 aromatizable first-ring fold," which must
be RENDERED, not inferred (the same standard the SMARTS was held to). Three pre-committed checks:

  (a) given the resorcylic register (C2-C7 aldol), the chemistry renders zearalenone -> implementability
      is NOT the bottleneck;
  (b) given the competing register (C1-C6 Claisen), it renders a DIFFERENT valid aromatic -> the second
      mode is productive, not a geometric ghost;
  (c) given no register, it returns None -> the soundness behaviour a sound executor must show on the
      RAL cohort, whose PT register is recoverable from neither the grammar nor MIBiG 4.0 annotation.

The two folds are the classic PT-domain regiocontrol alternatives (C2-C7 aldol -> resorcylic acid;
C1-C6 Claisen -> acylphloroglucinol). Connectivity only; the executor is achiral.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from lpi.executor.cyclize import aromatize_register, macrolactonize

# zearalenone's verified pre-aromatization nonaketide precursor (program acetyl + [KR,ER,KETO,ER,DH,
# KETO,KETO,KETO], hydrolysis); atom indices below are stable for this exact SMILES.
PRECURSOR = "CC(O)CCCC(=O)CCCC=CC(=O)CC(=O)CC(=O)CC(=O)O"
ZEARALENONE = "CC1CCCC(=O)CCCC=Cc2cc(O)cc(O)c2C(=O)O1"  # retrieved: MIBiG BGC0001057, flat
SECO_ACID = "CC(O)CCCC(=O)CCCC=Cc1cc(O)cc(O)c1C(=O)O"     # the aromatized open (resorcylic) form
ACYLPHLOROGLUCINOL = "CC(O)CCCC(=O)CCCC=CC(=O)c1c(O)cc(O)cc1O"  # pinned competing product

RESORCYLIC_REGISTER = (21, 13)  # C2->C7 aldol: alpha-C21 attacks ketone-C13
CLAISEN_REGISTER = (15, 22)     # C1->C6 Claisen: alpha-C15 attacks carboxyl-C22


def _canon(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_check_a_resorcylic_register_renders_zearalenone():
    """Implementability: given the resorcylic register, the chemistry reaches zearalenone."""
    pre = Chem.MolFromSmiles(PRECURSOR)
    seco = aromatize_register(pre, RESORCYLIC_REGISTER)
    assert seco is not None
    assert Chem.MolToSmiles(seco) == _canon(SECO_ACID)         # aromatized open form
    zea = macrolactonize(seco)                                  # then close the macrolactone
    assert zea is not None
    assert Chem.MolToSmiles(zea) == _canon(ZEARALENONE)        # == retrieved deposited connectivity


def test_check_b_competing_register_renders_distinct_valid_aromatic():
    """The competing fold is productive (a real second mode), not a geometric ghost."""
    pre = Chem.MolFromSmiles(PRECURSOR)
    mode_a = aromatize_register(pre, RESORCYLIC_REGISTER)
    mode_b = aromatize_register(pre, CLAISEN_REGISTER)
    assert mode_b is not None
    assert Chem.MolToSmiles(mode_b) != Chem.MolToSmiles(mode_a)               # distinct products
    assert any(at.GetIsAromatic() for at in mode_b.GetAtoms())                # a valid aromatic ring
    assert not mode_b.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))  # no exocyclic COOH
    assert Chem.MolToSmiles(mode_b) == _canon(ACYLPHLOROGLUCINOL)             # matches the pin
    # both folds are isomeric aromatizations of one precursor (same formula, different connectivity)
    assert rdMolDescriptors.CalcMolFormula(mode_a) == rdMolDescriptors.CalcMolFormula(mode_b)


def test_check_c_no_register_returns_none():
    """Soundness: with no register the fold is underdetermined -> the executor must not guess."""
    pre = Chem.MolFromSmiles(PRECURSOR)
    assert aromatize_register(pre, None) is None
