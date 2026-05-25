"""Per-operator unit tests + mass-bookkeeping property tests.

The executor is the verifier; a bug here silently corrupts every training target, so
each operator is pinned both structurally (canonical SMILES) and by exact-mass delta.
"""

from __future__ import annotations

import pytest
from rdkit.Chem import Descriptors

from lpi.executor import operators as op

H2 = 2.015650
H2O = 18.010565
C2H2O = 42.010565  # net atoms added by a decarboxylative Claisen (malonyl) extension
CH2 = 14.015650


def mass(mol) -> float:
    return Descriptors.ExactMolWt(mol)


def smi(mol) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(mol)


def test_starter_acetyl():
    assert smi(op.load_starter("acetyl")) == "*SC(C)=O"


def test_extension_adds_C2H2O():
    m = op.load_starter("acetyl")
    m1 = op.extend(m)
    assert smi(m1) == "*SC(=O)CC(C)=O"  # acetoacetyl-S*
    assert mass(m1) - mass(m) == pytest.approx(C2H2O, abs=1e-4)


def test_kr_adds_H2():
    m1 = op.extend(op.load_starter("acetyl"))
    m2 = op.ketoreduce(m1)
    assert mass(m2) - mass(m1) == pytest.approx(H2, abs=1e-4)


def test_dh_removes_H2O():
    m2 = op.ketoreduce(op.extend(op.load_starter("acetyl")))
    m3 = op.dehydrate(m2)
    assert mass(m3) - mass(m2) == pytest.approx(-H2O, abs=1e-4)


def test_er_adds_H2():
    m3 = op.dehydrate(op.ketoreduce(op.extend(op.load_starter("acetyl"))))
    m4 = op.enoylreduce(m3)
    assert mass(m4) - mass(m3) == pytest.approx(H2, abs=1e-4)
    assert smi(op.hydrolyse(m4)) == "CCCC(=O)O"  # butyric acid


def test_cmet_adds_CH2():
    m1 = op.extend(op.load_starter("acetyl"))
    mm = op.c_methylate(m1)
    assert mass(mm) - mass(m1) == pytest.approx(CH2, abs=1e-4)
    assert smi(mm) == "*SC(=O)C(C)C(C)=O"


def test_cmet_will_not_methylate_twice():
    # alpha is CH(CH3) after one C-MeT -> the CH2 pattern no longer matches.
    mm = op.c_methylate(op.extend(op.load_starter("acetyl")))
    with pytest.raises(op.OperatorError):
        op.c_methylate(mm)


def test_hydrolysis_gives_carboxylic_acid():
    m1 = op.extend(op.load_starter("acetyl"))
    acid = op.hydrolyse(m1)
    assert smi(acid) == "CC(=O)CC(=O)O"  # acetoacetic acid
    assert not any(a.GetAtomicNum() == 0 for a in acid.GetAtoms())  # sentinel gone


def test_full_cascade_is_deterministic():
    def build():
        return smi(op.enoylreduce(op.dehydrate(op.ketoreduce(
            op.extend(op.load_starter("acetyl"))))))

    assert build() == build()


def test_operator_does_not_mutate_input():
    m = op.load_starter("acetyl")
    before = smi(m)
    _ = op.extend(m)
    assert smi(m) == before  # operators return new mols, never mutate
