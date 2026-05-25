"""Differential PoC tests — label derivation + baselines (no network; use committed parquet)."""

from __future__ import annotations

import pytest

from lpi.chem import mol as M
from lpi.model.clustercad_labels import reduction_state
from lpi.model.differential import domain_rule_reduction


def test_reduction_state_from_structure():
    cases = {
        "CC(=O)CC(=O)[S]": "KETO",   # beta-keto
        "CC(O)CC(=O)[S]": "KR",      # beta-hydroxyl
        "CC=[C]C(=O)[S]": "DH",      # enoyl
        "CCCC(=O)[S]": "ER",         # methylene
    }
    for smi, expected in cases.items():
        red, _stereo = reduction_state(M.mol_from_smiles(smi))
        assert red == expected, f"{smi}: got {red}, want {expected}"


def test_domain_rule_baseline():
    from lpi.model.clustercad_labels import REDUCTION_CLASSES

    assert REDUCTION_CLASSES[domain_rule_reduction({"KR", "DH", "ER"})] == "ER"
    assert REDUCTION_CLASSES[domain_rule_reduction({"KR", "DH"})] == "DH"
    assert REDUCTION_CLASSES[domain_rule_reduction({"KR"})] == "KR"
    assert REDUCTION_CLASSES[domain_rule_reduction(set())] == "KETO"


def test_stereo_read():
    _r, stereo = reduction_state(M.mol_from_smiles("C[C@H](O)CC(=O)[S]"))
    assert stereo in ("R", "S")
