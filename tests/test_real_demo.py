"""Real Mining Demo v1 (6-MSA) -- lock the blind real-cluster reconstruction as a regression.

Inputs are the real, documented values from results/real_demo_6msa_sources.md (MIBiG BGC0001275,
canonical 6-MSAS domain architecture, real monoisotopic mass). The product structure is withheld
from inference and only checked afterwards.
"""
from __future__ import annotations

from lpi.chem import mol as M
from lpi.engine import State, contains
from lpi.grammars import PKS
from lpi.observe import ADDUCTS, MSObservables, ion_mz, infer_ms

_TRUE_6MSA = "CC1=C(C(=CC=C1)O)C(=O)O"     # MIBiG compound record; withheld from inference
_REAL_MASS = 152.047344116                  # MIBiG / PubChem CID 11279 (C8H8O3)


def test_real_6msa_blind_reconstruction():
    alpha = PKS.alphabet_from_domains({"KS", "AT", "DH", "KR", "ACP"})   # literature 6-MSAS domains
    mz = ion_mz(_REAL_MASS, ADDUCTS["[M-H]-"])
    res = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M-H]-",), ppm=5), PKS, 2, 6)
    assert res.state is State.VERIFIED
    assert res.formulas == ["C8H8O3"]
    assert contains(res, M.canonical_smiles(M.mol_from_smiles(_TRUE_6MSA))) == 1
