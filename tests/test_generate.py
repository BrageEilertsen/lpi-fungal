"""Candidate-generator tests: alphabet-constrained enumeration + the two-walls size signal."""

from __future__ import annotations

from lpi.chem.program import ReductionState as R, Release
from lpi.search.generate import Alphabet, generate, rank_of


def test_nr_alphabet_generates_orsellinic_in_a_tiny_set():
    alpha = Alphabet(reductions=(R.KETO,), releases=(Release.ALDOL_AROMATIC,),
                     starters=("acetyl",))
    res = generate(alpha, 3, 3)
    assert rank_of(res, "Cc1cc(O)cc(O)c1C(=O)O") is not None
    assert len(res.candidates) <= 3  # keto-only => ~one product per length


def test_hr_alphabet_generates_octanoic():
    alpha = Alphabet(reductions=(R.KETO, R.KR, R.DH, R.ER), releases=(Release.HYDROLYSIS,))
    res = generate(alpha, 3, 3)
    assert rank_of(res, "CCCCCCCC(=O)O") is not None  # acetyl + 3x ER


def test_candidate_set_grows_with_alphabet_richness():
    nr = generate(Alphabet(reductions=(R.KETO,), releases=(Release.HYDROLYSIS,)), 4, 4)
    hr = generate(Alphabet(reductions=(R.KETO, R.KR, R.DH, R.ER),
                           releases=(Release.HYDROLYSIS,)), 4, 4)
    assert len(hr.candidates) > len(nr.candidates)  # the two-walls signal, made concrete


def test_formula_conditioning_collapses_to_target():
    """The mining lever: an MS-observable formula skips off-mass programs before execution."""
    from lpi.chem import mol as M
    from lpi.search.beam import _formula_cho
    alpha = Alphabet(reductions=(R.KETO, R.KR, R.DH, R.ER),
                     releases=(Release.ALDOL_AROMATIC, Release.HYDROLYSIS), starters=("acetyl",))
    cho = _formula_cho(M.mol_from_smiles("Cc1cc(O)cc(O)c1C(=O)O"))  # orsellinic, C8H8O4
    conditioned = generate(alpha, 3, 3, target_cho=cho)
    unconditioned = generate(alpha, 3, 3)
    assert rank_of(conditioned, "Cc1cc(O)cc(O)c1C(=O)O") is not None
    assert all(_formula_cho(M.mol_from_smiles(c.smiles)) == cho for c in conditioned.candidates)
    assert len(conditioned.candidates) <= len(unconditioned.candidates)
