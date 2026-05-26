"""Engine v0 tests: observable-constrained inference + the per-cluster ladder."""

from __future__ import annotations

from lpi.chem import mol as M
from lpi.chem.program import ReductionState as R, Release
from lpi.engine import Observables, contains, frag_fingerprint, infer
from lpi.search.beam import _formula_cho
from lpi.search.generate import Alphabet

_ORS = "Cc1cc(O)cc(O)c1C(=O)O"          # orsellinic, C8H8O4
_OLI = "CCCCCc1cc(O)cc(O)c1C(=O)O"      # olivetolic, C12H16O4


def test_engine_mass_reconstructs_orsellinic():
    obs = Observables(Alphabet(reductions=(R.KETO,), releases=(Release.ALDOL_AROMATIC,),
                               starters=("acetyl",)), 3, 3,
                      target_cho=_formula_cho(M.mol_from_smiles(_ORS)))
    res = infer(obs)
    assert res.regime == "reconstructed"
    assert contains(res, _ORS) == 1


def test_engine_msms_resolves_isomers():
    cho = _formula_cho(M.mol_from_smiles(_OLI))
    alpha = Alphabet(reductions=(R.KETO, R.KR, R.DH, R.ER),
                     releases=(Release.ALDOL_AROMATIC, Release.HYDROLYSIS),
                     starters=("acetyl", "hexanoyl"), allow_cmet=True)
    mass = infer(Observables(alpha, 3, 5, target_cho=cho), ladder=False)
    msms = infer(Observables(alpha, 3, 5, target_cho=cho, msms=frag_fingerprint(_OLI)), ladder=False)
    assert len(msms.candidates) <= len(mass.candidates)   # MS/MS can only narrow
    assert contains(msms, _OLI) is not None


def test_ladder_is_monotone_nonincreasing():
    obs = Observables(Alphabet(reductions=(R.KETO, R.KR),
                               releases=(Release.ALDOL_AROMATIC, Release.HYDROLYSIS),
                               starters=("acetyl",)), 3, 3,
                      target_cho=_formula_cho(M.mol_from_smiles(_ORS)),
                      msms=frag_fingerprint(_ORS))
    L = infer(obs).ladder
    assert L["alphabet"] >= L["alphabet+mass"] >= L["alphabet+mass+msms"] >= 1
