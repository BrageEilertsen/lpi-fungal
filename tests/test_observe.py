"""Track 2 observability layer: the seven acceptance criteria, plus adduct/ppm/formula checks.

Acceptance criteria (gate definition):
  1. exact infer() unchanged + green               -> the rest of the suite (test_engine etc.)
  2. tolerant infer_ms works for PKS and NRPS
  3. adduct-aware mass recovers the same neutral candidate under >= [M+H]+ and [M-H]-
  4. wrong adduct / wrong mass -> UNDER_OBSERVED or OUT_OF_GRAMMAR with an explicit reason
  5. noisy MS/MS resolves at least one mass-ambiguous candidate set
  6. mass withheld -> UNDER_OBSERVED with next_observable = accurate mass
  7. all verdicts include grammar-relative wording
"""
from __future__ import annotations

from lpi.chem.program import ReductionState as R, Release
from lpi.engine import State, contains
from lpi.grammars import NRPS, PKS
from lpi.grammars.nrps import NRPSAlphabet
from lpi.observe import (ADDUCTS, MSObservables, exact_mass, infer_ms, ion_mz,
                         minimum_sufficient_observables, msms_score, neutral_from_mz)
from lpi.search.generate import Alphabet
from lpi.engine import frag_fingerprint

_ORS = "Cc1cc(O)cc(O)c1C(=O)O"          # orsellinic, C8H8O4
_OLI = "CCCCCc1cc(O)cc(O)c1C(=O)O"      # olivetolic, C12H16O4
_DKP = "O=C1CNC(=O)CN1"                  # cyclo(Gly-Gly)

_NR = Alphabet((R.KETO,), (Release.ALDOL_AROMATIC, Release.LACTONIZATION, Release.HYDROLYSIS),
               ("acetyl",), True)
_PR = Alphabet((R.KETO, R.KR), (Release.ALDOL_AROMATIC, Release.LACTONIZATION,
                                Release.HYDROLYSIS, Release.DIHYDROISOCOUMARIN), ("acetyl",), True)


# ---- adduct + ppm arithmetic ----------------------------------------------------
def test_adduct_neutral_mass_roundtrips():
    M = exact_mass(_ORS)
    for name in ("[M+H]+", "[M+Na]+", "[M-H]-", "[M+Cl]-", "[M+NH4]+"):
        assert abs(neutral_from_mz(ion_mz(M, ADDUCTS[name]), ADDUCTS[name]) - M) < 1e-6


def test_adducts_carry_charge_and_mode():
    assert ADDUCTS["[M+H]+"].charge == 1 and ADDUCTS["[M+H]+"].mode == "positive"
    assert ADDUCTS["[M-H]-"].mode == "negative"


# ---- (2) tolerant infer_ms works for PKS and NRPS -------------------------------
def test_tolerant_infer_pks():
    mz = ion_mz(exact_mass(_ORS), ADDUCTS["[M-H]-"])
    res = infer_ms(_NR, MSObservables(observed_mz=mz, adducts=("[M-H]-",), ppm=5), PKS, 3, 3)
    assert res.state is State.VERIFIED
    assert contains(res, _ORS) == 1


def test_tolerant_infer_nrps():
    mz = ion_mz(exact_mass(_DKP), ADDUCTS["[M+H]+"])
    res = infer_ms(NRPSAlphabet(), MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5),
                   NRPS, 2, 2)
    assert res.state is State.VERIFIED
    assert contains(res, _DKP) == 1


# ---- (3) adduct-aware mass recovers the same neutral candidate under +/- ---------
def test_same_neutral_candidate_under_pos_and_neg_adduct():
    pos = infer_ms(_NR, MSObservables(observed_mz=ion_mz(exact_mass(_ORS), ADDUCTS["[M+H]+"]),
                                      adducts=("[M+H]+",), ppm=5), PKS, 3, 3)
    neg = infer_ms(_NR, MSObservables(observed_mz=ion_mz(exact_mass(_ORS), ADDUCTS["[M-H]-"]),
                                      adducts=("[M-H]-",), ppm=5), PKS, 3, 3)
    assert contains(pos, _ORS) == 1 and contains(neg, _ORS) == 1
    assert pos.formulas == neg.formulas == ["C8H8O4"]


# ---- (4) wrong mass / wrong adduct -> explicit OUT_OF_GRAMMAR --------------------
def test_wrong_mass_is_out_of_grammar_with_reason():
    res = infer_ms(_NR, MSObservables(neutral_mass=999.0, ppm=5), PKS, 3, 3, ladder=False)
    assert res.state is State.OUT_OF_GRAMMAR
    assert "ppm" in res.reason and "grammar" in res.reason


def test_wrong_adduct_is_out_of_grammar():
    # observed as a sodium adduct, but searched assuming protonation -> wrong neutral -> no match
    mz_na = ion_mz(exact_mass(_ORS), ADDUCTS["[M+Na]+"])
    res = infer_ms(_NR, MSObservables(observed_mz=mz_na, adducts=("[M+H]+",), ppm=5), PKS, 3, 3,
                   ladder=False)
    assert res.state is State.OUT_OF_GRAMMAR
    assert "adduct" in res.reason


# ---- (5) noisy MS/MS resolves a mass-ambiguous candidate set ---------------------
def test_noisy_msms_resolves_mass_ambiguous_set():
    mz = ion_mz(exact_mass(_OLI), ADDUCTS["[M+H]+"])
    alpha = PKS.alphabet_from_domains({"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"})
    mass_only = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5),
                         PKS, 3, 6, ladder=False)
    assert mass_only.ladder["+mass"] > 1                       # genuinely ambiguous on mass alone
    # observed MS/MS = true fragments + deterministic noise peaks
    noisy = tuple(frag_fingerprint(_OLI)) + (33.3, 71.7, 150.5)
    res = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5,
                                        msms_peaks=noisy, msms_tau=0.7), PKS, 3, 6, ladder=False)
    assert res.ladder["+msms"] < mass_only.ladder["+mass"]     # MS/MS narrowed it
    assert contains(res, _OLI) is not None                     # truth retained despite noise
    assert res.state is State.VERIFIED


# ---- (6) mass withheld -> UNDER_OBSERVED, next = accurate mass -------------------
def test_mass_withheld_is_under_observed():
    res = infer_ms(_NR, MSObservables(), PKS, 3, 3)            # NR genome = 9 > near-unique
    assert res.state is State.UNDER_OBSERVED
    assert "mass" in res.next_observable


# ---- (7) verdicts are grammar-relative ------------------------------------------
def test_verdicts_are_grammar_relative():
    pks = infer_ms(_NR, MSObservables(observed_mz=ion_mz(exact_mass(_ORS), ADDUCTS["[M-H]-"]),
                                      adducts=("[M-H]-",), ppm=5), PKS, 3, 3)
    nrps = infer_ms(NRPSAlphabet(), MSObservables(), NRPS, 2, 2, ladder=True)
    assert "pks" in pks.reason
    assert "nrps" in nrps.reason


# ---- formula ambiguity is within-grammar; ppm stability -------------------------
def test_formula_ambiguity_is_within_grammar():
    mz = ion_mz(exact_mass(_OLI), ADDUCTS["[M+H]+"])
    res = infer_ms(PKS.alphabet_from_domains({"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"}),
                   MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5), PKS, 3, 6, ladder=False)
    # the reported formulas are those PRODUCED by legal candidate programs, not all isobars
    assert res.formulas == ["C12H16O4"]


def test_mass_collapse_stable_across_ppm():
    mz = ion_mz(exact_mass(_OLI), ADDUCTS["[M+H]+"])
    alpha = PKS.alphabet_from_domains({"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"})
    counts = {ppm: infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=ppm),
                            PKS, 3, 6, ladder=False).ladder["+mass"] for ppm in (2, 5, 10, 20)}
    assert len(set(counts.values())) == 1                      # identical across realistic tolerances


# ---- MS/MS scorer sanity --------------------------------------------------------
def test_msms_score_identity_and_noise_robustness():
    fp = frag_fingerprint(_OLI)
    assert msms_score(fp, fp, 0.02) == 1.0                     # identical spectra
    noisy = tuple(fp) + (33.3, 71.7)
    assert msms_score(fp, noisy, 0.02) > 0.7                   # extra noise peaks degrade gently
    assert msms_score(fp, (1.1, 2.2, 3.3), 0.02) == 0.0        # disjoint -> zero


# ---- the experiment planner -----------------------------------------------------
def test_minimum_sufficient_observables_plan():
    mz = ion_mz(exact_mass(_ORS), ADDUCTS["[M-H]-"])
    plan = minimum_sufficient_observables(
        _NR, MSObservables(observed_mz=mz, adducts=("[M-H]-",), ppm=5,
                           msms_peaks=tuple(frag_fingerprint(_ORS))), PKS, 3, 3)
    assert plan.verified_at == "+mass"                         # mass alone suffices here
    assert plan.recommended is None
