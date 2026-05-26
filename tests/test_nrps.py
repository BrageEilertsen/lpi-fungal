"""NRPS v0 grammar + executor tests.

The executor is critical code (it defines reconstruction targets), so this mirrors the PKS
executor's rigor: known-product round-trips with *independent* ground-truth SMILES, mass
bookkeeping (analytic formula == built-molecule formula), valence validity over every residue,
determinism, and engine integration showing the shared three-state verdict / observable ladder
work through the NRPS grammar unchanged.
"""

from __future__ import annotations

import itertools

import pytest

from lpi.chem import mol as M
from lpi.engine import Observables, State, contains, infer, infer_cluster
from lpi.executor import nrps as N
from lpi.executor.nrps import Peptide, Release, RESIDUES, formula_chno, peptide_formula, run
from lpi.grammars import NRPS
from lpi.grammars.nrps import NRPSAlphabet

# Independent ground truth (written differently from the executor's own assembly order).
_GLYGLY = "OC(=O)CNC(=O)CN"        # glycylglycine, H2N-CH2-CO-NH-CH2-COOH
_DKP_GG = "O=C1CNC(=O)CN1"          # cyclo(Gly-Gly), glycine 2,5-diketopiperazine


def _canon(smiles: str) -> str:
    return M.canonical_smiles(M.mol_from_smiles(smiles))


# ---- round-trip: known products -------------------------------------------------
def test_roundtrip_glycylglycine_linear():
    got = M.canonical_smiles(run(Peptide(("Gly", "Gly"), Release.HYDROLYSIS)))
    assert got == _canon(_GLYGLY)


def test_roundtrip_cyclo_glygly_diketopiperazine():
    got = M.canonical_smiles(run(Peptide(("Gly", "Gly"), Release.MACROLACTAM)))
    assert got == _canon(_DKP_GG)


# ---- mass bookkeeping: analytic formula == built-molecule formula ----------------
@pytest.mark.parametrize("residues", [
    ("Gly", "Gly"), ("Ala", "Gly"), ("Phe", "Ala"), ("Ser", "Thr", "Val"),
    ("Tyr", "Leu"), ("Phe", "Phe", "Gly"),
])
@pytest.mark.parametrize("release", [Release.HYDROLYSIS, Release.MACROLACTAM])
def test_analytic_formula_matches_built_molecule(residues, release):
    analytic = peptide_formula(residues, release)
    built = formula_chno(run(Peptide(residues, release)))
    assert analytic == built, f"{residues}/{release}: analytic {analytic} != built {built}"


def test_each_amide_bond_loses_one_water():
    # linear loses n-1 waters, macrolactam loses n; difference is exactly one water (H2,O1).
    res = ("Ala", "Ser", "Gly")
    lin = peptide_formula(res, Release.HYDROLYSIS)
    cyc = peptide_formula(res, Release.MACROLACTAM)
    assert (lin[0] - cyc[0], lin[1] - cyc[1], lin[2] - cyc[2], lin[3] - cyc[3]) == (0, 2, 0, 1)


# ---- valence validity over every residue ----------------------------------------
def test_every_residue_builds_valid_linear_and_cyclic():
    for r in RESIDUES:
        assert run(Peptide((r, "Gly"), Release.HYDROLYSIS)).GetNumAtoms() > 0
        assert run(Peptide((r, "Gly"), Release.MACROLACTAM)).GetNumAtoms() > 0


def test_aromatic_sidechain_macrocycle_no_ring_digit_collision():
    # Phe/Tyr carry their own ring-closure '1'; the macrocycle must use a distinct label.
    mol = run(Peptide(("Phe", "Tyr"), Release.MACROLACTAM))  # would raise if SMILES malformed
    assert mol.GetNumAtoms() > 0


# ---- determinism -----------------------------------------------------------------
def test_executor_is_deterministic():
    pep = Peptide(("Phe", "Ser", "Ala"), Release.MACROLACTAM)
    assert M.canonical_smiles(run(pep)) == M.canonical_smiles(run(pep))


# ---- guards ----------------------------------------------------------------------
def test_macrolactam_requires_two_residues():
    with pytest.raises(N.NRPSExecError):
        run(Peptide(("Gly",), Release.MACROLACTAM))


def test_unknown_residue_raises():
    with pytest.raises(N.NRPSExecError):
        run(Peptide(("Xaa", "Gly"), Release.HYDROLYSIS))


# ---- domain -> alphabet bridge ---------------------------------------------------
def test_alphabet_from_domains_te_enables_macrocyclization():
    with_te = NRPS.alphabet_from_domains({"C", "A", "T", "TE"})
    assert Release.MACROLACTAM in with_te.releases
    without_te = NRPS.alphabet_from_domains({"C", "A", "T"})
    assert without_te.releases == (Release.HYDROLYSIS,)


# ---- engine integration: shared verdict + ladder through the NRPS grammar --------
def test_engine_reconstructs_cyclo_glygly_from_mass():
    cho = formula_chno(M.mol_from_smiles(_DKP_GG))  # (C,H,N,O) incl. nitrogen
    res = infer_cluster({"C", "A", "T", "TE"}, 2, 2, target_cho=cho, grammar=NRPS)
    assert res.state is State.VERIFIED
    assert contains(res, _DKP_GG) == 1


def test_engine_nrps_ladder_monotone_nonincreasing():
    cho = formula_chno(M.mol_from_smiles(_DKP_GG))
    obs = Observables(NRPSAlphabet(), 2, 3, target_cho=cho)
    L = infer(obs, grammar=NRPS).ladder
    assert L["alphabet"] >= L["alphabet+mass"] >= L["alphabet+mass+msms"] >= 1


def test_engine_nrps_underobserved_without_mass_names_next_observable():
    obs = Observables(NRPSAlphabet(), 2, 3)  # no mass -> many peptides
    res = infer(obs, grammar=NRPS, ladder=False)
    assert res.state is State.UNDER_OBSERVED
    assert "mass" in res.next_observable


def test_engine_out_of_grammar_when_no_sequence_matches_mass():
    # A formula no short peptide over the alphabet can hit (absurd O count) -> empty -> out-of-grammar.
    res = infer(Observables(NRPSAlphabet(), 2, 2, target_cho=(99, 2, 1, 1)), grammar=NRPS, ladder=False)
    assert res.state is State.OUT_OF_GRAMMAR
