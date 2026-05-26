"""PKS-NRPS hybrid bridge (Track 2.5): the seven acceptance criteria.

  1. existing PKS + NRPS tests still green                 -> the rest of the suite
  2. the hybrid grammar COMPOSES the PKS + NRPS generators, it does not duplicate them
  3. the handoff is a typed, explicit operator (not a hidden special case)
  4. a simple synthetic PKS-NRPS round-trip passes (independent ground truth)
  5. a TenS-like checkpoint works up to the known bridge/scope boundary
  6. the mass/adduct/MS-MS/verdict layer runs UNCHANGED on hybrid candidates
  7. an unimplemented release is a TYPED failure (ReleaseNotImplemented / OUT_OF_GRAMMAR), not silent

The claim is typed compositionality, not PKS-NRPS biochemical completeness.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.engine import State, contains
from lpi.executor import core
from lpi.executor import hybrid as H
from lpi.executor.hybrid import Hybrid, HybridRelease, ReleaseNotImplemented
from lpi.executor.nrps import RESIDUES, formula_chno
from lpi.grammars import HYBRID
from lpi.grammars.hybrid import HybridAlphabet
from lpi.engine import frag_fingerprint
from lpi.observe import ADDUCTS, MSObservables, exact_mass, infer_ms, ion_mz

# synthetic hybrid: acetyl + 1 keto cycle = acetoacetic acid; handoff Gly -> acetoacetyl-glycine
_PK_DIKETIDE = Program("acetyl", (Cycle(reduction=R.KETO),), Release.HYDROLYSIS)
_ACETOACETYL_GLY = "CC(=O)CC(=O)NCC(=O)O"            # independent ground truth
# TenS-like: a pentaketide handed off to tyrosine (the polyketide-tetramic-acid hybrid family)
_PK_PENTAKETIDE = Program("acetyl", (Cycle(R.KR), Cycle(R.DH), Cycle(R.KR), Cycle(R.KETO)),
                          Release.HYDROLYSIS)


# ---- (4) synthetic round-trip ---------------------------------------------------
def test_synthetic_roundtrip_matches_ground_truth():
    got = M.canonical_smiles(H.run(Hybrid(_PK_DIKETIDE, ("Gly",), HybridRelease.HYDROLYSIS)))
    assert got == M.canonical_smiles(M.mol_from_smiles(_ACETOACETYL_GLY))


def test_hybrid_formula_composes_and_matches_built():
    for residues in [("Gly",), ("Phe",), ("Ala",)]:
        analytic = H.hybrid_formula(_PK_DIKETIDE, residues)
        built = formula_chno(H.run(Hybrid(_PK_DIKETIDE, residues, HybridRelease.HYDROLYSIS)))
        assert analytic == built, f"{residues}: {analytic} != {built}"


# ---- (3) typed, explicit handoff ------------------------------------------------
def test_handoff_forms_an_amide_bond():
    mol = H.run(Hybrid(_PK_DIKETIDE, ("Gly",), HybridRelease.HYDROLYSIS))
    # the handoff introduces an amide C(=O)-N joining the PK acyl to the peptide
    assert mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))


def test_unimplemented_release_is_typed_not_silent():
    with pytest.raises(ReleaseNotImplemented):
        H.run(Hybrid(_PK_DIKETIDE, ("Gly",), HybridRelease.TETRAMIC_ACID))
    assert HybridRelease.TETRAMIC_ACID not in H.IMPLEMENTED_RELEASES


# ---- (2) composes the two generators (does not duplicate) -----------------------
def test_hybrid_program_is_pks_program_plus_nrps_residues():
    res = _infer_acetoacetyl_gly()
    cand = res.candidates[contains(res, _ACETOACETYL_GLY) - 1]
    prog = cand.program
    assert isinstance(prog, Hybrid)
    assert isinstance(prog.pks, Program)                       # reuses the PKS program representation
    assert all(r in RESIDUES for r in prog.residues)           # reuses the NRPS monomer table
    # and the PKS executor renders the PK part to a carboxylic acid -- the handoff input
    assert core.run(prog.pks).linear.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))


# ---- (5) TenS-like checkpoint up to the scope boundary --------------------------
def test_tens_like_checkpoint_runs_but_tetramic_release_is_a_boundary():
    # the polyketide backbone + tyrosine amide (the linear checkpoint) renders fine ...
    mol = H.run(Hybrid(_PK_PENTAKETIDE, ("Tyr",), HybridRelease.HYDROLYSIS))
    assert mol.GetNumAtoms() > 0
    assert mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[NX3]"))   # PK->Tyr amide present
    # ... while the tetramic-acid Dieckmann release is the declared, unimplemented boundary
    with pytest.raises(ReleaseNotImplemented):
        H.run(Hybrid(_PK_PENTAKETIDE, ("Tyr",), HybridRelease.TETRAMIC_ACID))


# ---- (6) shared observability/verdict layer runs UNCHANGED on hybrid candidates --
def _infer_acetoacetyl_gly():
    mz = ion_mz(exact_mass(_ACETOACETYL_GLY), ADDUCTS["[M+H]+"])
    return infer_ms(HybridAlphabet(), MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5),
                    HYBRID, 1, 2)


def test_shared_observability_layer_verifies_hybrid():
    res = _infer_acetoacetyl_gly()                  # same infer_ms as PKS/NRPS, only grammar=HYBRID
    assert res.state is State.VERIFIED
    assert contains(res, _ACETOACETYL_GLY) == 1
    assert res.formulas == ["C6H9NO4"]


def test_hybrid_ladder_monotone_and_grammar_relative():
    mz = ion_mz(exact_mass(_ACETOACETYL_GLY), ADDUCTS["[M+H]+"])
    peaks = tuple(frag_fingerprint(_ACETOACETYL_GLY))
    res = infer_ms(HybridAlphabet(), MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5,
                                                   msms_peaks=peaks), HYBRID, 1, 2)
    L = res.ladder
    assert L["genome"] >= L["+mass"] >= L["+msms"] >= 1
    assert "hybrid" in res.reason


# ---- (7) engine-level: a cluster whose only release is unimplemented -> OUT_OF_GRAMMAR
def test_only_unimplemented_release_is_out_of_grammar():
    alpha = HybridAlphabet(releases=(HybridRelease.TETRAMIC_ACID,))   # no renderable release
    mz = ion_mz(exact_mass(_ACETOACETYL_GLY), ADDUCTS["[M+H]+"])
    res = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=("[M+H]+",), ppm=5), HYBRID, 1, 2)
    assert res.state is State.OUT_OF_GRAMMAR
    assert "operator grammar" in res.reason          # advice names the operator/release gap
