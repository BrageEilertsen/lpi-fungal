"""The PT-domain pentaketide -> naphthalene register: witness-validated, soundly motif-gated.

`cyclize_naphthalene` renders the deposited 1,3,6,8-tetrahydroxynaphthalene (MIBiG BGC0001257/0001258)
EXACTLY from the all-keto pentaketide program, and returns None on every other chain -- the Exp-A
resorcylic soundness pattern (round-trip the deposited witness + None outside the motif). This locks the
operator as the curated single-mode PT naphthalene register and certifies it for promotion into the sound
base universe Theta_0 (the rung-2 build-loop's proof-of-loop step). It also pins the correction of the prior
"skeleton-correct, not exact-match" docstring, which the deposited witness refutes.
"""
from __future__ import annotations

from rdkit import Chem

from lpi.chem.program import Cycle, Program, Release, ReductionState as R
from lpi.executor import core
from lpi.search.beam import _formula_cho

# deposited 1,3,6,8-tetrahydroxynaphthalene -- MIBiG BGC0001257 / BGC0001258 (also the BGC0002154 deposit)
THN = "Oc1cc(O)c2c(O)cc(O)cc2c1"
NAPHTHALENE_PROGRAM = Program("acetyl", (Cycle(R.KETO),) * 4, Release.PT_NAPHTHALENE)


def _flat(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)


def test_pt_naphthalene_renders_deposited_thn():
    """z -> structure: the all-keto pentaketide + PT_NAPHTHALENE renders the deposited 1,3,6,8-THN EXACTLY
    (canonical-SMILES identity, C10H8O4). The witness, not recalled SMARTS, is the curation authority."""
    final = core.run(NAPHTHALENE_PROGRAM).final
    assert final is not None
    assert Chem.MolToSmiles(final, isomericSmiles=False) == _flat(THN)
    assert _formula_cho(final) == (10, 8, 4)


def test_pt_naphthalene_none_off_motif():
    """Soundness: PT_NAPHTHALENE fires ONLY on the all-keto pentaketide; it returns None on every other
    chain (wrong length, any reduction, non-acetyl starter), so promoting it into Theta_0 cannot poach or
    spuriously recover a non-THN core."""
    for p in (
        Program("acetyl", (Cycle(R.KETO),) * 3, Release.PT_NAPHTHALENE),                  # tetraketide
        Program("acetyl", (Cycle(R.KETO),) * 5, Release.PT_NAPHTHALENE),                  # hexaketide
        Program("acetyl", (Cycle(R.KR),) + (Cycle(R.KETO),) * 3, Release.PT_NAPHTHALENE), # reduced
        Program("propionyl", (Cycle(R.KETO),) * 4, Release.PT_NAPHTHALENE),               # non-acetyl starter
    ):
        assert core.run(p).final is None


def test_pt_naphthalene_chain_within_scan_grammar():
    """Theta-extension premise: the naphthalene program's chain (acetyl starter, all-KETO reductions, no
    C-MeT, malonyl extender) is already in the scan grammar `_scan_spec()`; the ONLY missing piece is the
    Release.PT_NAPHTHALENE register. So adding it to _ALL_RELEASES is the minimal sound Theta-extension that
    moves the THN cores gap -> reachable (the build-loop's first monotone step)."""
    from lpi.search.reachability import _scan_spec

    S = _scan_spec()
    p = NAPHTHALENE_PROGRAM
    assert p.starter in S.starters
    assert all(c.reduction in S.reductions for c in p.cycles)
    assert all(not c.c_methyl for c in p.cycles)
