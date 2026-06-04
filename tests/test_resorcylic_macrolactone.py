"""The curated resorcylic-acid-lactone release: C2-C7 aromatization + cis-TE macrolactonization.

This promotes the (previously control-only) C2-C7 resorcylic fold into a registered grammar
operator, so a beta-resorcylic-acid lactone is reachable BY CONSTRUCTION (program -> structure),
exactly as orsellinic/6-MSA/mellein already are. The fold is the curated single-mode OrsA-family
C2-C7 register (relative soundness w.r.t. the curated grammar; sec:guarantees), NOT a register
read from observables -- the competing C1-C6 Claisen fold is deliberately absent.

Anchors are the retrieved/rendered SMILES validated in test_aromatization_control.py and the
MIBiG BGC0001057 deposited connectivity. Connectivity only; the executor is achiral.

NOTE (scope, rendered 2026-06-01): this makes zearalenone reachable by CONSTRUCTION, but NOT yet
by the reachability SCAN -- the C18 8-cycle no-C-MeT program is starved by the beam's
carbon-closeness ranking (C-MeT-heavy partials fill the beam at depth 7), a beam-completeness
limit independent of this operator. So these tests assert the executor/release behaviour, not
scan recovery.
"""

from __future__ import annotations

from rdkit import Chem

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.executor import core
from lpi.executor.cyclize import resorcylic_aromatic, resorcylic_macrolactone

# zearalenone's verified pre-aromatization nonaketide precursor (program below, hydrolysed).
PRECURSOR = "CC(O)CCCC(=O)CCCC=CC(=O)CC(=O)CC(=O)CC(=O)O"
SECO_ACID = "CC(O)CCCC(=O)CCCC=Cc1cc(O)cc(O)c1C(=O)O"     # aromatized open (resorcylic) form
ZEARALENONE = "CC1CCCC(=O)CCCC=Cc2cc(O)cc(O)c2C(=O)O1"    # retrieved: MIBiG BGC0001057, flat

ZEARALENONE_PROGRAM = Program(
    "acetyl",
    (Cycle(R.KR), Cycle(R.ER), Cycle(R.KETO), Cycle(R.ER), Cycle(R.DH),
     Cycle(R.KETO), Cycle(R.KETO), Cycle(R.KETO)),
    Release.RESORCYLIC_MACROLACTONE,
)
# orsellinic's non-reduced tetraketide acid: aromatizes, but the resorcylic acid it yields has no
# aliphatic hydroxyl, so the macrolactonization step cannot fire -> the composite returns None.
ORSELLINIC_PRECURSOR = "CC(=O)CC(=O)CC(=O)CC(=O)O"


def _canon(smi: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def test_resorcylic_aromatic_renders_seco_acid():
    seco = resorcylic_aromatic(Chem.MolFromSmiles(PRECURSOR))
    assert seco is not None
    assert Chem.MolToSmiles(seco) == _canon(SECO_ACID)


def test_resorcylic_macrolactone_renders_zearalenone_from_acid():
    prod = resorcylic_macrolactone(Chem.MolFromSmiles(PRECURSOR))
    assert prod is not None
    assert Chem.MolToSmiles(prod) == _canon(ZEARALENONE)


def test_release_dispatches_resorcylic_macrolactone_via_executor():
    """The full program z -> structure path: core.run renders the deposited zearalenone connectivity."""
    final = core.run(ZEARALENONE_PROGRAM).final
    assert final is not None
    assert Chem.MolToSmiles(final) == _canon(ZEARALENONE)


def test_resorcylic_macrolactone_sound_on_orsellinic():
    """Soundness: orsellinic's precursor aromatizes but yields no aliphatic hydroxyl, so the
    macrolactonization cannot fire and the composite returns None rather than a spurious product."""
    assert resorcylic_macrolactone(Chem.MolFromSmiles(ORSELLINIC_PRECURSOR)) is None


def test_resorcylic_aromatic_none_when_motif_absent():
    """A chain lacking the C2-C7 poly-beta-keto acid stretch does not aromatize."""
    assert resorcylic_aromatic(Chem.MolFromSmiles("CCCCCCCC(=O)O")) is None


def test_resorcylic_macrolactone_excludes_delta_lactone():
    """Isolation: the composite is a MACROlactone operator. On a short chain whose closure would be a
    6-membered delta-lactone (the dihydroisocoumarin regime, e.g. 6-hydroxymellein's acetyl+[KR,KETO,
    KETO,KETO] chain), it returns None rather than poaching that core -- regression for the
    6-hydroxymellein |Z*| 1->2 inflation the macrocycle bound fixes."""
    delta_lactone_chain = core.run(
        Program("acetyl", (Cycle(R.KR), Cycle(R.KETO), Cycle(R.KETO), Cycle(R.KETO)),
                Release.HYDROLYSIS)).linear
    assert resorcylic_macrolactone(delta_lactone_chain) is None


def test_zearalenone_program_within_scan_grammar():
    """Theta_0-reachability (the generability witness's load-bearing premise): EVERY operator the
    zearalenone program uses is in the scan grammar `_scan_spec()` -- acetyl starter, malonyl extender,
    {KR,ER,KETO,DH} reductions, no C-MeT, RESORCYLIC_MACROLACTONE release, n_cycles within cap. Together
    with test_release_dispatches... (the program renders BGC0001057), this proves zearalenone is in the
    UNPRUNED reachable set under Theta_0 (kappa>=1). With the kappa-monotonicity certificate
    (kappa_beta <= kappa, non-decreasing in beta; Prop 16.4c / make kappa-check), it follows that
    zearalenone is 'recoverable at sufficient beta' -- so the generability witness's kappa_beta=0 through
    beta=200000 is a beam UNDER-COUNT (an engineering/beam limit), NOT a Theta_0 coverage gap. This is the
    fact that makes the `coverage-limited-structural` class beam-contaminated rather than a clean
    coverage certificate."""
    from lpi.search.reachability import _scan_spec

    S = _scan_spec()
    p = ZEARALENONE_PROGRAM
    assert p.starter in S.starters
    assert all(c.reduction in S.reductions for c in p.cycles)
    assert all(c.extender in S.extenders for c in p.cycles)
    assert (not any(c.c_methyl for c in p.cycles)) or S.allow_c_methyl
    assert p.release in S.releases
    assert S.max_cycles is None or p.n_cycles <= S.max_cycles
