"""Lanthipeptide (RiPP) grammar -- SCAFFOLD tests (paper Sec. 16.14, the cross-family crossing).

This grammar is deliberately a scaffold: it owns the SOUND combinatorics (topology DOF +
mass bookkeeping) and STUBS the structural renderers (raise, pending chemist-signed SMARTS +
an independent structure anchor). These tests therefore split in two:

  * the SOUND part is validated against a REAL, RETRIEVED molecule -- lacticin 481 -- with the
    Remark-16.14a containment check and the Cor-16.4 O-knob monotonicity holding on real data;
  * the GATED part is pinned by TRIPWIRE tests asserting the renderers still raise. Those are the
    honest gate in executable form: un-stubbing a renderer (claiming a structural =~-datapoint)
    must deliberately flip a tripwire, never slip through silently.

Lacticin 481 ground truth (retrieved, NOT recalled; carried with provenance):
  * precursor LctA, 51 aa: MKEQNSFNLLQEVTESELDLILGAKGGSGVIHTISHECNMNSWQFVFTCCS
    -- UniProt P36499 (LAN4_LACLL, gene lctA), .fasta + .txt feature table.
  * leader 1-24, core 25-51 (27 aa: KGGSGVIHTISHECNMNSWQFVFTCCS).
  * UniProt curated CROSSLNK features (PRECURSOR numbering): 33-38 beta-methyllanthionine
    (Thr-Cys), 35-49 lanthionine (Ser-Cys), 42-50 lanthionine (Ser-Cys); MOD_RES 48 = (Z)-2,3-
    didehydrobutyrine (a FREE Dhb). In CORE-local numbering (precursor-24): bridges 9-14, 11-25,
    18-26 -- which reproduces the primary NMR structure of van den Hooven et al., FEBS Lett. 1996
    (PMID 8764998); cf. total synthesis Knerr & van der Donk, JACS 2013 (PMC3736828).

No SMILES is rendered here and no structural =~-datapoint is claimed. The =~-gate needs an
INDEPENDENT, experimentally-deposited ATOMIC structure the model did not construct. A connectivity
anchor (sequence + bridge table) does NOT qualify -- it is exactly the engine's INPUT, so building
a structure from it and =~-matching the render against it passes by construction and certifies
nothing (the mirror, re-introduced; it does not bypass the gate, it reinstates the circularity).
No such deposited atomic structure exists machine-readable for this 27-mer (PubChem/NPAtlas/ChEBI),
and that under-deposition of multi-ring RiPPs is itself consistent with the product-resolvable-but-
under-resolved U-verdict the framework assigns them. The gate therefore stays correctly blocked:
the honest terminal state, and the one the theory predicted.
"""

from __future__ import annotations

import math

import pytest

from lpi.grammars import REGISTRY, get
from lpi.grammars.base import Grammar
from lpi.grammars.lanthipeptide import (
    BRIDGE_DONOR,
    DEHYDRATABLE,
    LANTHI,
    LanthiGrammar,
    _pairings,
    _render_dehydration,
    _render_lanthionine_bridge,
    _render_leader_cleavage,
    partition,
)

# --- the retrieved, provenanced ground truth (see module docstring) ---------------
LACTICIN_CORE = "KGGSGVIHTISHECNMNSWQFVFTCCS"          # UniProt P36499, residues 25-51
# true topology in 0-based CORE indices (core 1-based - 1):
#   9-14  -> (cys 13, dha 8)   Thr9 -Cys14  methyllanthionine
#  11-25  -> (cys 24, dha 10)  Ser11-Cys25  lanthionine
#  18-26  -> (cys 25, dha 17)  Ser18-Cys26  lanthionine
LACTICIN_TRUE_PAIRING = tuple(sorted([(13, 8), (24, 10), (25, 17)]))   # _pairings key shape (cys,dha)
LACTICIN_TRUE_DHA = (8, 10, 17, 23)   # residues that lose H2O: 3 bridged + Thr(core 24, 0b 23) free Dhb


# =====================================================================================
# SOUND part -- validated on the real, retrieved lacticin 481 core
# =====================================================================================
def test_lacticin_core_shape_matches_uniprot():
    assert len(LACTICIN_CORE) == 27
    alpha = LANTHI.alphabet_from_domains(domains=None, core=LACTICIN_CORE)
    assert alpha.cys == (13, 24, 25)                       # the three Cys bridge donors
    assert alpha.dehydrated == (3, 8, 10, 17, 23, 26)      # all six Ser/Thr (v0 conservative superset)


def test_lacticin_true_bridge_endpoints_are_serthr_to_cys():
    # every annotated bridge connects a dehydratable Ser/Thr to a Cys, as UniProt states.
    for cys, dha in LACTICIN_TRUE_PAIRING:
        assert LACTICIN_CORE[cys] == BRIDGE_DONOR
        assert LACTICIN_CORE[dha] in DEHYDRATABLE


def test_lacticin_permissive_kappa_is_falling_factorial_and_contains_truth():
    # genome-only knowledge: any of the 6 S/T could be the Dha; every injective Cys->Dha pairing.
    p = partition(LACTICIN_CORE)                            # default dehydrate = all S/T
    assert p["kappa_mass"] == math.perm(6, 3) == 120        # P(6,3): conservative upper bound
    assert p["topology_U"] is True
    pairings = _pairings((13, 24, 25), (3, 8, 10, 17, 23, 26))
    assert LACTICIN_TRUE_PAIRING in pairings                # Remark 16.14a containment, real case


def test_lacticin_positional_kappa_and_containment():
    # given the retrieved dehydration POSITIONS (a positional observable), the set tightens.
    p = partition(LACTICIN_CORE, dehydrate=LACTICIN_TRUE_DHA)
    assert p["kappa_mass"] == math.perm(4, 3) == 24         # P(4,3)
    assert p["topology_U"] is True
    assert LACTICIN_TRUE_PAIRING in _pairings((13, 24, 25), LACTICIN_TRUE_DHA)


def test_lacticin_observable_monotone_nonincreasing():
    # Cor 16.4 O-knob on real data: adding the positional observable cannot grow kappa.
    k_perm = partition(LACTICIN_CORE)["kappa_mass"]
    k_obs = partition(LACTICIN_CORE, dehydrate=LACTICIN_TRUE_DHA)["kappa_mass"]
    assert k_perm >= k_obs >= 1


def test_lacticin_modality_cell_is_trajectory_product():
    p = partition(LACTICIN_CORE, dehydrate=LACTICIN_TRUE_DHA)
    assert p["genome"]["verdict"] == "R"                    # core sequence is genome-read
    assert p["mass"]["verdict"] == "U"                      # mass-neutral bridging => isomass topologies
    assert p["product"]["verdict"] == "PENDING-renderers"   # MS/MS-across-rings, pending
    assert p["trajectory"]["verdict"] == "trajectory-controlled"
    assert "(trajectory, product)" in p["cell"]


# =====================================================================================
# SOUND part -- combinatorial properties of the permissive enumeration
# =====================================================================================
@pytest.mark.parametrize("core,n_cys,n_dha", [
    ("SC", 1, 1), ("SCSC", 2, 2), ("SCSCSC", 3, 3),
    ("SCCSS", 2, 3), ("SSSC", 1, 3),
])
def test_pairings_count_is_falling_factorial(core, n_cys, n_dha):
    cys = tuple(i for i, a in enumerate(core) if a == BRIDGE_DONOR)
    dha = tuple(i for i, a in enumerate(core) if a in DEHYDRATABLE)
    assert (len(cys), len(dha)) == (n_cys, n_dha)
    assert len(_pairings(cys, dha)) == math.perm(n_dha, n_cys)


def test_pairings_toy_cores_match_module_demo():
    # the values printed by the module's __main__ banner.
    assert partition("SC")["kappa_mass"] == 1
    assert partition("SCSC")["kappa_mass"] == 2
    assert partition("SCSCSC")["kappa_mass"] == 6


def test_pairings_are_injective():
    # each Cys bonds a distinct Dha (no Dha reused) across every enumerated topology.
    for pairing in _pairings((1, 3), (0, 2, 4)):
        dhas = [d for _c, d in pairing]
        assert len(dhas) == len(set(dhas))


def test_pairings_are_permissive_no_ring_size_prior():
    # the crossing (non-nested) topology must be present: enumeration applies no ring-size filter.
    pairings = _pairings((1, 3), (0, 2))                    # cys 1,3 ; dha 0,2
    crossing = tuple(sorted([(1, 2), (3, 0)]))              # cys1->dha2, cys3->dha0 (crossing)
    assert crossing in pairings


def test_pairings_edge_cases():
    assert _pairings((), (0, 2)) == [()]                    # no Cys -> one (empty) topology, kappa=1
    assert _pairings((1,), ()) == []                        # Cys with no Dha acceptor -> none


def test_topology_dof_len_equals_kappa_mass():
    for core in ["SCSC", "SCSCSC", LACTICIN_CORE]:
        alpha = LANTHI.alphabet_from_domains(domains=None, core=core)
        assert len(LANTHI.topology_dof(alpha)) == partition(core)["kappa_mass"]


# =====================================================================================
# Grammar protocol conformance -- plugs into the one engine interface
# =====================================================================================
def test_lanthi_satisfies_grammar_protocol_and_is_registered():
    assert isinstance(LANTHI, Grammar)
    assert isinstance(LANTHI, LanthiGrammar)
    assert LANTHI.name == "lanthipeptide"
    assert REGISTRY["lanthipeptide"] is LANTHI
    assert get("lanthipeptide") is LANTHI


# =====================================================================================
# GATED part -- TRIPWIRES. These assert the renderers still raise. The honest gate, in code:
# un-stubbing one (claiming a structural datapoint) must deliberately flip a tripwire here.
# =====================================================================================
def test_tripwire_enumerate_is_stubbed():
    alpha = LANTHI.alphabet_from_domains(domains=None, core=LACTICIN_CORE)
    with pytest.raises(NotImplementedError):
        LANTHI.enumerate(alpha)


def test_tripwire_mass_prefilter_is_stubbed():
    with pytest.raises(NotImplementedError):
        LANTHI.mass_prefilter_keys(500.0, 10.0)


def test_tripwire_structural_renderers_are_stubbed():
    for render in (_render_dehydration, _render_lanthionine_bridge, _render_leader_cleavage):
        with pytest.raises(NotImplementedError):
            render()
