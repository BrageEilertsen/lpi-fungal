"""Core-match scoring tests (Step 3, tightened): skeleton-complete vs subgraph-only."""

from __future__ import annotations

from lpi.chem import mol as M
from lpi.eval.corematch import (any_subgraph_match, best_skeleton_complete,
                                build_core_library, coverage, skeleton_complete_match)

_LIB = build_core_library()


def test_library_has_cyclic_ring_systems():
    cyclic = [c for c in _LIB if c.is_cyclic]
    assert len(cyclic) >= 4
    assert any(c.n_carbons == 10 and c.smiles.count("c") >= 10 for c in cyclic)  # naphthalene


def test_skeleton_complete_absorbs_separable_decorations():
    mellein = M.mol_from_smiles("CC1Cc2cccc(O)c2C(=O)O1")
    # add ONLY single-bond-separable decorations: prenyl + extra alkyl + OH
    tailored = M.mol_from_smiles("CCCCC1Cc2cc(O)c(CC=C(C)C)c(O)c2C(=O)O1")
    assert skeleton_complete_match(mellein, tailored)


def test_subgraph_embed_but_not_skeleton_complete_for_fused_extra_ring():
    # benzene core vs a fused bicyclic where the extra ring attaches by TWO bonds:
    # it embeds as a subgraph but is NOT skeleton-complete (fused = part of skeleton).
    benzene_core = M.mol_from_smiles("Cc1cc(O)cc(O)c1C(=O)O")     # resorcylate, one ring
    naphthalene_y = M.mol_from_smiles("Cc1cc(O)cc(O)c1-c1ccccc1") # biaryl: 2nd ring via 1 bond
    fused_y = M.mol_from_smiles("Oc1cc(O)c2cc(O)cc(O)c2c1")        # fused naphthalene
    # fused second ring attaches by 2 bonds -> not skeleton-complete by a single benzene core
    assert not skeleton_complete_match(benzene_core, fused_y)


def test_linear_core_into_larger_ring_is_rejected():
    # a short linear core must NOT skeleton-complete-match a big fused ring system as a
    # sub-path (unmatched ring carbons attach by >=2 bonds).
    short = M.mol_from_smiles("CCCCCCC(=O)O")  # C7 chain
    big_fused = M.mol_from_smiles("Oc1cc(O)c2cc(O)cc(O)c2c1")  # naphthalene, 10 C
    assert not skeleton_complete_match(short, big_fused)


def test_coverage_one_for_untailored():
    orsellinic = M.mol_from_smiles("Cc1cc(O)cc(O)c1C(=O)O")
    m = best_skeleton_complete(orsellinic, _LIB)
    assert m is not None and coverage(m, orsellinic) == 1.0


def test_ceiling_when_no_core_embeds():
    tiny = M.mol_from_smiles("c1ccncc1")  # pyridine, 5 C
    assert best_skeleton_complete(tiny, _LIB) is None
