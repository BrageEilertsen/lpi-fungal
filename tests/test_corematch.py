"""Core-match scoring tests (Step 3): carbon-skeleton subgraph + coverage."""

from __future__ import annotations

from lpi.chem import mol as M
from lpi.eval.corematch import (best_core_match, build_core_library, core_in_target,
                                coverage)

_LIB = build_core_library()


def test_library_has_cyclic_ring_systems():
    cyclic = [c for c in _LIB if c.is_cyclic]
    # pyranone, resorcylate benzene, mellein dihydroisocoumarin, naphthalene
    assert len(cyclic) >= 4
    assert any(c.n_carbons == 10 and c.smiles.count("c") >= 10 for c in cyclic)  # naphthalene


def test_additive_tailoring_is_absorbed():
    mellein = M.mol_from_smiles("CC1Cc2cccc(O)c2C(=O)O1")
    tailored = M.mol_from_smiles("CCCCC1Cc2cc(O)c(CC=C(C)C)c(O)c2C(=O)O1")  # +prenyl +OH +alkyl
    assert core_in_target(mellein, tailored)


def test_ring_aware_matching_rejects_wrong_ring_system():
    orsellinic = M.mol_from_smiles("Cc1cc(O)cc(O)c1C(=O)O")  # benzene
    mellein = M.mol_from_smiles("CC1Cc2cccc(O)c2C(=O)O1")    # bicyclic, smaller benzene+lactone
    # orsellinic's benzene-with-two-substituents should not embed in mellein's skeleton
    assert not core_in_target(orsellinic, mellein)


def test_coverage_is_one_for_untailored_match():
    orsellinic = M.mol_from_smiles("Cc1cc(O)cc(O)c1C(=O)O")
    m = best_core_match(orsellinic, _LIB)
    assert m is not None
    assert coverage(m, orsellinic) == 1.0


def test_ceiling_when_no_core_embeds():
    # a small heteroaromatic with no producible carbon skeleton of >=6 C
    tiny = M.mol_from_smiles("c1ccncc1")  # pyridine, 5 C
    assert best_core_match(tiny, _LIB) is None
