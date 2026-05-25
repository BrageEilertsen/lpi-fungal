"""Phase 1 verifier/search tests.

Gate 1 core requirement: the known curated program is contained in Z*(y) for every
gate system. Plus determinism, pruning, and a genuine |Z*(y)| > 1 degeneracy case.
"""

from __future__ import annotations

import pytest

from lpi.data.curated import load_all
from lpi.search.beam import OperatorSpec, search

_GATE = [e for e in load_all() if e.tier == "gate"]


def _target(entry):
    return (entry.expected_final_smiles if entry.scored_level == "cyclized"
            else entry.expected_linear_smiles)


@pytest.mark.parametrize("entry", _GATE, ids=lambda e: e.source_path.stem)
def test_known_program_in_z_star(entry):
    result = search(_target(entry))
    assert result.size >= 1, f"{entry.name}: Z*(y) is empty"
    assert any(p == entry.program for p in result.z_star), (
        f"{entry.name}: known program not recovered in Z*(y)\n"
        f"  known: {entry.program}\n  found: {result.z_star}"
    )


def test_search_is_deterministic():
    a = search("Cc1cc(O)cc(O)c1C(=O)O")
    b = search("Cc1cc(O)cc(O)c1C(=O)O")
    assert [repr(p) for p in a.z_star] == [repr(p) for p in b.z_star]


def test_pruning_runs():
    r = search("CCCCCCCC(=O)O")  # octanoic; many overshoot partials pruned
    assert r.stats.n_pruned_overshoot > 0


def test_genuine_degeneracy_starter_ambiguity():
    # hexanoic acid = acetyl + 2 ER  OR  butyryl + 1 ER -> |Z*(y)| = 2
    r = search("CCCCCC(=O)O", OperatorSpec(starters=("acetyl", "propionyl", "butyryl")))
    assert r.size == 2
    starters = sorted(p.starter for p in r.z_star)
    assert starters == ["acetyl", "butyryl"]


def test_failed_cyclization_does_not_inflate_z_star():
    # A linear acid must not also match via a non-firing cyclization release.
    r = search("CCC(C)C(=O)O")  # 2-methylbutyric acid
    assert r.size == 1
    assert r.z_star[0].release.value == "hydrolysis"
