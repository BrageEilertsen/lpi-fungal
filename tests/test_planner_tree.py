"""Planner item 5b: the decision-tree output (both branches first-class; Mermaid + JSON)."""
from __future__ import annotations

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.planner.plan import TriageVerdict
from lpi.planner.tree import decision_tree, to_dict, to_json, to_mermaid
from lpi.realizability import natural_manifold, realize

_MAN = natural_manifold()
_A = Release.ALDOL_AROMATIC
_6MSA = realize(next(t for t in _MAN if "6-MSA" in t.id).program, _MAN)
_BAB = realize(next(t for t in _MAN if "BAB" in t.id).program, _MAN)


def _walk(node):
    yield node
    for b in node.branches:
        yield from _walk(b.child)


def test_every_decision_node_has_both_branches_first_class():
    # the load-bearing invariant: refuted is never collapsed into a footnote
    for design in (_6MSA, _BAB):
        for n in _walk(decision_tree(design)):
            if not n.is_terminal:
                labels = {b.outcome for b in n.branches}
                assert labels == {"confirmed", "refuted"}                 # both, always
                assert abs(sum(b.probability for b in n.branches) - 1.0) < 1e-9


def test_6msa_tree_is_shallow_verified_at_mass():
    root = decision_tree(_6MSA)
    assert root.recommend == "mass" and not root.is_terminal
    confirmed = next(b.child for b in root.branches if b.outcome == "confirmed")
    refuted = next(b.child for b in root.branches if b.outcome == "refuted")
    assert confirmed.verdict is TriageVerdict.VERIFIED_BY_TRIAGE      # planner doing its job
    assert refuted.verdict is TriageVerdict.OUT_OF_GRAMMAR            # the negative case, first-class


def test_bab_tree_exhausts_modelled_observables_to_elucidation():
    root = decision_tree(_BAB)
    assert root.recommend == "mass"
    after_mass = next(b.child for b in root.branches if b.outcome == "confirmed")
    assert after_mass.recommend == "msms"                            # mass didn't collapse it
    after_msms = next(b.child for b in after_mass.branches if b.outcome == "confirmed")
    assert after_msms.verdict is TriageVerdict.ELUCIDATION_REQUIRED  # observables exhausted -> its limit
    assert after_msms.z_star > 5


def test_json_serialization_round_trips_structure():
    d = to_dict(decision_tree(_6MSA))
    assert d["recommend"] == "mass" and len(d["branches"]) == 2
    assert {b["outcome"] for b in d["branches"]} == {"confirmed", "refuted"}
    assert "verified-by-triage" in to_json(decision_tree(_6MSA))


def test_mermaid_emits_both_outcomes_and_verdicts():
    m = to_mermaid(decision_tree(_BAB), title="BAB")
    assert m.startswith("graph TD")
    assert "confirmed p=" in m and "refuted p=" in m                 # both edges rendered
    assert "elucidation-required" in m and "out-of-grammar" in m
