"""Planner item 1: the design-resolution lattice with three-valued, probabilistic transitions."""
from __future__ import annotations

import pytest

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.planner.state import (
    OBSERVATION_COSTS,
    DesignState,
    Observation,
    Resolution,
    available_transitions,
    transition,
    uninformative_forward_model,
)
from lpi.realizability import Axis, natural_manifold, realize

_MAN = natural_manifold()
_ALDOL = Release.ALDOL_AROMATIC

# 6-MSA = acetyl (K, KR, K). Designs at known edit-distances from it:
_ONE_CONTROL = realize(Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), _ALDOL), _MAN)
_TWO_CONTROL = realize(Program("acetyl", (Cycle(R.KR), Cycle(R.KR), Cycle(R.KR)), _ALDOL), _MAN)
_NATURAL = realize(next(t for t in _MAN if "6-MSA" in t.id).program, _MAN)

_MASS = Observation("mass", OBSERVATION_COSTS["mass"])
_MSMS = Observation("msms", OBSERVATION_COSTS["msms"])


def test_design_setup_is_as_expected():
    assert _ONE_CONTROL.control_cost == 1 and _ONE_CONTROL.structural_cost == 0
    assert _TWO_CONTROL.control_cost == 2 and _TWO_CONTROL.structural_cost == 0
    assert _NATURAL.verdict == "natural" and len(_NATURAL.edits) == 0


def test_initial_state_all_unresolved():
    s = DesignState.initial(_ONE_CONTROL)
    assert s.status == (Resolution.UNRESOLVED,)
    assert s.unresolved_indices() == [0] and not s.is_terminal


def test_natural_design_initial_state_is_terminal():
    # a natural product has no edits -> the lattice is a single terminal node (degenerate case)
    s = DesignState.initial(_NATURAL)
    assert s.is_terminal and s.unresolved_indices() == [] and s.status == ()


def test_with_resolution_is_immutable_and_three_valued():
    s = DesignState.initial(_TWO_CONTROL)
    s1 = s.with_resolution(0, Resolution.CONFIRMED)
    assert s1.status[0] is Resolution.CONFIRMED and s1.status[1] is Resolution.UNRESOLVED
    assert s.status == (Resolution.UNRESOLVED, Resolution.UNRESOLVED)   # original untouched
    s2 = s.with_resolution(0, Resolution.REFUTED)                      # refuted is a distinct value
    assert s2.status[0] is Resolution.REFUTED
    with pytest.raises(ValueError):
        s1.with_resolution(0, Resolution.CONFIRMED)                    # cannot re-resolve
    with pytest.raises(ValueError):
        s.with_resolution(0, Resolution.UNRESOLVED)                    # cannot resolve TO unresolved


def test_transition_is_probabilistic_with_two_branches():
    s = DesignState.initial(_ONE_CONTROL)
    outcomes = transition(s, _MASS, 0, lambda st, o, i: 0.7)
    assert {o.label for o in outcomes} == {"confirmed", "refuted"}
    assert abs(sum(o.probability for o in outcomes) - 1.0) < 1e-9
    conf = next(o for o in outcomes if o.label == "confirmed")
    refu = next(o for o in outcomes if o.label == "refuted")
    assert conf.probability == pytest.approx(0.7) and refu.probability == pytest.approx(0.3)
    assert conf.next_state.status[0] is Resolution.CONFIRMED
    assert refu.next_state.status[0] is Resolution.REFUTED


def test_transition_rejects_resolved_edit_and_bad_probability():
    s = DesignState.initial(_ONE_CONTROL).with_resolution(0, Resolution.CONFIRMED)
    with pytest.raises(ValueError):
        transition(s, _MASS, 0, uninformative_forward_model)          # edit 0 already resolved
    s0 = DesignState.initial(_ONE_CONTROL)
    with pytest.raises(ValueError):
        transition(s0, _MASS, 0, lambda st, o, i: 1.7)                # probability out of range


def test_uninformative_forward_model_is_fifty_fifty():
    outcomes = transition(DesignState.initial(_ONE_CONTROL), _MASS, 0)
    assert all(o.probability == pytest.approx(0.5) for o in outcomes)


def test_residual_cost_is_super_additive_and_drops_nonlinearly_on_resolution():
    s1 = DesignState.initial(_ONE_CONTROL)
    s2 = DesignState.initial(_TWO_CONTROL)
    # two stacked control edits cost MORE than twice one (quadratic-in-count, PLANNER_SKETCH Q3)
    assert s2.residual_realizability_cost() > 2 * s1.residual_realizability_cost()
    # resolving one of the two stacked control edits drops the residual super-linearly (not merely halved)
    after = s2.with_resolution(0, Resolution.CONFIRMED)
    assert after.residual_realizability_cost() < s2.residual_realizability_cost() / 2
    # a refuted edit's cost also drops out of the residual (it is settled, not pending)
    refuted = s2.with_resolution(0, Resolution.REFUTED)
    assert refuted.residual_realizability_cost() == after.residual_realizability_cost()


def test_residual_cost_zero_when_terminal():
    s = DesignState.initial(_ONE_CONTROL).with_resolution(0, Resolution.CONFIRMED)
    assert s.is_terminal and s.residual_realizability_cost() == 0


def test_available_transitions_enumerate_obs_times_unresolved_edges():
    s = DesignState.initial(_TWO_CONTROL)
    edges = available_transitions(s, [_MASS, _MSMS])
    assert len(edges) == 2 * 2                                         # 2 observations x 2 unresolved edits
    s1 = s.with_resolution(0, Resolution.CONFIRMED)
    assert len(available_transitions(s1, [_MASS, _MSMS])) == 2 * 1     # one edit left
    term = DesignState.initial(_NATURAL)
    assert available_transitions(term, [_MASS, _MSMS]) == []           # terminal -> no edges


def test_state_equality_and_hash_use_design_and_status():
    a = DesignState.initial(_ONE_CONTROL)
    b = DesignState.initial(_ONE_CONTROL)
    assert a == b and hash(a) == hash(b)                              # same design + status
    assert a != a.with_resolution(0, Resolution.CONFIRMED)           # status distinguishes
    assert DesignState.initial(_ONE_CONTROL) != DesignState.initial(_TWO_CONTROL)  # design distinguishes
    assert len({a, b}) == 1                                          # hashable, dedupes
