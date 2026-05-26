"""Planner item 4: the triage planner over the resolution lattice."""
from __future__ import annotations

from lpi.chem.program import Cycle, Extender, Program, ReductionState as R, Release
from lpi.planner.plan import (
    TriageVerdict,
    next_edit_to_resolve,
    rank_designs,
    triage,
)
from lpi.realizability import Axis, natural_manifold, realize

_MAN = natural_manifold()
_A = Release.ALDOL_AROMATIC

_REDUCTION = realize(Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), _A), _MAN)
_EXTENDER = realize(
    Program("acetyl", (Cycle(R.KETO), Cycle(R.KR, extender=Extender.METHYLMALONYL), Cycle(R.KETO)), _A),
    _MAN)
_SPECULATIVE = realize(Program("acetyl", (Cycle(R.KETO, c_methyl=True), Cycle(R.KR), Cycle(R.KETO)), _A),
                       _MAN)
_STRUCT_CONTROL = realize(Program("acetyl", (Cycle(R.KETO), Cycle(R.ER), Cycle(R.KETO)), _A), _MAN)
_TWO_CONTROL = realize(Program("acetyl", (Cycle(R.KR), Cycle(R.KR), Cycle(R.KR)), _A), _MAN)
_STARTER = realize(Program("hexanoyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), _A), _MAN)
_BAB = realize(next(t for t in _MAN if "BAB" in t.id).program, _MAN)


def test_triage_small_aromatic_verified_by_mass():
    plan = triage(_REDUCTION)
    assert plan.verdict is TriageVerdict.VERIFIED_BY_TRIAGE
    assert plan.resolving_observable == "mass" and plan.sequence == ["mass"]


def test_triage_names_elucidation_required_for_hr_core():
    # BAB (C12 highly-reducing core) is mass-degenerate; modelled observables don't collapse it ->
    # the planner routes it to expression + NMR. Knowing its own limits IS the contribution.
    plan = triage(_BAB)
    assert plan.verdict is TriageVerdict.ELUCIDATION_REQUIRED
    assert "mass" in plan.sequence and "msms" in plan.sequence       # it tried the modelled set first


def test_triage_out_of_grammar_design_refuted_by_mass():
    plan = triage(_EXTENDER)
    assert plan.verdict is TriageVerdict.OUT_OF_GRAMMAR


def test_triage_speculative_design_not_pursued():
    assert _SPECULATIVE.verdict == "speculative"
    plan = triage(_SPECULATIVE)
    assert plan.verdict is TriageVerdict.SPECULATIVE and plan.sequence == []


def test_next_edit_resolves_the_control_edit_first():
    from lpi.planner.state import DesignState
    # +ER (structural, tier 1) + fire it at cyc2 (control, tier 2): resolving the control edit drops the
    # residual more, so it is resolved first.
    s = DesignState.initial(_STRUCT_CONTROL)
    assert s.realiz.structural_cost == 1 and s.realiz.control_cost == 1
    i = next_edit_to_resolve(s)
    assert s.realiz.edits[i].axis is Axis.CONTROL
    # and in a two-control stacked design the chosen edit is (necessarily) a control edit
    s2 = DesignState.initial(_TWO_CONTROL)
    assert _TWO_CONTROL.edits[next_edit_to_resolve(s2)].axis is Axis.CONTROL
    # terminal -> nothing to resolve
    term = s.with_resolution(0, __import__("lpi.planner.state", fromlist=["Resolution"]).Resolution.CONFIRMED)
    term = term.with_resolution(1, __import__("lpi.planner.state", fromlist=["Resolution"]).Resolution.CONFIRMED)
    assert next_edit_to_resolve(term) is None


def test_rank_designs_sinks_speculative_below_engineerable_and_frontier():
    ranked = rank_designs([_SPECULATIVE, _REDUCTION, _STARTER])
    weights = [w for _, w in ranked]
    assert weights == sorted(weights, reverse=True)        # descending by realizability weight
    assert ranked[-1][0] is _SPECULATIVE and ranked[-1][1] == 0.0   # speculative last, weight 0
    assert ranked[0][0] is _STARTER                        # engineerable (cheap structural) ranked first
