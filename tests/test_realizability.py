"""Realizability metric: dual-axis (structural / control) edit distance + the design-space finding."""
from __future__ import annotations

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.realizability import (Axis, Tier, _target_domains, design_space, edits_from,
                               natural_manifold, realize)

_MAN = natural_manifold()
_BY = {t.id: t for t in _MAN}
_MSA = _BY["BGC0001275 6-MSA"]      # acetyl K,KR,K -> aldol; PR domains {KS,AT,DH,KR,ACP}


def test_natural_program_is_distance_zero():
    r = realize(_MSA.program, _MAN)
    assert r.verdict == "natural" and r.structural_cost == 0 and r.control_cost == 0


def test_reprogram_within_capability_is_control_only():
    # 6-MSA carries a KR domain; firing KR on cycle 3 as well is a pure control (iteration) edit
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert r.nearest_id == "BGC0001275 6-MSA"
    assert r.structural_cost == 0 and r.control_cost == 1
    assert r.control_edits[0].kind == "reduction" and r.verdict == "frontier"


def test_new_domain_is_structural_plus_control_to_fire():
    # ER is not in 6-MSA's domain set: add the ER domain (structural) AND fire it on cyc2 (control)
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.ER), Cycle(R.KETO)), Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert any(e.kind == "add_domain" and e.axis is Axis.STRUCTURAL for e in r.edits)
    assert r.structural_cost == 1 and r.control_cost == 1 and r.verdict == "frontier"


def test_iteration_count_change_is_control():
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO), Cycle(R.KETO)),
                     Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert any(e.kind in ("cycle_add", "cycle_remove") and e.axis is Axis.CONTROL
               for e in r.control_edits)
    assert r.verdict == "frontier"


def test_release_reprogramming_is_speculative():
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), Release.LACTONIZATION)
    r = realize(target, _MAN)
    assert any(e.kind == "release" and e.tier is Tier.SPECULATIVE for e in r.edits)
    assert r.verdict == "speculative"


def test_target_domains_inference():
    p = Program("acetyl", (Cycle(R.KETO), Cycle(R.ER)), Release.HYDROLYSIS)   # ER needs KR,DH,ER
    assert _target_domains(p) == frozenset({"KR", "DH", "ER"})


def test_design_space_control_axis_dominates_for_fungi():
    ds = design_space(_MAN)
    assert ds["n_designs"] > 0
    # the finding: for fungal iterative PKS the control axis dominates the 1-edit design space
    assert len(ds["frontier"]) > len(ds["engineerable"])
    assert all(r.control_cost == 0 for r in ds["engineerable"])   # engineerable = structural-only
    assert all(r.control_cost >= 1 for r in ds["frontier"])       # frontier = involves a control edit
