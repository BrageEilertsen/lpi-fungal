"""Realizability metric: dual-axis (structural / control) edit distance + the design-space finding."""
from __future__ import annotations

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.realizability import (Axis, Edit, Tier, _target_domains, cost, design_space, edits_from,
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
    assert any(e.kind == "add_reductive" and e.axis is Axis.STRUCTURAL for e in r.edits)
    assert r.structural_cost == 1 and r.control_cost == 1 and r.verdict == "frontier"


def test_iteration_count_change_is_control():
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO), Cycle(R.KETO)),
                     Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert any(e.kind in ("cycle_add", "cycle_remove") and e.axis is Axis.CONTROL
               for e in r.control_edits)
    assert r.verdict == "frontier"


def test_release_reprogramming_is_structural_and_engineerable():
    # release reprogram is enzymatically a TE/cyclase domain swap -> structural, PLAUSIBLE, not the
    # iteration frontier (curation Call 3: demoted from SPECULATIVE so the cost table stops ranking a
    # domain swap as harder than reprogramming the iteration).
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KETO)), Release.LACTONIZATION)
    r = realize(target, _MAN)
    rel = [e for e in r.edits if e.kind == "release"]
    assert rel and rel[0].tier is Tier.PLAUSIBLE and rel[0].axis is Axis.STRUCTURAL
    assert r.control_cost == 0 and r.verdict == "engineerable"


def test_add_cmt_is_speculative_de_novo():
    # de-novo C-MeT insertion into a methylation-free synthase (6-MSA has no CMT): no precedent
    # (Cox: methylation precedent is reprogramming a PRESENT C-MeT, never inserting one).
    target = Program("acetyl", (Cycle(R.KETO, c_methyl=True), Cycle(R.KR), Cycle(R.KETO)),
                     Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    assert r.nearest_id == "BGC0001275 6-MSA"
    cmt = [e for e in r.edits if e.kind == "add_cmt"]
    assert cmt and cmt[0].tier is Tier.SPECULATIVE and cmt[0].axis is Axis.STRUCTURAL
    assert r.verdict == "speculative"


def test_cycle_remove_is_speculative():
    # removing an iteration changes the program's termination count -- deep iteration-program editing,
    # the least-precedented control edit (curation Call 4): SPECULATIVE, unlike cycle_add (PLAUSIBLE).
    target = Program("acetyl", (Cycle(R.KETO), Cycle(R.KR)), Release.ALDOL_AROMATIC)
    r = realize(target, _MAN)
    rm = [e for e in r.edits if e.kind == "cycle_remove"]
    assert rm and rm[0].tier is Tier.SPECULATIVE and rm[0].axis is Axis.CONTROL
    assert r.verdict == "speculative"


def test_control_cost_is_super_additive():
    # curation Call 2: control edits stack super-additively (~k^2) because the iteration programme is
    # emergent/non-separable (Cox 2023); structural edits stay linear.
    one = [Edit("reduction", "a", Tier.PLAUSIBLE, Axis.CONTROL)]
    two = [Edit("reduction", "a", Tier.PLAUSIBLE, Axis.CONTROL),
           Edit("reduction", "b", Tier.PLAUSIBLE, Axis.CONTROL)]
    assert cost(one) == int(Tier.PLAUSIBLE)          # reduces to linear at k=1
    assert cost(two) > 2 * cost(one)                 # super-additive at k=2 (4*2=8 > 2*2)
    structural = [Edit("starter", "x", Tier.DOCUMENTED, Axis.STRUCTURAL),
                  Edit("extender", "y", Tier.DOCUMENTED, Axis.STRUCTURAL)]
    assert cost(structural) == 2 * int(Tier.DOCUMENTED)   # structural stays linear


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
