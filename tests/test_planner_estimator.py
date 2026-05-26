"""Planner item 2: the E[Δ|Z*|] estimator grounded in the sound verifier."""
from __future__ import annotations

from lpi.chem.program import Cycle, Extender, Program, ReductionState as R, Release
from lpi.planner.estimator import (
    MASS,
    MSMS,
    delta_z,
    design_product,
    forward_model_confirm,
    make_forward_model,
    z_star,
)
from lpi.planner.state import DesignState, Observation, transition
from lpi.realizability import natural_manifold, realize

_MAN = natural_manifold()
_A = Release.ALDOL_AROMATIC

# in-grammar reduction design: 6-MSA + KR fired at cycle 3
_REDUCTION = realize(Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), _A), _MAN)
# out-of-grammar design: methylmalonyl extender swap (the malonyl-only grammar cannot produce it)
_EXTENDER = realize(
    Program("acetyl", (Cycle(R.KETO), Cycle(R.KR, extender=Extender.METHYLMALONYL), Cycle(R.KETO)), _A),
    _MAN)


def test_design_product_executes_target():
    p = design_product(_REDUCTION)
    assert p is not None and p.mass > 0 and p.smiles


def test_z_star_monotone_non_increasing_as_observables_added():
    prod = design_product(_REDUCTION)
    genome = z_star(_REDUCTION, prod, False, False)
    mass = z_star(_REDUCTION, prod, True, False)
    mass_msms = z_star(_REDUCTION, prod, True, True)
    assert genome >= mass >= mass_msms >= 1          # adding evidence never grows the consistent set
    assert genome > mass                              # accurate mass is informative here


def test_delta_z_is_the_collapse_from_adding_an_observable():
    prod = design_product(_REDUCTION)
    genome = z_star(_REDUCTION, prod, False, False)
    mass = z_star(_REDUCTION, prod, True, False)
    assert delta_z(_REDUCTION, prod, frozenset(), MASS) == genome - mass > 0
    # adding mass again resolves nothing
    assert delta_z(_REDUCTION, prod, frozenset({MASS}), MASS) == 0


def test_forward_model_confirms_in_grammar_refutes_out_of_grammar():
    # an in-grammar design's product is mass-consistent with >=1 legal program -> confirm
    assert forward_model_confirm(_REDUCTION, design_product(_REDUCTION)) == 1.0
    # the methylmalonyl extender swap executes to a structure but no malonyl-only program matches its
    # mass -> the observation would refute (out-of-grammar)
    assert forward_model_confirm(_EXTENDER, design_product(_EXTENDER)) == 0.0


def test_forward_model_plugs_into_the_probabilistic_transition():
    fm = make_forward_model()
    obs = Observation("mass", 1.0)
    # in-grammar design -> confirmed branch certain
    outs = transition(DesignState.initial(_REDUCTION), obs, 0, fm)
    conf = next(o for o in outs if o.label == "confirmed")
    assert conf.probability == 1.0
    # out-of-grammar design -> refuted branch certain (the three-valued state's REFUTED actually fires)
    oute = transition(DesignState.initial(_EXTENDER), obs, 0, fm)
    refu = next(o for o in oute if o.label == "refuted")
    assert refu.probability == 1.0
