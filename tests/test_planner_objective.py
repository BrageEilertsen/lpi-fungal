"""Planner item 3: the objective -- E[Δ|Z*|] * w_real / c_obs."""
from __future__ import annotations

import pytest

from lpi.chem.program import Cycle, Program, ReductionState as R, Release
from lpi.planner.estimator import MASS, MSMS, design_product, z_star
from lpi.planner.objective import observation_score, w_real
from lpi.planner.state import DesignState, Observation
from lpi.realizability import natural_manifold, realize

_MAN = natural_manifold()
_A = Release.ALDOL_AROMATIC

_ONE_CONTROL = realize(Program("acetyl", (Cycle(R.KETO), Cycle(R.KR), Cycle(R.KR)), _A), _MAN)
_TWO_CONTROL = realize(Program("acetyl", (Cycle(R.KR), Cycle(R.KR), Cycle(R.KR)), _A), _MAN)
# de-novo C-MeT insertion -> add_cmt SPECULATIVE -> verdict speculative
_SPECULATIVE = realize(Program("acetyl", (Cycle(R.KETO, c_methyl=True), Cycle(R.KR), Cycle(R.KETO)), _A),
                       _MAN)

_MASS = Observation("mass", 1.0)
_MSMS = Observation("msms", 3.0)


def test_w_real_is_inverse_residual_cost_and_higher_for_more_realizable():
    w1 = w_real(DesignState.initial(_ONE_CONTROL))     # residual = cost(1 control) = 2 -> 0.5
    w2 = w_real(DesignState.initial(_TWO_CONTROL))     # residual = cost(2 control) = 8 -> 0.125
    assert w1 == pytest.approx(0.5) and w2 == pytest.approx(0.125)
    assert w1 > w2                                     # the more-realizable design is weighted higher


def test_w_real_hard_excludes_speculative_designs():
    assert _SPECULATIVE.verdict == "speculative"
    assert w_real(DesignState.initial(_SPECULATIVE)) == 0.0


def test_observation_score_rewards_informative_cheap_observation():
    s = DesignState.initial(_ONE_CONTROL)
    prod = design_product(_ONE_CONTROL)
    gain = z_star(_ONE_CONTROL, prod, False, False) - z_star(_ONE_CONTROL, prod, True, False)
    # mass: gain * w_real(0.5) / cost(1)
    assert observation_score(s, frozenset(), _MASS, prod) == pytest.approx(gain * 0.5 / 1.0)
    assert observation_score(s, frozenset(), _MASS, prod) > 0
    # once mass is incorporated, MS/MS adds nothing here -> zero score (verified at +mass)
    assert observation_score(s, frozenset({MASS}), _MSMS, prod) == 0.0


def test_observation_score_zero_for_speculative_design():
    s = DesignState.initial(_SPECULATIVE)
    prod = design_product(_SPECULATIVE)
    assert observation_score(s, frozenset(), _MASS, prod) == 0.0
