"""Planner item 3: the objective -- expected information gain per unit cost, weighted by realizability.

From a state, the planner scores a candidate observation by

    score = E[Δ|Z*|] * w_real(state) / c_obs(observation)

  * E[Δ|Z*|]  -- the |Z*| collapse the observation produces (item 2, ``delta_z``); the inference value.
  * w_real    -- the realizability weight: 1 / residual super-additive cost over the UNRESOLVED edits,
                 with a HARD THRESHOLD excluding ``speculative`` designs entirely (they must not dilute
                 attention from engineerable/frontier work). On the design side -- it is how much we care
                 to resolve this design at all, not something the observation reduces.
  * c_obs     -- the committed relative observation cost (mass=1, MS/MS=3, isotope=15, knockout=30).

Two cost terms on different sides (PLANNER_SKETCH Q3): observation cost in the denominator, super-additive
realizability as the design-side weight. The single-step score uses ``w_real`` at the CURRENT state; the
joint-resolution sensitivity -- resolving one of two stacked control edits drops the residual penalty
non-linearly and so raises w_real for the *next* step -- emerges in the lattice search (item 4), which
sees the residual-cost trajectory. Here we provide the single-step primitives that search consumes.
"""
from __future__ import annotations

from lpi.grammars import PKS, Grammar
from lpi.planner.estimator import DesignProduct, delta_z
from lpi.planner.state import DesignState, Observation


def w_real(state: DesignState) -> float:
    """Realizability weight of a state: 0 for a ``speculative`` design (hard-excluded), else
    1 / residual super-additive cost over the unresolved edits (linear in 1/c_real). A fully-resolved
    state has residual 0 and weight 1.0 (no observations remain to score there anyway)."""
    if state.realiz.verdict == "speculative":
        return 0.0
    residual = state.residual_realizability_cost()
    return 1.0 / residual if residual > 0 else 1.0


def observation_score(state: DesignState, current_observables: frozenset[str], observation: Observation,
                      prod: DesignProduct, grammar: Grammar = PKS) -> float:
    """E[Δ|Z*|] * w_real(state) / c_obs. Zero for a speculative design (w_real 0) or an uninformative
    observation (Δ|Z*| 0). ``current_observables`` is the subset of {"mass","msms"} already incorporated
    on the path to this state."""
    weight = w_real(state)
    if weight == 0.0:
        return 0.0
    gain = delta_z(state.realiz, prod, current_observables, observation.kind, grammar)
    return gain * weight / observation.cost
