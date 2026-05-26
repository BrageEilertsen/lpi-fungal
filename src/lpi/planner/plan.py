"""Planner item 4: the triage planner over the resolution lattice.

Two coupled jobs, both built from items 1-3:

(A) TRIAGE -- observable-driven, the harness-scored core. For a design, walk the modelled observable set
    ({} -> +mass -> +mass+MS/MS), greedily adding the most informative-per-cost observable, and classify:
      * VERIFIED_BY_TRIAGE   -- |Z*| collapsed to near-unique within the modelled observables;
      * ELUCIDATION_REQUIRED -- observables exhausted, |Z*| still large -> route to expression + NMR;
      * OUT_OF_GRAMMAR       -- a mass observation refutes every candidate (no legal program matches);
      * SPECULATIVE          -- design hard-excluded (w_real 0); do not spend observations.
    Naming ELUCIDATION_REQUIRED is the contribution: a triage tool that knows its own limits (the BAB
    case). This generalises observe.minimum_sufficient_observables with the triage-vs-elucidation verdict.

(B) ALLOCATION -- edit-driven, the design-neighbourhood validation. ``rank_designs`` orders designs by the
    realizability weight (engineerable/frontier above speculative); ``next_edit_to_resolve`` picks the edit
    whose resolution most reduces the residual super-additive cost -- the control edit in a stacked design
    -- the emergent joint-resolution prioritisation the lattice exists to express.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lpi.engine import NEAR_UNIQUE_MAX
from lpi.grammars import PKS, Grammar
from lpi.planner.estimator import DesignProduct, delta_z, design_product, z_star
from lpi.planner.objective import w_real
from lpi.planner.state import OBSERVATION_COSTS, DesignState, Observation, Resolution
from lpi.realizability import Realizability

# the modelled metabologenomics-triage observables (mass, then MS/MS); isotope/knockout are v1+
TRIAGE_OBSERVATIONS = (Observation("mass", OBSERVATION_COSTS["mass"]),
                       Observation("msms", OBSERVATION_COSTS["msms"]))


class TriageVerdict(Enum):
    VERIFIED_BY_TRIAGE = "verified-by-triage"
    ELUCIDATION_REQUIRED = "elucidation-required"
    OUT_OF_GRAMMAR = "out-of-grammar"
    SPECULATIVE = "speculative"


@dataclass
class TriagePlan:
    verdict: TriageVerdict
    sequence: list[str]                       # observables incorporated, in order
    resolving_observable: str | None          # the observable that reached near-unique, or None
    z_ladder: list[tuple[frozenset, int]]     # |Z*| at each rung (incorporated-set, count)


def triage(realiz: Realizability, prod: DesignProduct | None = None,
           observations=TRIAGE_OBSERVATIONS, grammar: Grammar = PKS,
           threshold: int = NEAR_UNIQUE_MAX) -> TriagePlan:
    """Walk the modelled observables greedily (most |Z*|-collapse per cost first) and classify the design
    into the triage-vs-elucidation verdict. The harness scores the planner against this."""
    if w_real(DesignState.initial(realiz)) == 0.0:
        return TriagePlan(TriageVerdict.SPECULATIVE, [], None, [])
    if prod is None:
        prod = design_product(realiz)
    if prod is None:
        return TriagePlan(TriageVerdict.OUT_OF_GRAMMAR, [], None, [])

    current: frozenset[str] = frozenset()
    z = z_star(realiz, prod, False, False, grammar)
    ladder: list[tuple[frozenset, int]] = [(current, z)]
    if z == 0:
        return TriagePlan(TriageVerdict.OUT_OF_GRAMMAR, [], None, ladder)
    if z <= threshold:
        return TriagePlan(TriageVerdict.VERIFIED_BY_TRIAGE, [], "genome", ladder)

    sequence: list[str] = []
    remaining = list(observations)
    while remaining:
        # greedily pick the most informative-per-cost remaining observable
        scored = [(delta_z(realiz, prod, current, o.kind, grammar) / o.cost, o) for o in remaining]
        gain_per_cost, obs = max(scored, key=lambda t: t[0])
        current = current | {obs.kind}
        sequence.append(obs.kind)
        remaining = [o for o in remaining if o.kind != obs.kind]
        z = z_star(realiz, prod, "mass" in current, "msms" in current, grammar)
        ladder.append((current, z))
        if z == 0:
            return TriagePlan(TriageVerdict.OUT_OF_GRAMMAR, sequence, None, ladder)
        if z <= threshold:
            return TriagePlan(TriageVerdict.VERIFIED_BY_TRIAGE, sequence, obs.kind, ladder)
    # modelled observables exhausted and still under-determined -> structure elucidation required
    return TriagePlan(TriageVerdict.ELUCIDATION_REQUIRED, sequence, None, ladder)


def next_edit_to_resolve(state: DesignState) -> int | None:
    """Among the unresolved edits, the one whose resolution most reduces the residual super-additive cost
    -- i.e. unlocks the most realizability. In a stacked design this is a control edit (its resolution
    drops the residual quadratic->linear), so the planner resolves control edits first. None if terminal."""
    unresolved = state.unresolved_indices()
    if not unresolved:
        return None
    base = state.residual_realizability_cost()

    def drop(i: int) -> int:
        return base - state.with_resolution(i, Resolution.CONFIRMED).residual_realizability_cost()

    return max(unresolved, key=drop)


def rank_designs(realizs: list[Realizability]) -> list[tuple[Realizability, float]]:
    """Order designs by realizability weight (descending): engineerable/frontier above speculative
    (weight 0). The allocation half of the objective -- which designs are worth spending observations on."""
    scored = [(r, w_real(DesignState.initial(r))) for r in realizs]
    return sorted(scored, key=lambda t: -t[1])
