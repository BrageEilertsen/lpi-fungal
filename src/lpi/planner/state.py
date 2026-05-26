"""Planner item 1: the design-resolution lattice with three-valued, probabilistic transitions.

A design proposal is a target program a few grammar-legal edits from a natural template (lpi.realizability).
The planner reasons not about the design as a whole but about its RESOLUTION STATE: which of its edits an
observation has CONFIRMED, which it has REFUTED, and which remain UNRESOLVED. The lattice is over these
states; an observation is an edge.

The edge is PROBABILISTIC, and that is the load-bearing design choice. An observation targeting an
unresolved edit does not deterministically resolve it -- it transitions to a confirmed-outcome state with
some probability and a refuted-outcome state with the complement, where the probability comes from a
forward model (item 2: P(outcome | design) from the mass / MS-MS forward models). Modelling the transition
as deterministic and bolting probabilities on later is the painful refactor we are avoiding; the API
returns both branches with probabilities from the start.

What consumes this:
  * item 2 (E[Δ|Z*|] estimator) supplies the forward model and evaluates |Z*| in the hypothetical
    next-states this lattice enumerates (the verifier is pure, so hypothetical evaluation is safe);
  * item 3 (objective) weights states by ``residual_realizability_cost`` (super-additive on the
    UNRESOLVED control edits -- so resolving one control edit drops the remaining penalty non-linearly);
  * item 4 (planner) chooses among ``available_transitions``;
  * item 5 (decision-tree output) is a presentation layer over these probabilistic edges -- each tree
    branch is an Outcome, carrying its probability.

Three-valued, not binary: REFUTED is distinct from UNRESOLVED. Negative evidence is real (the curated
verifications ruled candidates out as much as confirmed them); collapsing refuted into unresolved would
let the planner re-recommend an observation that has already failed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from lpi.realizability import Edit, Realizability, cost


class Resolution(Enum):
    UNRESOLVED = "unresolved"
    CONFIRMED = "confirmed"   # an observation gave evidence consistent with this edit having realized
    REFUTED = "refuted"       # an observation gave evidence INCONSISTENT with it (negative evidence)


@dataclass(frozen=True)
class Observation:
    """A modelled experiment the planner may choose. ``cost`` is the committed relative cost
    (mass=1, MS/MS=3, isotope=15, knockout=30; PLANNER_SKETCH). v0 models mass + MS/MS soundly;
    isotope/knockout are placeholders for the cost ratio (their forward models are item v1+)."""

    kind: str
    cost: float


OBSERVATION_COSTS: dict[str, float] = {"mass": 1.0, "msms": 3.0, "isotope": 15.0, "knockout": 30.0}


@dataclass(frozen=True, eq=False)
class DesignState:
    """A design proposal + the resolution status of each of its edits. Immutable; transitions return new
    states. ``status`` is parallel to ``realiz.edits`` (same order). Identity (eq/hash) is the design's
    target plus the status tuple -- ``realiz`` is carried for the estimator but is not part of identity."""

    realiz: Realizability
    status: tuple[Resolution, ...]

    def __post_init__(self) -> None:
        if len(self.status) != len(self.realiz.edits):
            raise ValueError("status must have exactly one entry per edit")

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, DesignState)
                and repr(self.realiz.target) == repr(other.realiz.target)
                and self.status == other.status)

    def __hash__(self) -> int:
        return hash((repr(self.realiz.target), self.status))

    @staticmethod
    def initial(realiz: Realizability) -> "DesignState":
        """All edits unresolved. For a natural product (no edits) this is already terminal."""
        return DesignState(realiz, tuple(Resolution.UNRESOLVED for _ in realiz.edits))

    @property
    def edits(self) -> tuple[Edit, ...]:
        return tuple(self.realiz.edits)

    def unresolved_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.status) if s is Resolution.UNRESOLVED]

    def unresolved_edits(self) -> list[Edit]:
        return [self.realiz.edits[i] for i in self.unresolved_indices()]

    @property
    def is_terminal(self) -> bool:
        """No unresolved edits remain -- the design's realizability is fully adjudicated."""
        return not self.unresolved_indices()

    def with_resolution(self, edit_index: int, res: Resolution) -> "DesignState":
        """Return a new state with one unresolved edit set to ``res`` (CONFIRMED or REFUTED)."""
        if res is Resolution.UNRESOLVED:
            raise ValueError("a transition resolves an edit to CONFIRMED or REFUTED, not UNRESOLVED")
        if self.status[edit_index] is not Resolution.UNRESOLVED:
            raise ValueError(f"edit {edit_index} is already resolved ({self.status[edit_index].value})")
        new = list(self.status)
        new[edit_index] = res
        return DesignState(self.realiz, tuple(new))

    def residual_realizability_cost(self) -> int:
        """Super-additive realizability cost over the UNRESOLVED edits only -- confirmed/refuted edits'
        cost is settled and drops out. This is the joint-resolution sensitivity: resolving one of two
        stacked control edits drops the remaining penalty from quadratic-in-2 to linear-in-1."""
        return cost(self.unresolved_edits())


@dataclass(frozen=True)
class Outcome:
    """One probabilistic branch of an observation: a label, the resulting state, and its probability.
    The decision-tree output (item 5) renders these directly."""

    label: str            # "confirmed" | "refuted"
    next_state: DesignState
    probability: float


#: A forward model: probability that ``obs`` applied to resolve edit ``edit_index`` of ``state`` returns a
#: CONFIRMED outcome. Item 2 supplies the real one (from |Z*| forward modelling over mass / MS-MS).
ForwardModel = Callable[["DesignState", Observation, int], float]


def uninformative_forward_model(state: DesignState, obs: Observation, edit_index: int) -> float:
    """Stub forward model: every observation is maximally uninformative (50/50). Lets item 1 and its
    tests exercise the lattice STRUCTURE before item 2's real estimator exists; also a sane default."""
    return 0.5


def transition(state: DesignState, obs: Observation, edit_index: int,
               forward_model: ForwardModel = uninformative_forward_model) -> list[Outcome]:
    """Probabilistic transition. Applying ``obs`` to resolve edit ``edit_index`` yields a confirmed-outcome
    state with probability ``p`` and a refuted-outcome state with ``1 - p``, ``p`` from ``forward_model``.
    Returns BOTH branches -- the estimator weights them, the planner chooses among observations, the
    decision tree renders them."""
    if edit_index not in state.unresolved_indices():
        raise ValueError(f"edit {edit_index} is not an unresolved edit of this state")
    p = float(forward_model(state, obs, edit_index))
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"forward-model probability out of range [0,1]: {p}")
    return [
        Outcome("confirmed", state.with_resolution(edit_index, Resolution.CONFIRMED), p),
        Outcome("refuted", state.with_resolution(edit_index, Resolution.REFUTED), 1.0 - p),
    ]


def available_transitions(state: DesignState,
                          observations: list[Observation]) -> list[tuple[Observation, int]]:
    """The lattice's out-edges from ``state``: every (observation, unresolved-edit) pair the planner may
    choose. Empty when the state is terminal. The planner (item 4) ranks these; here we enumerate them."""
    return [(o, i) for o in observations for i in state.unresolved_indices()]
