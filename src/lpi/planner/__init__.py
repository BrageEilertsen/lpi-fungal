"""The experiment-selection planner (item 1+).

A metabologenomics-TRIAGE planner with named limits: it ranks observations by expected information gain
per unit cost over the modelled observable set and names the clusters where its observables are
insufficient (which route to expression + NMR). It is not a structure-elucidation replacement -- the
curated verification corpus (data/curation/verification_sequences.csv) shows every verified product was
pinned by heterologous expression + NMR, with mass/MS-MS serving as triage.

Build order (see PLANNER_SKETCH.md): item 1 = the resolution-state lattice with three-valued,
probabilistic transitions (this package's ``state`` module); items 2-5 = estimator, objective, planner,
decision-tree output. Scored against the retrospective harness (scripts/validation_harness.py).
"""
from lpi.planner.plan import (
    TRIAGE_OBSERVATIONS,
    TriagePlan,
    TriageVerdict,
    next_edit_to_resolve,
    rank_designs,
    triage,
)
from lpi.planner.state import (
    OBSERVATION_COSTS,
    DesignState,
    Observation,
    Outcome,
    Resolution,
    available_transitions,
    transition,
    uninformative_forward_model,
)
from lpi.planner.tree import Branch, TreeNode, decision_tree, to_dict, to_json, to_mermaid

__all__ = [
    "Resolution",
    "Observation",
    "OBSERVATION_COSTS",
    "DesignState",
    "Outcome",
    "transition",
    "available_transitions",
    "uninformative_forward_model",
    "TriageVerdict",
    "TriagePlan",
    "TRIAGE_OBSERVATIONS",
    "triage",
    "next_edit_to_resolve",
    "rank_designs",
    "TreeNode",
    "Branch",
    "decision_tree",
    "to_dict",
    "to_json",
    "to_mermaid",
]
