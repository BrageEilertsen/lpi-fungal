"""Planner item 5b: the decision-tree output -- serialize the lattice walk, don't re-derive it.

The triage planner (item 4) walks the modelled observables; this renders that walk as a decision TREE
whose edges are observation OUTCOMES. The load-bearing design choice (and the thing most tree
visualizations get wrong): **both children of every decision node are first-class.** The refuted-outcome
branch carries as much information as the confirmed one -- a refuted observation says the design's product
is not what the grammar predicted (out-of-grammar) -- so it is rendered, never collapsed into an "if not,
then..." footnote. BAB's elucidation-required verdict only reads correctly when the tree shows the
modelled observables being exhausted along the confirmed path with the refuted leaves drawn beside them.

Outputs: a ``TreeNode`` object, ``to_dict`` (JSON artifact), and ``to_mermaid`` (paper figure source).
v0 probabilities are design-level and near-deterministic (feasible -> confirm p=1, out-of-grammar ->
refute p=1; see estimator); the branch STRUCTURE is v1-ready for genuinely uncertain build outcomes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from lpi.engine import NEAR_UNIQUE_MAX
from lpi.grammars import PKS, Grammar
from lpi.planner.estimator import DesignProduct, delta_z, design_product, forward_model_confirm, z_star
from lpi.planner.objective import w_real
from lpi.planner.plan import TRIAGE_OBSERVATIONS, TriageVerdict
from lpi.planner.state import DesignState, Observation
from lpi.realizability import Realizability


@dataclass
class Branch:
    outcome: str            # "confirmed" | "refuted"
    probability: float
    child: "TreeNode"


@dataclass
class TreeNode:
    observables: frozenset[str]          # observables incorporated to reach this node
    z_star: int                          # |Z*| here
    verdict: TriageVerdict | None        # set iff terminal
    recommend: str | None                # next observation kind iff internal
    branches: list[Branch] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.verdict is not None


def _terminal(observables, z, verdict) -> TreeNode:
    return TreeNode(observables, z, verdict, None, [])


def decision_tree(realiz: Realizability, prod: DesignProduct | None = None,
                  observations=TRIAGE_OBSERVATIONS, grammar: Grammar = PKS,
                  threshold: int = NEAR_UNIQUE_MAX) -> TreeNode:
    """Build the planner's decision tree for a design. Every internal node has exactly two first-class
    branches (confirmed, refuted); the refuted child is the out-of-grammar leaf (the negative case)."""
    if w_real(DesignState.initial(realiz)) == 0.0:
        return _terminal(frozenset(), 0, TriageVerdict.SPECULATIVE)
    if prod is None:
        prod = design_product(realiz)
    if prod is None:
        return _terminal(frozenset(), 0, TriageVerdict.OUT_OF_GRAMMAR)
    p_confirm = forward_model_confirm(realiz, prod, grammar)
    return _build(realiz, prod, frozenset(), list(observations), grammar, threshold, p_confirm)


def _build(realiz, prod, current, remaining, grammar, threshold, p_confirm) -> TreeNode:
    z = z_star(realiz, prod, "mass" in current, "msms" in current, grammar)
    if z == 0:
        return _terminal(current, 0, TriageVerdict.OUT_OF_GRAMMAR)
    if z <= threshold:
        return _terminal(current, z, TriageVerdict.VERIFIED_BY_TRIAGE)
    if not remaining:
        return _terminal(current, z, TriageVerdict.ELUCIDATION_REQUIRED)
    obs = max(remaining, key=lambda o: delta_z(realiz, prod, current, o.kind, grammar) / o.cost)
    rest = [o for o in remaining if o.kind != obs.kind]
    confirmed = _build(realiz, prod, current | {obs.kind}, rest, grammar, threshold, p_confirm)
    # refuted: the observation's outcome does not match the predicted product -> out-of-grammar. Drawn as
    # a first-class sibling, not a footnote -- the negative-information case the planner must represent.
    refuted = _terminal(current, z, TriageVerdict.OUT_OF_GRAMMAR)
    return TreeNode(current, z, None, obs.kind,
                    [Branch("confirmed", p_confirm, confirmed),
                     Branch("refuted", 1.0 - p_confirm, refuted)])


# ---- serialization -------------------------------------------------------------------
def to_dict(node: TreeNode) -> dict:
    """JSON-able nested dict -- the machine artifact."""
    d = {
        "observables": sorted(node.observables),
        "z_star": node.z_star,
        "verdict": node.verdict.value if node.verdict else None,
        "recommend": node.recommend,
        "branches": [{"outcome": b.outcome, "probability": b.probability, "child": to_dict(b.child)}
                     for b in node.branches],
    }
    return d


def to_json(node: TreeNode, indent: int = 2) -> str:
    return json.dumps(to_dict(node), indent=indent)


_SHAPE = {  # terminal verdict -> (open, close) Mermaid node delimiters
    TriageVerdict.VERIFIED_BY_TRIAGE: ("([", "])"),     # stadium
    TriageVerdict.ELUCIDATION_REQUIRED: ("[[", "]]"),   # subroutine box -- the "route to expression+NMR"
    TriageVerdict.OUT_OF_GRAMMAR: ("{{", "}}"),         # hexagon
    TriageVerdict.SPECULATIVE: ("{{", "}}"),
}


def to_mermaid(node: TreeNode, title: str = "") -> str:
    """Mermaid ``graph TD`` -- the paper-figure source. Decision nodes are diamonds; terminal verdicts get
    distinct shapes. Both outcome edges of every decision are emitted (refuted is never dropped)."""
    lines = ["graph TD"]
    counter = [0]

    def label(n: TreeNode) -> str:
        obs = "+".join(sorted(n.observables)) or "genome"
        if n.is_terminal:
            return f"|Z*|={n.z_star} @ {obs}<br/>{n.verdict.value}"
        return f"|Z*|={n.z_star} @ {obs}<br/>run {n.recommend}?"

    def emit(n: TreeNode) -> str:
        nid = f"n{counter[0]}"
        counter[0] += 1
        text = label(n).replace('"', "'")
        if n.is_terminal:
            o, c = _SHAPE[n.verdict]
            lines.append(f'  {nid}{o}"{text}"{c}')
        else:
            lines.append(f'  {nid}{{"{text}"}}')          # decision diamond
        for b in n.branches:
            child_id = emit(b.child)
            lines.append(f'  {nid} -->|"{b.outcome} p={b.probability:.2f}"| {child_id}')
        return nid

    emit(node)
    if title:
        lines.insert(1, f"  %% {title}")
    return "\n".join(lines)
