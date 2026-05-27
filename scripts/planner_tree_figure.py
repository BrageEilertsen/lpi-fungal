"""Planner item 5b figure: the two contrasting decision trees that land the triage scoping visually.

6-MSA (verified-by-triage at +mass -- a shallow tree: the planner doing its job) beside BAB
(elucidation-required -- the tree exhausts the modelled observables without collapsing: the planner
naming its own limit). Side by side, these two trees are the paper figure.

Writes Mermaid (.mmd, the figure source) and JSON (the machine artifact) for each to results/.
"""
from __future__ import annotations

from pathlib import Path

from rdkit import RDLogger

from lpi.planner.tree import decision_tree, to_json, to_mermaid
from lpi.realizability import natural_manifold, realize

RDLogger.DisableLog("rdApp.*")
_RESULTS = Path(__file__).resolve().parents[1] / "results"

_CASES = [("6msa", "6-MSA", "6-MSA"), ("bab", "BAB", "BAB")]


def main() -> None:
    man = natural_manifold()
    for slug, needle, title in _CASES:
        tmpl = next(t for t in man if needle in t.id)
        tree = decision_tree(realize(tmpl.program, man))
        mmd = to_mermaid(tree, title=f"{title} -- experiment-selection planner")
        (_RESULTS / f"planner_tree_{slug}.mmd").write_text(mmd + "\n")
        (_RESULTS / f"planner_tree_{slug}.json").write_text(to_json(tree) + "\n")
        print(f"=== {title} ===")
        print(mmd)
        print(f"   -> results/planner_tree_{slug}.mmd, results/planner_tree_{slug}.json\n")
    print("The two trees side by side are the figure: 6-MSA shallow (verified-by-triage at +mass);")
    print("BAB exhausts mass+MS/MS to elucidation-required. Refuted branches drawn first-class on both.")


if __name__ == "__main__":
    main()
