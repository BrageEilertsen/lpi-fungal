"""Retrospective planner evaluation (planner item 5a): the triage planner against curated ground truth.

Two halves of the objective, two checks (PLANNER_SKETCH; the harness ground truth is
data/curation/verification_sequences.csv, curated from primary sources):

  A. TRIAGE ROUTING -- on each verified product, does the planner route triage-sufficient vs
     elucidation-required correctly? Scored WITHIN the planner's frame (mass/MS-MS), with the historical
     expression+NMR path shown beside it as the honest scope caveat -- NOT as the planner's target. (The
     curation finding: all 7 were pinned by expression + NMR; the planner is a triage tool, so the right
     question is which MS observable collapses |Z*|, and where the modelled set runs out.)

  B. REALIZABILITY ALLOCATION -- on the 6-MSA design neighbourhood, does the planner steer attention to
     engineerable/frontier designs over speculative ones (w_real ranking)?

This is the retrospective measure Brage asked for: the planner judged against a corpus whose ground
truth is curated from primary literature, not assumed.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from rdkit import RDLogger

from lpi.planner.plan import TriageVerdict, rank_designs, triage
from lpi.realizability import natural_manifold, one_edit_neighbours, realize

RDLogger.DisableLog("rdApp.*")
_CSV = Path(__file__).resolve().parents[1] / "data" / "curation" / "verification_sequences.csv"


def _manifold_by_bgc() -> dict:
    return {t.id.split()[0]: t for t in natural_manifold()}


def main() -> None:
    man = natural_manifold()
    by_bgc = _manifold_by_bgc()
    with _CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    print("A. TRIAGE ROUTING -- planner verdict vs curated historical path (within-frame; history = caveat)\n")
    print(f"{'product':30s} {'planner triage':22s} {'|Z*| ladder':22s} historical (literature)")
    print("-" * 112)
    routed = Counter()
    for r in rows:
        tmpl = next((by_bgc[b] for b in r["clusters"].split(";") if b in by_bgc), None)
        if tmpl is None:
            print(f"{r['product'][:30]:30s} {'(core not executable)':22s} {'-':22s} {r['historical_sequence'][:36]}")
            continue
        plan = triage(realize(tmpl.program, man))
        ladder = "->".join(str(z) for _, z in plan.z_ladder)
        routed[plan.verdict] += 1
        print(f"{r['product'][:30]:30s} {plan.verdict.value:22s} {ladder:22s} {r['historical_sequence'][:36]}")
    print("\n  routing: " + ", ".join(f"{v.value}={n}" for v, n in routed.items()))
    print("  reading: small aromatics collapse at +mass (triage decides); the C12 HR core stays")
    print("  under-determined through mass+MS/MS -> elucidation-required (the planner names its own limit).")

    print("\nB. REALIZABILITY ALLOCATION -- 6-MSA 1-edit neighbourhood, ranked by w_real\n")
    msa = next(t for t in man if "6-MSA" in t.id)
    designs = [realize(p, man) for p in one_edit_neighbours(msa)]
    ranked = rank_designs(designs)
    by_verdict: dict[str, list[float]] = {}
    for rz, w in ranked:
        by_verdict.setdefault(rz.verdict, []).append(w)
    for verdict in ("engineerable", "frontier", "speculative"):
        ws = by_verdict.get(verdict, [])
        if ws:
            print(f"  {verdict:13s} n={len(ws):3d}  w_real in [{min(ws):.3f}, {max(ws):.3f}]")
    spec_w = by_verdict.get("speculative", [0])
    eng_fro = by_verdict.get("engineerable", []) + by_verdict.get("frontier", [])
    print(f"\n  speculative designs all weight {max(spec_w):.1f} (hard-excluded); "
          f"engineerable/frontier weight > 0 -> attention steers to them.")
    assert all(w == 0.0 for w in spec_w) and all(w > 0 for w in eng_fro), "allocation invariant violated"
    print("  allocation invariant holds: speculative == 0, engineerable/frontier > 0.")


if __name__ == "__main__":
    main()
