"""Retrospective validation harness (planner item 0) -- built BEFORE the planner so there is ground
truth from day one.

Two notions of "verification sequence", and the gap between them is the finding:

* HISTORICAL (data/curation/verification_sequences.csv): how each structure was ACTUALLY pinned in the
  literature -- curated from primary sources. For every curated product this used heterologous
  EXPRESSION + NMR (and isotope feeding / in-vitro assay), NOT accurate mass / MS-MS.
* WITHIN-FRAME (computed here): given the genome alphabet + the MS observables a metabologenomics
  workflow actually has, which observable first collapses |Z*| to the verified program -- the rung the
  engine ladder reaches VERIFIED at. THIS is the planner's ground-truth target, because the planner is a
  metabologenomics-TRIAGE tool, not a replacement for definitive structure elucidation.

So the harness does not score a (future) planner against the historical NMR/expression path -- that is a
different question. It scores it against the within-frame disambiguation point, and reports the
historical path beside it to keep the scope honest. Today it runs the fixed-ladder baseline
(`minimum_sufficient_observables`); when the planner lands it plugs in here unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.engine import State
from lpi.executor import core
from lpi.grammars import PKS
from lpi.observe import MSObservables, exact_mass, minimum_sufficient_observables
from lpi.realizability import natural_manifold

RDLogger.DisableLog("rdApp.*")
_CSV = Path(__file__).resolve().parents[1] / "data" / "curation" / "verification_sequences.csv"


def _manifold_by_bgc() -> dict:
    """Map each BGC id in the realizability manifold to its (template) -- the clusters we can execute."""
    out = {}
    for t in natural_manifold():
        bgc = t.id.split()[0]          # "BGC0001275 6-MSA" -> "BGC0001275"
        out[bgc] = t
    return out


def _within_frame_target(template) -> tuple[str, dict]:
    """The rung at which the engine ladder (genome -> +mass -> +msms) reaches VERIFIED for this
    cluster's known product = the cheapest modelled observable that collapses |Z*|. The planner's
    ground-truth target."""
    prog = template.program
    smi = M.canonical_smiles(core.exec(prog))
    mass = exact_mass(smi)
    domains = set(template.domains) | {"KS", "AT", "ACP"}
    alpha = PKS.alphabet_from_domains(domains, starters=(prog.starter,))
    n = len(prog.cycles)
    plan = minimum_sufficient_observables(
        alpha, MSObservables(neutral_mass=mass), PKS, max(1, n - 1), n + 1)
    counts = {k: f"{st.value}:{c}" for k, (st, c) in plan.rungs.items()}
    return (plan.verified_at or "not-verified-by-modelled-observables"), counts


def main() -> None:
    with _CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    man = _manifold_by_bgc()

    print("Retrospective validation harness -- ground truth for the experiment-selection planner\n")
    print(f"{'product':30s} {'within-frame target':22s} {'historical (literature) path'}")
    print("-" * 110)
    curated = todo = computable = 0
    for r in rows:
        bgcs = r["clusters"].split(";")
        tmpl = next((man[b] for b in bgcs if b in man), None)
        if tmpl is not None:
            target, _counts = _within_frame_target(tmpl)
            computable += 1
        else:
            target = "core not in executable manifold"
        hist = r["historical_sequence"]
        if r["sourcing_status"] == "curated":
            curated += 1
        else:
            todo += 1
            hist = "[TODO] " + hist
        print(f"{r['product'][:30]:30s} {target:22s} {hist[:54]}")

    print(f"\n{curated}/{len(rows)} products curated from primary sources, {todo}/{len(rows)} TODO; "
          f"{computable}/{len(rows)} have an executable core (within-frame target computed).")
    print("\nFINDING (the curation surfaced it): every curated product was pinned historically by")
    print("heterologous EXPRESSION + NMR (+ isotope / in-vitro), NOT by accurate mass / MS-MS. The planner")
    print("models the metabologenomics-triage observables (mass, MS/MS), so its ground truth is the")
    print("WITHIN-FRAME disambiguation point above -- it is a triage tool, not a replacement for definitive")
    print("structure elucidation. The harness scores the planner against that point; the historical path is")
    print("recorded beside it to keep the claim honestly scoped (NOT as the planner's target).")


if __name__ == "__main__":
    main()
