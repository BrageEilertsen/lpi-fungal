"""The bounded literature-curation worklist: every edit kind that actually appears in the design space.

The realizability layer assigns each design a per-edit feasibility *tier* (DOCUMENTED / PLAUSIBLE /
SPECULATIVE). Those tiers are currently CLASS-level guesses hard-coded in ``realizability.edits_from``
-- the "vibes-based costs". The defensibility moat is replacing them with real, citable engineering
precedent. The scope of that reading is NOT "all of combinatorial-biosynthesis"; it is exactly the
edit kinds the 1-edit design enumeration produces. This script enumerates them, so the worklist is
bounded by the machine's own output rather than by a guess at the literature's size.

Outputs:
  * results/curation_worklist.csv  -- reproducible snapshot: (axis, kind) x current tier x how much
    design space rides on the cost (n_designs touched) x concrete instances seen.
  * data/curation/edit_tiers.csv   -- the curation WORKING TABLE, seeded once with the class-level
    tiers and empty precedent columns for the expert to fill. Not overwritten if it already exists
    (it is hand-edited over the reading); re-run only refreshes the results/ snapshot.

No precedent or citation is invented here -- the precedent columns are deliberately empty.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from lpi.realizability import design_space, natural_manifold, one_edit_neighbours, realize

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT = _ROOT / "results" / "curation_worklist.csv"
_TABLE = _ROOT / "data" / "curation" / "edit_tiers.csv"

# axis -> how its cost transfers, for the human reading the worklist.
_AXIS_NOTE = {
    "structural": "domain content; transfers from bacterial modular-PKS engineering precedent",
    "control": "iteration program (which cycle a domain fires on / iteration count); the 0.57 frontier",
}


def _aggregate() -> list[dict]:
    """Per (axis, kind): the tier the code currently asserts, the number of distinct designs in the
    whole-manifold 1-edit neighbourhood whose most-realizable route uses that kind, and example
    instances. n_designs is the planner-relevant weight: how much design space rides on this cost."""
    man = natural_manifold()
    ds = design_space(man)
    all_designs = ds["engineerable"] + ds["frontier"] + ds["speculative"]

    n_designs: dict[tuple[str, str], int] = defaultdict(int)
    tier: dict[tuple[str, str], str] = {}
    examples: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in all_designs:
        seen: set[tuple[str, str]] = set()
        for e in r.edits:
            key = (e.axis.value, e.kind)
            tier[key] = e.tier.name
            examples[key].add(e.detail)
            seen.add(key)
        for key in seen:
            n_designs[key] += 1

    # 6-MSA neighbourhood (the worked figure) -- per-kind design count there too.
    msa = next(t for t in man if "6-MSA" in t.id)
    n_msa: dict[tuple[str, str], int] = defaultdict(int)
    for prog in one_edit_neighbours(msa):
        r = realize(prog, man)
        if r.verdict == "natural":
            continue
        for key in {(e.axis.value, e.kind) for e in r.edits}:
            n_msa[key] += 1

    rows = []
    for key in sorted(tier, key=lambda k: (k[0], -n_designs[k])):
        axis, kind = key
        ex = sorted(examples[key])
        rows.append({
            "axis": axis,
            "kind": kind,
            "class_tier": tier[key],
            "n_designs_manifold": n_designs[key],
            "n_designs_6msa": n_msa.get(key, 0),
            "example_instances": "; ".join(ex[:8]) + (" ..." if len(ex) > 8 else ""),
            "axis_note": _AXIS_NOTE[axis],
        })
    return rows


def _write_snapshot(rows: list[dict]) -> None:
    _SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["axis", "kind", "class_tier", "n_designs_manifold", "n_designs_6msa",
            "example_instances", "axis_note"]
    with _SNAPSHOT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _seed_table(rows: list[dict]) -> bool:
    """Seed the hand-edited curation table once; never clobber existing reading."""
    if _TABLE.exists():
        return False
    _TABLE.parent.mkdir(parents=True, exist_ok=True)
    cols = ["axis", "kind", "example_instances", "n_designs_manifold", "class_tier",
            "curated_tier", "demonstrated", "system", "precedent_refs", "confidence",
            "notes", "sourcing_status"]
    with _TABLE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "axis": r["axis"],
                "kind": r["kind"],
                "example_instances": r["example_instances"],
                "n_designs_manifold": r["n_designs_manifold"],
                "class_tier": r["class_tier"],
                "curated_tier": "",        # DOCUMENTED / PLAUSIBLE / SPECULATIVE after reading
                "demonstrated": "",        # yes / no / partial -- realized in a real PKS?
                "system": "",              # which PKS/organism the precedent comes from
                "precedent_refs": "",      # real citations only -- never fabricated
                "confidence": "",          # high / med / low
                "notes": "",               # yield / success-rate / caveat
                "sourcing_status": "uncurated",
            })
    return True


def main() -> None:
    rows = _aggregate()
    _write_snapshot(rows)
    seeded = _seed_table(rows)

    structural = [r for r in rows if r["axis"] == "structural"]
    control = [r for r in rows if r["axis"] == "control"]
    print("Bounded literature-curation worklist (edit kinds in the 1-edit design space)\n")
    print(f"{'axis':11s} {'kind':13s} {'tier(now)':11s} {'#designs':>9s}  example instances")
    print("-" * 100)
    for grp in (structural, control):
        for r in grp:
            print(f"{r['axis']:11s} {r['kind']:13s} {r['class_tier']:11s} "
                  f"{r['n_designs_manifold']:9d}  {r['example_instances']}")
        print()
    print(f"=> {len(rows)} edit kinds total: {len(structural)} structural "
          f"(bacterial-PKS-engineering precedent), {len(control)} control (iteration-grammar frontier).")
    print(f"   That is the entire reading list. Snapshot: {_SNAPSHOT.relative_to(_ROOT)}")
    print(f"   Curation table {'SEEDED' if seeded else 'already exists (left untouched)'}: "
          f"{_TABLE.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
