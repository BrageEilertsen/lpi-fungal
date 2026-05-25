"""CLI: run the verifier/search over the curated set; report Z*(y) + Gate 1 status.

    python -m lpi.cli.search
"""

from __future__ import annotations

import statistics
import sys

from lpi.data.curated import load_all
from lpi.search.beam import search


def _target(entry):
    return (entry.expected_final_smiles if entry.scored_level == "cyclized"
            else entry.expected_linear_smiles)


def main() -> int:
    entries = [e for e in load_all() if e.tier in ("gate", "sanity")
               and _target(e) is not None]
    sizes: list[int] = []
    runtimes: list[float] = []
    all_known_found = True

    print(f"{'name':36} {'|Z*|':>4} {'known∈Z*':>9} {'exec':>6} {'pruned':>7} {'t(s)':>6}")
    print("-" * 78)
    for e in entries:
        r = search(_target(e))
        known = any(p == e.program for p in r.z_star)
        all_known_found = all_known_found and known
        sizes.append(r.size)
        runtimes.append(r.stats.runtime_s)
        print(f"{e.name[:36]:36} {r.size:>4} {str(known):>9} "
              f"{r.stats.n_programs_executed:>6} {r.stats.n_pruned_overshoot:>7} "
              f"{r.stats.runtime_s:>6.2f}")

    print("-" * 78)
    dist = {n: sizes.count(n) for n in sorted(set(sizes))}
    print(f"|Z*(y)| distribution: {dist}  (min={min(sizes)} max={max(sizes)} "
          f"mean={statistics.mean(sizes):.2f})")
    print(f"runtime/target: mean={statistics.mean(runtimes):.2f}s max={max(runtimes):.2f}s")
    print(f"known program in Z*(y) for every system: {all_known_found}")
    print(f"\nGATE 1: {'MET' if all_known_found else 'NOT MET'} "
          f"(known program recovered for all curated systems; search practical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
