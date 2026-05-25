"""CLI: Phase-2 gate-zero reachability scan over the fungal PKS pairs.

    python -m lpi.cli.reachability
"""

from __future__ import annotations

import sys
from pathlib import Path

from lpi.data.mibig import PROCESSED, ROOT
from lpi.search.reachability import CARBON_CAP, ReachReport, scan_parquet, write_csv


def format_report(rep: ReachReport) -> str:
    n = len(rep.rows)
    order = ["reachable", "unreachable", "budget", "too_large", "too_small",
             "skip_element", "skip_fragment", "parse_error"]
    lines = [f"Phase-2 gate-zero: executor reachability over {n} fungal PKS pairs", "-" * 64]
    for s in order:
        c = rep.count(s)
        if c:
            lines.append(f"  {s:14}: {c}")
    lines.append("-" * 64)
    lines.append(f"  scanned (search actually run) : {rep.scanned}")
    lines.append(f"  REACHABLE (non-empty Z*(y))   : {rep.reachable}")
    lines.append("")
    lines.append("Reachable products (the trainable set under the current rule set):")
    for r in rep.rows:
        if r.status == "reachable":
            lines.append(f"  {r.bgc_id}  C{r.carbons:<2} |Z*|={r.z_star_size}  {r.name}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0]) if argv else (PROCESSED / "fungal_pks_pairs.parquet")
    rep = scan_parquet(path, progress=True)
    out = write_csv(rep, ROOT / "results" / "reachability.csv")
    print(format_report(rep))
    print(f"\n(carbon cap for scanning = C{CARBON_CAP}; full per-target detail -> {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
