"""CLI: run the curated round-trip and print the Gate 0 status table.

    python -m lpi.cli.roundtrip
"""

from __future__ import annotations

import sys

from lpi.eval.roundtrip import GateReport, run_all

_GLYPH = {"PASS": "PASS", "FAIL": "FAIL", "PENDING": "PEND", "ERROR": "ERR "}


def format_report(report: GateReport) -> str:
    lines: list[str] = []
    lines.append(f"{'status':6} {'tier':7} {'gt':12} {'name'}")
    lines.append("-" * 88)
    order = {"gate": 0, "sanity": 1, "bonus": 2}
    for r in sorted(report.results, key=lambda x: (order.get(x.entry.tier, 9), x.entry.name)):
        lines.append(
            f"{_GLYPH[r.status]:6} {r.entry.tier:7} {r.entry.ground_truth:12} {r.entry.name}"
        )
        if r.status in ("FAIL", "ERROR"):
            lines.append(f"        exec : {r.executor_smiles}")
            lines.append(f"        want : {r.expected_smiles}")
            lines.append(f"        why  : {r.detail}")
    lines.append("-" * 88)
    for tier in ("gate", "sanity", "bonus"):
        c = report.counts(tier)
        if c["TOTAL"]:
            lines.append(
                f"{tier:7}: {c['PASS']}/{c['TOTAL']} pass  "
                f"(fail={c['FAIL']} pending={c['PENDING']} error={c['ERROR']})"
            )
    rate = report.gate_pass_rate()
    lines.append("")
    lines.append(f"GATE 0 round-trip (gate tier): {rate:.0%}  "
                 f"-> {'MET' if report.gate_met() else 'NOT MET'} (threshold 80%)")
    return "\n".join(lines)


def main() -> int:
    report = run_all()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
