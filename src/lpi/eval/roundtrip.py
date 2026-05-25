"""Curated-set round-trip scoring and the Gate 0 go/no-go computation.

Executes each curated program and compares the executor output at the entry's declared
scoring level against its expected structure. Honest by construction: an entry whose
ground truth is ``pending`` (no usable expected SMILES) is reported as PENDING, never
silently counted as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from lpi.chem import mol as M
from lpi.data.curated import CuratedEntry, load_all
from lpi.executor import core


@dataclass
class RoundTripResult:
    entry: CuratedEntry
    status: str  # PASS / FAIL / PENDING / ERROR
    executor_smiles: str | None
    expected_smiles: str | None
    detail: str


def score_entry(entry: CuratedEntry) -> RoundTripResult:
    expected = entry.expected_smiles
    if entry.ground_truth == "pending" or not expected:
        # Try to execute anyway (surfaces executor errors), but cannot score a match.
        try:
            res = core.run(entry.program)
            got = (
                M.canonical_smiles(res.final)
                if (entry.scored_level == "cyclized" and res.final is not None)
                else M.canonical_smiles(res.linear)
            )
        except Exception as exc:  # noqa: BLE001
            got = None
            return RoundTripResult(entry, "PENDING", None, expected,
                                   f"pending ground truth; executor raised: {exc}")
        return RoundTripResult(entry, "PENDING", got, expected,
                               "pending authoritative ground truth (not scored)")

    try:
        res = core.run(entry.program)
    except Exception as exc:  # noqa: BLE001
        return RoundTripResult(entry, "ERROR", None, expected, f"executor raised: {exc}")

    scored = res.final if entry.scored_level == "cyclized" else res.linear
    if scored is None:
        return RoundTripResult(
            entry, "FAIL", None, expected,
            f"no product at level '{entry.scored_level}' (cyclization not produced)",
        )

    got_smiles = M.canonical_smiles(scored)
    try:
        match = M.structures_match(scored, M.mol_from_smiles(expected))
    except ValueError as exc:
        return RoundTripResult(entry, "ERROR", got_smiles, expected,
                               f"could not parse expected SMILES: {exc}")
    # Phase-0 executor is achiral by design (stereo is a Phase-2 head, paper Section 4.1):
    # a constitutional (stereo-insensitive) match is a PASS; note whether stereo agreed.
    if match["match_flat"]:
        stereo = "stereo+constitution" if match["match"] else "constitution (stereo deferred)"
        return RoundTripResult(entry, "PASS", got_smiles, expected, f"matched: {stereo}")
    return RoundTripResult(entry, "FAIL", got_smiles, expected, "structure mismatch")


@dataclass
class GateReport:
    results: list[RoundTripResult]

    def by_tier(self, tier: str) -> list[RoundTripResult]:
        return [r for r in self.results if r.entry.tier == tier]

    def counts(self, tier: str | None = None) -> dict[str, int]:
        rs = self.results if tier is None else self.by_tier(tier)
        out = {"PASS": 0, "FAIL": 0, "PENDING": 0, "ERROR": 0}
        for r in rs:
            out[r.status] += 1
        out["TOTAL"] = len(rs)
        return out

    def gate_pass_rate(self) -> float:
        c = self.counts("gate")
        return c["PASS"] / c["TOTAL"] if c["TOTAL"] else 0.0

    def gate_met(self, threshold: float = 0.80) -> bool:
        return self.gate_pass_rate() >= threshold


def run_all() -> GateReport:
    return GateReport([score_entry(e) for e in load_all()])
