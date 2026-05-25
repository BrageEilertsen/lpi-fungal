"""Regression: every independently-grounded curated entry must round-trip exactly.

This is the executor's correctness regression. Entries whose ``ground_truth`` is
``pending`` are not asserted (they are blocked on authoritative structures, tracked in
QUESTIONS.md) but are surfaced via a reporting test so they cannot be quietly forgotten.
"""

from __future__ import annotations

import pytest

from lpi.data.curated import load_all
from lpi.eval.roundtrip import score_entry

_ALL = load_all()
_INDEPENDENT = [e for e in _ALL if e.ground_truth == "independent"]
_PENDING = [e for e in _ALL if e.ground_truth == "pending"]


@pytest.mark.parametrize("entry", _INDEPENDENT, ids=lambda e: e.source_path.stem)
def test_independent_entries_roundtrip(entry):
    result = score_entry(entry)
    assert result.status == "PASS", (
        f"{entry.name}: {result.detail}\n  exec={result.executor_smiles}\n"
        f"  want={result.expected_smiles}"
    )


def test_there_are_enough_independent_gate_entries():
    gate_independent = [e for e in _INDEPENDENT if e.tier == "gate"]
    assert len(gate_independent) >= 4  # orsellinic, 6-MSA, TAL, LovF diketide


@pytest.mark.parametrize("entry", _PENDING, ids=lambda e: e.source_path.stem)
def test_pending_entries_do_not_silently_pass(entry):
    # A pending entry must NOT report PASS (that would be a circular/false success).
    result = score_entry(entry)
    assert result.status != "PASS", (
        f"{entry.name} is marked pending but scored PASS -- check provenance"
    )
