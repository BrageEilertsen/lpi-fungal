"""Loader for the hand-curated validation set in ``data/curated/``.

Each entry is a YAML file describing a fungal polyketide whose biosynthetic program
is (claimed to be) established in the literature, with an explicit
:class:`~lpi.chem.program.Program` and the expected structure at a declared scoring
level. See ``data/curated/SCHEMA.md`` for the field spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from lpi.chem.program import Cycle, Extender, Program, ReductionState, Release

CURATED_DIR = Path(__file__).resolve().parents[3] / "data" / "curated"


@dataclass
class CuratedEntry:
    name: str
    organism: str
    bgc: str
    subclass: str  # HR / NR / PR
    citation: str
    program: Program
    scored_level: str  # "linear" or "cyclized"
    expected_linear_smiles: str | None
    expected_final_smiles: str | None
    tier: str  # "gate" / "sanity" / "bonus"
    ground_truth: str  # "independent" / "self_consistent" / "pending"
    program_confidence: str  # "high" / "medium" / "low"
    needs_expert_review: bool
    notes: str
    source_path: Path

    @property
    def expected_smiles(self) -> str | None:
        return (
            self.expected_final_smiles
            if self.scored_level == "cyclized"
            else self.expected_linear_smiles
        )


def _parse_cycle(d: dict) -> Cycle:
    return Cycle(
        reduction=ReductionState(d.get("reduction", "keto")),
        c_methyl=bool(d.get("c_methyl", False)),
        extender=Extender(d.get("extender", "malonyl")),
    )


def _parse_program(d: dict) -> Program:
    return Program(
        starter=d.get("starter", "acetyl"),
        cycles=tuple(_parse_cycle(c) for c in d.get("cycles", [])),
        release=Release(d.get("release", "hydrolysis")),
    )


def load_entry(path: Path) -> CuratedEntry:
    raw = yaml.safe_load(path.read_text())
    return CuratedEntry(
        name=raw["name"],
        organism=raw.get("organism", ""),
        bgc=raw.get("bgc", ""),
        subclass=raw.get("subclass", ""),
        citation=raw.get("citation", ""),
        program=_parse_program(raw["program"]),
        scored_level=raw.get("scored_level", "linear"),
        expected_linear_smiles=raw.get("expected_linear_smiles"),
        expected_final_smiles=raw.get("expected_final_smiles"),
        tier=raw.get("tier", "gate"),
        ground_truth=raw.get("ground_truth", "self_consistent"),
        program_confidence=raw.get("program_confidence", "medium"),
        needs_expert_review=bool(raw.get("needs_expert_review", False)),
        notes=raw.get("notes", ""),
        source_path=path,
    )


def load_all(curated_dir: Path = CURATED_DIR) -> list[CuratedEntry]:
    entries = [load_entry(p) for p in sorted(curated_dir.glob("*.yaml"))]
    return entries
