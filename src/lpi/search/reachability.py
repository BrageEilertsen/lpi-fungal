"""Phase-2 gate-zero: which MIBiG fungal PKS products can the executor actually reach?

For every fungal PKS pair, ask whether there exists ANY program (under a permissive
operator spec) whose executor image matches the product -- i.e. whether Z*(y) is
non-empty. That count is the real trainable-set size: the controller can only be
supervised on end-products the deterministic verifier can reproduce.

Cheap pre-filters (a product the executor provably cannot make is skipped, not searched):
  * elements must be a subset of {C, H, O} -- the rule set makes only polyketide C/H/O
    backbones with phenol/lactone oxygens (no N, halogen, S, P, glycosylation, etc.);
  * a single connected molecule (no salts, dimers, conjugates);
  * carbon count within [4, CARBON_CAP] -- larger chains are not scanned (a tractability
    bound, reported separately as 'too_large', NOT counted as unreachable).

Honesty: this is a lower bound on reachability under the *current* rule set. Products
needing tailoring outside the rule set (PT multi-ring aromatics, oxidation, prenylation,
halogenation, Diels-Alder, ...) come back unreachable -- which is the finding, per the
Phase-2 plan: it quantifies how much executor coverage must expand before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from lpi.chem import mol as M
from lpi.search.beam import OperatorSpec, search

ALLOWED_ELEMENTS = {1, 6, 8}  # H, C, O
CARBON_CAP = 16  # chains larger than this are not scanned (tractability bound)
CARBON_MIN = 4


@dataclass
class ReachRow:
    bgc_id: str
    name: str
    smiles: str
    carbons: int
    status: str  # reachable / unreachable / skip_element / skip_fragment / too_large / too_small / parse_error / budget
    z_star_size: int = 0
    example_program: str | None = None


@dataclass
class ReachReport:
    rows: list[ReachRow] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.rows if r.status == status)

    @property
    def reachable(self) -> int:
        return self.count("reachable")

    @property
    def scanned(self) -> int:
        return sum(1 for r in self.rows if r.status in ("reachable", "unreachable", "budget"))


def _element_ok(mol: Chem.Mol) -> bool:
    return all(a.GetAtomicNum() in ALLOWED_ELEMENTS for a in mol.GetAtoms())


def _scan_spec() -> OperatorSpec:
    # Permissive: the realistic fungal starters, every reduction state, C-MeT on, every
    # implemented release mode. (Search recovers the program from many alternatives.)
    return OperatorSpec(starters=("acetyl", "propionyl"))


def scan_target(bgc_id: str, name: str, smiles: str,
                beam_width: int = 8000, max_executions: int = 30000) -> ReachRow:
    # beam_width is exhaustive for C<=13 (4^5=1024, 4^6=4096 reduction patterns), where
    # all products the current rule set can reach actually live. Larger/hard targets that
    # exhaust max_executions are reported as 'budget' (indeterminate), NOT 'unreachable' --
    # avoiding the beam-width false-negatives that an earlier, smaller beam produced on
    # mellein / 6-hydroxymellein.
    try:
        mol = M.mol_from_smiles(smiles)
    except ValueError:
        return ReachRow(bgc_id, name, smiles, 0, "parse_error")
    if len(Chem.GetMolFrags(mol)) > 1:
        return ReachRow(bgc_id, name, smiles, 0, "skip_fragment")
    if not _element_ok(mol):
        return ReachRow(bgc_id, name, smiles, 0, "skip_element")
    carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    if carbons < CARBON_MIN:
        return ReachRow(bgc_id, name, smiles, carbons, "too_small")
    if carbons > CARBON_CAP:
        return ReachRow(bgc_id, name, smiles, carbons, "too_large")

    result = search(M.canonical_smiles(mol), _scan_spec(),
                    beam_width=beam_width, max_executions=max_executions)
    if result.size > 0:
        return ReachRow(bgc_id, name, smiles, carbons, "reachable",
                        z_star_size=result.size, example_program=repr(result.z_star[0]))
    status = "budget" if result.stats.budget_exhausted else "unreachable"
    return ReachRow(bgc_id, name, smiles, carbons, status)


def scan_parquet(parquet_path: Path, progress: bool = False) -> ReachReport:
    import sys

    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows: list[ReachRow] = []
    total = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        if not smi:
            rows.append(ReachRow(r["bgc_id"], str(r.get("compound_name")), "", 0, "parse_error"))
            continue
        row = scan_target(r["bgc_id"], str(r.get("compound_name")), smi)
        rows.append(row)
        if progress and (i % 25 == 0 or row.status in ("reachable", "budget")):
            print(f"[{i + 1}/{total}] {row.bgc_id} {row.status} "
                  f"(reachable so far: {sum(1 for x in rows if x.status=='reachable')})",
                  file=sys.stderr, flush=True)
    return ReachReport(rows)


def write_csv(rep: ReachReport, out: Path) -> Path:
    import csv

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bgc_id", "name", "carbons", "status", "z_star_size", "example_program"])
        for r in rep.rows:
            w.writerow([r.bgc_id, r.name, r.carbons, r.status, r.z_star_size,
                        r.example_program or ""])
    return out
