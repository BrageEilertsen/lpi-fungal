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
from lpi.search.beam import OperatorSpec, formula_feasible, search

ALLOWED_ELEMENTS = {1, 6, 8}  # H, C, O
# EXACT-match scan cap. Empirically every scanned C13-41 fungal PKS product is unreachable
# at exact match (they are core + post-PKS tailoring), so exhaustively enumerating cores
# for very large chains buys ~nothing at exact level while costing tens of seconds each.
# We cap exact-match scanning at C20 (covers the realistic untailored-core range with
# margin; the largest reachable so far is the C12 'BAB') and report C>20 separately. The
# CORE-match scan (core subgraph of y) examines ALL sizes -- that is where large products
# can still contribute via an embedded core.
CARBON_CAP = 20
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
    return OperatorSpec(starters=("acetyl", "propionyl", "hexanoyl"))


def scan_target(bgc_id: str, name: str, smiles: str,
                beam_width: int = 4000, max_executions: int = 60000) -> ReachRow:
    # With the formula prefilter + feasibility gate, executor calls are rare; beam_width
    # 4000 is exhaustive for the C<=20 exact-match range. (An earlier, smaller beam caused
    # false-negatives on mellein / 6-hydroxymellein -- fixed.)
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

    # Target-level formula feasibility: if no program's product formula can equal this
    # target, it is provably unreachable -- skip the (expensive) structural search.
    from lpi.search.beam import _formula_cho
    spec = _scan_spec()
    if not formula_feasible(_formula_cho(mol), spec, max_cycles=carbons // 2 + 1):
        return ReachRow(bgc_id, name, smiles, carbons, "unreachable")

    result = search(M.canonical_smiles(mol), spec,
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
