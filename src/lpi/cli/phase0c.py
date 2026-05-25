"""Phase 0c gate-zero: gene+formula-budgeted joint (z_pks, z_tail) verification.

On the tractable subset (MIBiG-annotated gene budget, <= MAX_TAILORING_GENES tailoring
genes, product carbons <= MAX_CARBONS, single fragment), run the joint search and report:
  (a) how many clusters get an exact-by-budget (program, tailoring) explanation;
  (b) the |Z*(y)| distribution (parsimonious) -- are explanations near-unique;
  (c) the edit-count distribution and runtime.
Clusters above the tailoring-gene cap or size limit are reported as the hard tier.

Gene budget source: MIBiG gene annotations (Phase 0b-lite; ~64 clusters). Extending to all
326 needs HMMER on NCBI-fetched sequences (Phase 0b-full, scaffolded, not run here) -- this
gate-zero is the fast falsification on the annotated subset.

    python -m lpi.cli.phase0c
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.data.genebudget import gene_budget
from lpi.data.mibig import PROCESSED, ROOT
from lpi.search.joint import explain_cluster

RDLogger.DisableLog("rdApp.*")
MAX_TAILORING_GENES = 8
MAX_CARBONS = 20
EMAIL = "brageei@uio.no"


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()))


def main(argv: list[str] | None = None) -> int:
    import csv

    import pandas as pd

    offline = "--offline" in (argv or [])
    df = pd.read_parquet(PROCESSED / "fungal_pks_pairs.parquet")
    rows = []
    verified = hard = skipped = 0
    zdist: collections.Counter[int] = collections.Counter()
    edist: collections.Counter[int] = collections.Counter()
    runtimes: list[float] = []

    for _, r in df.iterrows():
        bgc = r["bgc_id"]
        budget = gene_budget(bgc, email=EMAIL, offline=offline)
        if budget is None:
            skipped += 1
            continue  # no MIBiG annotation and no fetchable GenBank CDS
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        try:
            y = _largest_fragment(M.mol_from_smiles(smi))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        n_carbons = sum(a.GetAtomicNum() == 6 for a in y.GetAtoms())
        n_tail = sum(budget.values())
        if n_tail > MAX_TAILORING_GENES or n_carbons > MAX_CARBONS or n_carbons < 4:
            hard += 1
            rows.append((bgc, r.get("compound_name"), "hard_tier", "", -1, -1, ""))
            continue
        res = explain_cluster(M.canonical_smiles(y), budget)
        runtimes.append(res.runtime_s)
        if res.verified:
            verified += 1
            zdist[res.z_star_size] += 1
            edist[res.min_edits] += 1
            top = res.explanations[0]
            rows.append((bgc, r.get("compound_name"), "verified", top.core_smiles,
                         res.min_edits, res.z_star_size, str(top.edit_counts)))
        else:
            rows.append((bgc, r.get("compound_name"), "ceiling", "", -1, -1, ""))

    out = ROOT / "results" / "phase0c.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bgc_id", "name", "class", "core_smiles", "min_edits",
                    "z_star", "edits"])
        w.writerows(rows)

    scanned = verified + (len(rows) - verified - hard)
    print(f"Phase 0c gate-zero (MIBiG-annotated tractable subset)")
    print("-" * 62)
    print(f"  annotated clusters scanned (<= {MAX_TAILORING_GENES} genes, "
          f"<= C{MAX_CARBONS}): {scanned}")
    print(f"  hard tier (too many genes / too large)        : {hard}")
    print(f"  no gene budget (no MIBiG ann. + no GenBank CDS): {skipped}")
    print("-" * 62)
    print(f"  EXACT-BY-BUDGET verified                       : {verified} / {scanned}")
    print(f"  ceiling (no budgeted explanation)              : {scanned - verified}")
    print(f"  min-edits distribution  : {dict(sorted(edist.items()))}")
    print(f"  |Z*(y)| distribution     : {dict(sorted(zdist.items()))}")
    if runtimes:
        import statistics
        print(f"  runtime/cluster: mean={statistics.mean(runtimes):.2f}s "
              f"max={max(runtimes):.2f}s")
    print(f"\n(detail -> {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
