"""CLI: core-match gate-zero over the fungal PKS pairs (tightened metric).

Headline = SKELETON-COMPLETE match: a producible PKS-core carbon skeleton accounts for
ALL of the product's carbon skeleton modulo *separable* additive decorations (each
unmatched carbon component attaches to the core by exactly one bond). This is the
trainable-set number.

Also reported (diagnostics, NOT the trainable set):
  * any-subgraph match -- permissive upper bound (core embeds somewhere in y);
  * coverage distribution -- fraction of y's carbons accounted for by the matched core,
    which quantifies how heavily tailored the dataset is.

    python -m lpi.cli.corematch
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.data.mibig import PROCESSED, ROOT
from lpi.eval.corematch import (best_any_subgraph, best_skeleton_complete,
                                build_core_library, coverage)

RDLogger.DisableLog("rdApp.*")
_COV_BINS = (0.9, 0.7, 0.5, 0.3)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()))


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    path = Path(argv[0]) if argv else (PROCESSED / "fungal_pks_pairs.parquet")
    df = pd.read_parquet(path)
    lib = build_core_library()
    print(f"core library: {len(lib)} skeletons ({sum(c.is_cyclic for c in lib)} cyclic)")

    rows = []
    sk_complete = any_only = none = err = 0
    covs: list[float] = []
    for _, r in df.iterrows():
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        try:
            y = _largest_fragment(M.mol_from_smiles(smi))
        except Exception:  # noqa: BLE001
            err += 1
            rows.append((r["bgc_id"], r.get("compound_name"), "parse_error", 0.0, ""))
            continue
        sc = best_skeleton_complete(y, lib)
        if sc is not None:
            sk_complete += 1
            cov = coverage(sc, y)
            covs.append(cov)
            rows.append((r["bgc_id"], r.get("compound_name"), "skeleton_complete",
                         round(cov, 3), sc.smiles))
        elif best_any_subgraph(y, lib) is not None:
            any_only += 1
            rows.append((r["bgc_id"], r.get("compound_name"), "subgraph_only", 0.0, ""))
        else:
            none += 1
            rows.append((r["bgc_id"], r.get("compound_name"), "no_match", 0.0, ""))

    out = ROOT / "results" / "corematch.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bgc_id", "name", "class", "coverage", "matched_core_smiles"])
        w.writerows(rows)

    total = len(df)
    print("-" * 64)
    print(f"targets                                  : {total}")
    print(f"SKELETON-COMPLETE match (trainable set)  : {sk_complete}")
    print(f"subgraph-only (diagnostic, NOT trainable): {any_only}")
    print(f"no match (genuine ceiling)               : {none}")
    print(f"parse error                              : {err}")
    print(f"any-subgraph upper bound (sk+subgraph)   : {sk_complete + any_only}")
    print("-" * 64)
    print("coverage of skeleton-complete matches (core C / product C):")
    for thr in _COV_BINS:
        print(f"  >= {thr}: {sum(1 for c in covs if c >= thr)}")
    print(f"\n(detail -> {out})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
