"""CLI: core-match gate-zero over the fungal PKS pairs.

A target is core-matched if a producible PKS-core carbon skeleton embeds in the product's
carbon skeleton (absorbing additive tailoring + NRPS-appended amino acids). Because a
small ring embeds in almost any aromatic product, raw subgraph match is permissive; the
decision-useful signal is COVERAGE = matched-core carbons / product carbons. High coverage
=> the predicted backbone IS most of the molecule (tailoring is minor) => meaningfully
trainable.

    python -m lpi.cli.corematch
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.data.mibig import PROCESSED, ROOT
from lpi.eval.corematch import best_core_match, build_core_library, coverage

RDLogger.DisableLog("rdApp.*")
_THRESHOLDS = (0.9, 0.7, 0.5, 0.3)


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: sum(a.GetAtomicNum() == 6 for a in m.GetAtoms()))


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    path = Path(argv[0]) if argv else (PROCESSED / "fungal_pks_pairs.parquet")
    df = pd.read_parquet(path)
    lib = build_core_library()
    n_cyc = sum(c.is_cyclic for c in lib)
    print(f"core library: {len(lib)} distinct skeletons ({n_cyc} cyclic ring systems, "
          f"{len(lib) - n_cyc} linear)")

    rows = []
    covs: list[float] = []
    none = err = 0
    for _, r in df.iterrows():
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        try:
            y = _largest_fragment(M.mol_from_smiles(smi))
        except Exception:  # noqa: BLE001
            err += 1
            rows.append((r["bgc_id"], r.get("compound_name"), 0.0, "", "parse_error"))
            continue
        m = best_core_match(y, lib)
        if m is None:
            none += 1
            rows.append((r["bgc_id"], r.get("compound_name"), 0.0, "", "no_match"))
            continue
        cov = coverage(m, y)
        covs.append(cov)
        cls = "cyclic_core" if m.is_cyclic else "linear_only"
        rows.append((r["bgc_id"], r.get("compound_name"), round(cov, 3), m.smiles, cls))

    out = ROOT / "results" / "corematch.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bgc_id", "name", "coverage", "matched_core_smiles", "class"])
        w.writerows(rows)

    total = len(df)
    matched = len(covs)
    print("-" * 60)
    print(f"targets                          : {total}")
    print(f"core-match (any subgraph embed)  : {matched}  (permissive)")
    print(f"no core embeds (genuine ceiling) : {none}")
    print(f"parse error                      : {err}")
    print("-" * 60)
    print("core-COVERAGE (matched-core C / product C) -- the trainable-set gate:")
    for thr in _THRESHOLDS:
        print(f"  coverage >= {thr}: {sum(1 for c in covs if c >= thr)} targets")
    print(f"\n(detail -> {out})")
    print("The coverage threshold is a modeling judgment (Brage's call). Suggested: the")
    print(">=0.5 set is the moderate trainable set; >=0.7 the strict one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
