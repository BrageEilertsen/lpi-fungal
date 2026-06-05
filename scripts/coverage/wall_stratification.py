"""Wall stratification: pin the EXACT current-map HR denominator the pocket bet is against -- named + cell-
tagged -- in-house, no AF2/docking/MD. The design frame for the deliberate pocket-test start, not the bet.

Re-derives (not cites) the two-walls split from the data: NR/PR aromatic (coverage-addressable -> build-loop,
engineering) vs HR-reduced (per-cycle reductive-ROUTING wall -> the pocket bet). Names the HR set, tags each
by forward-index status, and frames the role x cycle-position pocket test so it cannot be fooled:

  * The HR cores ARE the reductive-ROUTING cell (load-bearing; what the 0.567 wall actually is). The powered
    routing test draws its sampling frame from HERE -- never from the aromatic/positional (Gate-3,
    citrinin's static-readable) cell, or it re-confirms an irrelevant success (the trap).
  * length-determined routing (termination) sub-cell: expected ~EMPTY. Within a target formula the carbon
    budget all-but-pins the cycle count (only the C-methyl/malonyl trade-off can move it), so termination is
    not a free routing decision -> the cell is context-determined reductive routing (which beta-keto
    reduces), the dynamics-hard slice with no cheap docked escape. Tagged here; a non-trivial length cell
    would be a (pleasant) surprise, settled when Z* is enumerable / at the docked rung.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, "scripts")
import two_walls as TW  # noqa: E402  (single source for the structural proxy: features + subclass)

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet")
    reach = pd.read_csv("results/reachability.csv").set_index("bgc_id")["status"].to_dict()
    fwd = {r["bgc_id"]: r["forward_verdict"]
           for r in csv.DictReader(open("results/forward_index.csv"))}

    rows = []
    for _, r in pairs.iterrows():
        smi = r["product_smiles_canonical"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        nC, arom_c, nO, nN, nH = TW.features(mol)
        rows.append({
            "bgc_id": r["bgc_id"], "compound": str(r["compound_name"])[:38], "formula": str(r.get("formula")),
            "subclass": TW.subclass(nC, arom_c, nO, nN, nH),
            "frac_arom": round(arom_c / nC, 2) if nC else 0.0, "H_per_C": round(nH / nC, 2) if nC else 0.0,
            "nC": nC, "reach": reach.get(r["bgc_id"], "?"), "forward": fwd.get(r["bgc_id"], "-"),
        })

    total = len(rows)
    by = Counter(x["subclass"] for x in rows)
    nr = by["NR-like (aromatic)"] + by["PR/mixed"]
    hr = by["HR-like (reduced)"]
    hyb = by["hybrid/non-PKS (N)"]
    print(f"=== Re-derived two-walls split over {total} fungal PKS (current data, NOT cited) ===")
    print(f"  NR/PR aromatic  (coverage-addressable -> build-loop, engineering): {nr:3d} ({100*nr/total:.0f}%)")
    print(f"  HR reduced      (per-cycle reductive-ROUTING wall -> POCKET BET) : {hr:3d} ({100*hr/total:.0f}%)")
    print(f"  N-hybrid/non-PKS (separate)                                      : {hyb:3d} ({100*hyb/total:.0f}%)")

    hr_rows = sorted((x for x in rows if x["subclass"] == "HR-like (reduced)"), key=lambda z: z["nC"])
    reachable_hr = sum(1 for x in hr_rows if x["reach"] == "reachable")
    c_gt20 = sum(1 for x in hr_rows if x["nC"] > 20)
    print(f"\n=== POCKET BET DENOMINATOR: {len(hr_rows)} named HR cores (the routing-cell sampling frame) ===")
    print(f"{'bgc_id':12s} {'compound':38s} {'formula':10s} {'fArom':>5s} {'H/C':>4s} {'C':>3s} "
          f"{'reach':>11s} {'forward_index':>20s}")
    for x in hr_rows:
        print(f"{x['bgc_id']:12s} {x['compound']:38s} {x['formula']:10s} {x['frac_arom']:>5} "
              f"{x['H_per_C']:>4} {x['nC']:>3} {x['reach']:>11s} {x['forward']:>20s}")

    print(f"\n  CELL FRAMING (the pocket test's design guard):")
    print(f"   * all {len(hr_rows)} are the context-determined reductive-ROUTING cell (load-bearing = what 0.567 is).")
    print(f"   * reachable HR (enumerable Z*): {reachable_hr}  -> the other {len(hr_rows)-reachable_hr} are the "
          f"coverage wall (forward-index gap/inconclusive).")
    print(f"   * {c_gt20} have C>20 -> overlap the DOF-B ordering-wall tail (same per-cycle policy degeneracy).")
    print(f"   * length/termination sub-cell: expected ~EMPTY (formula ~pins N); verify when Z* enumerable.")
    print(f"   * positional (Gate-3, citrinin's static-readable cell) lives in the AROMATIC/methylating set -- "
          f"NOT in this frame, so a routing test sampled from here cannot re-pass citrinin's cell.")

    with open("results/wall_stratification.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bgc_id", "compound", "formula", "subclass", "frac_arom",
                                          "H_per_C", "nC", "reach", "forward"])
        w.writeheader()
        for x in rows:
            w.writerow(x)
    print(f"\nwrote results/wall_stratification.csv ({total} rows; HR set = the {len(hr_rows)} pocket-bet targets)")


if __name__ == "__main__":
    main()
