"""Test the 'two orthogonal walls' hypothesis over the 326 fungal PKS products.

Claim: the reconstructibility ceiling is not one wall but two, by subclass character:
  * NR/aromatic products  -> reduction is architecturally trivial (non-reducing = all-keto);
    the hard part is CYCLIZATION regiochemistry (deterministic; coverage-addressable).
  * HR/reduced products    -> cyclization is often simple (hydrolysis/lactone); the hard part
    is PER-CYCLE REDUCTION programming (the grammar wall: not positional, not transferable).
If true, the difficulty is orthogonal across subclasses and 'data scarcity' is really a
chemistry-coverage problem for NR and a computability problem for HR.

Subclass proxy from structure (MIBiG lacks the label for 304/326):
  frac_aromatic = aromatic C / total C; high => NR-like (aromatized poly-beta-keto),
  ~0 with reduced aliphatic chains => HR-like, intermediate => PR/mixed.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def features(mol):
    atoms = mol.GetAtoms()
    nC = sum(a.GetAtomicNum() == 6 for a in atoms)
    arom_c = sum(a.GetAtomicNum() == 6 and a.GetIsAromatic() for a in atoms)
    nO = sum(a.GetAtomicNum() == 8 for a in atoms)
    nN = sum(a.GetAtomicNum() == 7 for a in atoms)
    nH = sum(a.GetTotalNumHs() for a in atoms)
    return nC, arom_c, nO, nN, nH


def subclass(nC, arom_c, nO, nN, nH):
    if nN > 0:
        return "hybrid/non-PKS (N)"
    frac = arom_c / nC if nC else 0
    if frac >= 0.40:
        return "NR-like (aromatic)"
    if frac <= 0.10 and (nH / nC) >= 1.4:  # saturated, aliphatic
        return "HR-like (reduced)"
    return "PR/mixed"


def main():
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet")
    reach = pd.read_csv("results/reachability_v2.csv").set_index("bgc_id")["status"].to_dict()
    by_sub = Counter()
    reach_by_sub = Counter()
    cyc_limited = Counter()  # aromatic + unreachable = cyclization-limited
    for _, r in pairs.iterrows():
        smi = r["product_smiles_canonical"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        nC, arom_c, nO, nN, nH = features(mol)
        sub = subclass(nC, arom_c, nO, nN, nH)
        st = reach.get(r["bgc_id"], "?")
        by_sub[sub] += 1
        if st == "reachable":
            reach_by_sub[sub] += 1
    total = sum(by_sub.values())
    print(f"Subclass-proxy split over {total} fungal PKS products:\n")
    print(f"  {'subclass':28s} {'n':>4s}  {'%':>5s}  {'reachable':>9s}")
    for sub, n in by_sub.most_common():
        print(f"  {sub:28s} {n:4d}  {100*n/total:4.0f}%  {reach_by_sub[sub]:9d}")
    nr = by_sub["NR-like (aromatic)"] + by_sub["PR/mixed"]
    hr = by_sub["HR-like (reduced)"]
    hyb = by_sub["hybrid/non-PKS (N)"]
    print(f"\n  Two-walls reading:")
    print(f"    NR/PR aromatic (cyclization-limited, reduction trivial): {nr} "
          f"({100*nr/total:.0f}%)")
    print(f"    HR reduced (per-cycle-reduction grammar wall):           {hr} "
          f"({100*hr/total:.0f}%)")
    print(f"    N-containing hybrids / non-PKS (separate):               {hyb} "
          f"({100*hyb/total:.0f}%)")


if __name__ == "__main__":
    main()
