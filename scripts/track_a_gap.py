"""Track A gap analysis: classify the unreachable fungal products by structural family.

Honest, data-driven landscape of WHY 318/326 are not exact-reachable, so the PT-domain
chemistry input can be prioritised by payoff. We compute neutral structural descriptors
(ring system, heteroatoms, carbon band) from the product SMILES -- NOT chemistry calls --
and bucket the unreachable set. The ">=3 fused aromatic rings" bucket is the anthrone/
anthraquinone/xanthone-precursor class the PT-aromatic unlock targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
HALO = {"Cl", "Br", "I", "F"}


def fused_aromatic_max(mol) -> int:
    ri = mol.GetRingInfo()
    arom = [set(r) for r in ri.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    n = len(arom)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if len(arom[i] & arom[j]) >= 2:  # share a bond -> fused
                parent[find(i)] = find(j)
    comp = Counter(find(i) for i in range(n))
    return max(comp.values()) if comp else 0


def main():
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet")
    reach = pd.read_csv("results/reachability.csv").set_index("bgc_id")["status"].to_dict()
    cm = pd.read_csv("results/corematch.csv").set_index("bgc_id")
    cm_class = cm["class"].to_dict()

    buckets = defaultdict(list)
    for _, r in pairs.iterrows():
        bgc = r["bgc_id"]
        status = reach.get(bgc, "?")
        if status == "reachable":
            continue  # already exact
        smi = r["product_smiles_canonical"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            buckets["unparseable"].append((bgc, r["compound_name"]))
            continue
        elems = Counter(a.GetSymbol() for a in mol.GetAtoms())
        nC = elems.get("C", 0)
        has_N = elems.get("N", 0) > 0
        has_halo = any(elems.get(h, 0) for h in HALO)
        fmax = fused_aromatic_max(mol)
        core_embeds = cm_class.get(bgc) == "skeleton_complete"

        if has_N:
            key = "has_N (NRPS-hybrid / alkaloid / non-PKS-core)"
        elif fmax >= 3:
            key = "polycyclic aromatic >=3 fused (anthrone/anthraquinone/xanthone class)"
        elif fmax == 2:
            key = "bicyclic aromatic (naphthalene class; PT-naphthalene partly built)"
        elif fmax == 1:
            key = "monocyclic aromatic (small; should be near-reachable -- investigate)"
        elif nC > 24:
            key = "large aliphatic (macrolactone / decalin / too_large band)"
        else:
            key = "aliphatic / small (lactone, linear -- investigate coverage)"
        buckets[key].append((bgc, r["compound_name"], nC, fmax, status,
                             "core-embeds" if core_embeds else "no-core"))

    print(f"UNREACHABLE structural landscape (n={sum(len(v) for v in buckets.values())} "
          f"of 326; 8 reachable excluded)\n")
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        items = buckets[key]
        n_core = sum(1 for it in items if len(it) > 5 and it[5] == "core-embeds")
        print(f"  [{len(items):3d}]  {key}" +
              (f"   (core already embeds: {n_core})" if n_core else ""))

    # the actionable bucket: list the >=3-fused-aromatic candidates (the anthrone unlock)
    print("\n  >=3 fused-aromatic candidates (PT-anthrone unlock target):")
    for it in sorted(buckets["polycyclic aromatic >=3 fused (anthrone/anthraquinone/xanthone class)"],
                     key=lambda x: x[2]):
        bgc, name, nC, fmax, status, core = it
        print(f"    {bgc}  C{nC:2d} fused{fmax}  {status:12s} {core:11s} {str(name)[:34]}")


if __name__ == "__main__":
    main()
