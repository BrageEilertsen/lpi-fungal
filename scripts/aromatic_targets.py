"""What cores do the unreachable small monocyclic-aromatic targets actually need?

Data-driven scoping for a verifier-grounded cyclization operator: classify the aromatic
substitution pattern so we implement the right ring closure (not guess). Soundness is
guaranteed downstream by the verifier (a wrong closure simply won't match y).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

COOH = Chem.MolFromSmarts("c[CX3](=O)[OX2H1]")
AR_OH = Chem.MolFromSmarts("c[OX2H1]")
AR_OME = Chem.MolFromSmarts("c[OX2][CH3]")
AR_ME = Chem.MolFromSmarts("c[CH3]")
AR_CO = Chem.MolFromSmarts("c[CX3]=O")
LACTONE = Chem.MolFromSmarts("[OX2][CX3]=O")


def fused_aromatic_max(mol):
    ri = mol.GetRingInfo()
    arom = [set(r) for r in ri.AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    return len(arom)


def classify(mol):
    return (len(mol.GetSubstructMatches(COOH)),
            len(mol.GetSubstructMatches(AR_OH)),
            len(mol.GetSubstructMatches(AR_OME)),
            len(mol.GetSubstructMatches(AR_ME)),
            len(mol.GetSubstructMatches(LACTONE)))


def main():
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet")
    reach = pd.read_csv("results/reachability.csv").set_index("bgc_id")["status"].to_dict()
    rows = []
    for _, r in pairs.iterrows():
        if reach.get(r["bgc_id"]) == "reachable":
            continue
        smi = r["product_smiles_canonical"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        elems = Counter(a.GetSymbol() for a in mol.GetAtoms())
        if elems.get("N", 0) or fused_aromatic_max(mol) != 1:
            continue
        nC = elems.get("C", 0)
        cooh, oh, ome, me, lac = classify(mol)
        rows.append((nC, r["compound_name"], r["formula"], cooh, oh, ome, me, lac, smi))
    rows.sort(key=lambda x: (x[0], str(x[1])))
    print(f"unreachable monocyclic-aromatic, no-N: {len(rows)}\n")
    # how many are 'orsellinic/resorcylate-family' (benzoic acid + phenol(s), small)?
    fam = sum(1 for x in rows if x[3] >= 1 and x[4] >= 1 and x[0] <= 12)
    untailored = sum(1 for x in rows if x[5] == 0 and x[0] <= 12)  # no OMe, small
    print(f"  benzoic-acid + phenol, <=C12 (orsellinic/resorcylate family): {fam}")
    print(f"  of those, no O-methyl (closer to a bare core): {untailored}\n")
    print("  nC name                         formula     COOH OH OMe Me Lac  SMILES")
    for nC, name, formula, cooh, oh, ome, me, lac, smi in rows[:46]:
        print(f"  {nC:3d} {str(name)[:26]:26s} {str(formula):10s}  {cooh}   {oh}  {ome}  {me}  {lac}   {smi[:42]}")


if __name__ == "__main__":
    main()
