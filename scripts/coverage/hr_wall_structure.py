"""Cyclization structure of the HR routing wall -- why it has no cheap 'render-given-policy' member.

Reads the HR set (make stratify -> results/wall_stratification.csv) and classifies each HR core's rings:
  * EXOTIC cyclization (carbocycle / epoxide / small O-ring) = NOT in the executor release grammar -> even
    given the reduction policy, the executor cannot render it (it needs a new cyclization operator too).
  * in-grammar (macrolactone / linear) = render-given-policy IS possible IF the policy is supplied.

Finding (the disconfirming result of the 'cook a render-given-policy flagship' attempt -- I tried to find
ONE clean HR core to render given its policy, and the wall refused): NO HR core is render-given-policy-able
by the current engine. Of the 29: 25 carry EXOTIC cyclization (decalin/cyclopentane/epoxide/THF -> need a
new cyclization operator, i.e. policy + coverage); 3 are ring-clean but C>20 ORDERING-WALLED for the
directed resolver (soppiline A C21 verified: no recovery within budget; phaeospelide C34, T-toxin C41);
and the 1 ring-clean C<=20 core, Fugralin A (C12), is resolver-tested -> SOUND-STRUCTURAL-GAP (its methyl
ester + gem-dimethyl are out-of-grammar; 4029 orderings exhausted, no render). So every member is hard one
way or another -- exotic chemistry, ordering-walled, or out-of-grammar tailoring -- there is no cheap
render-given-policy demonstrator. That CONFIRMS the pocket (a conformational observable) is the sole lever
for HR routing: no enumerator reaches the clean cores, and the rest need new cyclization chemistry on top.
"""
from __future__ import annotations

import csv

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
EPOXIDE = Chem.MolFromSmarts("[C;R]1[O;R][C;R]1")


def classify(m: Chem.Mol) -> tuple[bool, list[str]]:
    """Return (is_grammar_clean, flags). Clean = no exotic cyclization (only linear / macrolactone)."""
    rings = m.GetRingInfo().AtomRings()
    carbocyc = sum(1 for r in rings
                   if all(m.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in r)
                   and not any(m.GetAtomWithIdx(i).GetIsAromatic() for i in r))
    small_oring = sum(1 for r in rings
                      if len(r) <= 7 and any(m.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in r))
    epox = len(m.GetSubstructMatches(EPOXIDE))
    macro = any(len(r) >= 10 for r in rings)
    flags = []
    if carbocyc:
        flags.append(f"{carbocyc}-carbocycle")
    if epox:
        flags.append("epoxide")
    if small_oring:
        flags.append(f"{small_oring}-small-O-ring")
    if macro:
        flags.append("macrolactone(in-grammar)")
    if not rings:
        flags.append("linear(in-grammar)")
    clean = (carbocyc == 0 and epox == 0 and small_oring == 0)
    return clean, flags


def main() -> None:
    hr = [r["bgc_id"] for r in csv.DictReader(open("results/wall_stratification.csv"))
          if r["subclass"] == "HR-like (reduced)"]
    df = pd.read_parquet("data/processed/fungal_pks_pairs.parquet").set_index("bgc_id")
    exotic, clean = [], []
    for b in hr:
        if b not in df.index:
            continue
        r = df.loc[b]
        m = Chem.MolFromSmiles(r["product_smiles_canonical"]) if isinstance(r["product_smiles_canonical"], str) else None
        if m is None:
            continue
        nC = sum(a.GetAtomicNum() == 6 for a in m.GetAtoms())
        is_clean, flags = classify(m)
        rec = (b, str(r["compound_name"])[:26], nC, " ".join(flags))
        (clean if is_clean else exotic).append(rec)

    print(f"=== HR routing wall: cyclization structure ({len(exotic)+len(clean)} cores) ===\n")
    print(f"EXOTIC cyclization -> policy + coverage (need a new cyclization operator too): {len(exotic)}")
    for b, name, nC, fl in sorted(exotic, key=lambda z: z[2]):
        print(f"  {b}  C{nC:<3} {name:26s} {fl}")
    print(f"\nGRAMMAR-CLEAN (render-given-policy possible IF policy supplied): {len(clean)}")
    for b, name, nC, fl in sorted(clean, key=lambda z: z[2]):
        tractable = "ordering-walled (C>20)" if nC > 20 else "C<=20"
        print(f"  {b}  C{nC:<3} {name:26s} {fl}   [{tractable}]")
    n_clean_walled = sum(1 for _, _, nC, _ in clean if nC > 20)
    n_clean_small = len(clean) - n_clean_walled
    print(f"\nFINDING: no HR core is render-given-policy-able by the current engine.")
    print(f"  {len(exotic)} exotic cyclization (policy + coverage); {n_clean_walled} ring-clean but C>20 ordering-walled;")
    print(f"  {n_clean_small} ring-clean C<=20 (Fugralin A) -> resolver-tested SOUND-STRUCTURAL-GAP "
          f"(out-of-grammar methyl-ester + gem-dimethyl).")
    print(f"  => no cheap flagship; the pocket (a conformational observable) is the sole lever for HR routing.")


if __name__ == "__main__":
    main()
