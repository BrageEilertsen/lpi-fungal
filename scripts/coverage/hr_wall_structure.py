"""Cyclization structure of the HR routing wall -- why it has no cheap 'render-given-policy' member.

Reads the HR set (make stratify -> results/wall_stratification.csv) and classifies each HR core's rings:
  * EXOTIC cyclization (carbocycle / epoxide / small O-ring) = NOT in the executor release grammar -> even
    given the reduction policy, the executor cannot render it (it needs a new cyclization operator too).
  * in-grammar (macrolactone / linear) = render-given-policy IS possible IF the policy is supplied.

Finding (the disconfirming result of the 'cook a render-given-policy flagship' attempt -- I tried to find
ONE CHEAP clean HR core to render given its policy; the wall refused a cheap one). PRECISE claims only:
  * 25/29 carry EXOTIC cyclization (decalin/cyclopentane/epoxide/THF) NOT in the release grammar -> the
    pocket is NECESSARY but NOT SUFFICIENT for the full HR molecule: cracking the reduction policy yields the
    program, but RENDERING still needs cyclization/tailoring coverage for most. [load-bearing, confirmed]
  * Fugralin A (C12, ring-clean): EXHAUSTED at 4029 orderings -> confirmed SOUND coverage gap (methyl-ester
    + gem-dimethyl out-of-grammar). [definitive]
  * soppiline A (C21, ring-clean): the directed resolver SEARCHES for the program by enumerating orderings
    (this is NOT render-given-policy); it was capped/killed with no early recovery -> INCONCLUSIVE, ordering-
    wall vs coverage-gap UNDISTINGUISHED (the C>20 ambiguity). phaeospelide C34 / T-toxin C41: untried.
  * The resolver tests SEARCHABILITY, not render-given-policy. Executing a SUPPLIED policy on the clean cores
    is UNTESTED -> the dynamics claim ('given the policy, the executor renders it') is untested there, NOT
    refuted. The deferred render-given-policy flagship (hand-supply one clean core's policy + execute) is the
    DEMO + the DISAMBIGUATOR: a render confirms ordering-wall AND render-given-policy at once; a no-render
    exposes a coverage gap even given the policy.
CONFIRMED: no cheap SEARCHABLE HR flagship; and the pocket is the sole lever for the reduction POLICY
(no enumerator finds the clean cores' programs cheaply) -- necessary, not the whole molecule.
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
    print(f"\nFINDING: no cheap SEARCHABLE HR flagship (render-given-policy itself is UNTESTED on the clean cores).")
    print(f"  {len(exotic)} exotic cyclization -> pocket NECESSARY but NOT SUFFICIENT (need cyclization/tailoring coverage too);")
    print(f"  {n_clean_walled} ring-clean C>20 -> resolver SEARCH inconclusive (soppiline capped, NOT exhausted; "
          f"ordering-wall vs coverage-gap undistinguished);")
    print(f"  {n_clean_small} ring-clean C<=20 (Fugralin A) -> EXHAUSTED 4029 -> confirmed sound coverage-gap (out-of-grammar).")
    print(f"  => clean cores' render-given-policy is the deferred flagship's job (demo + disambiguator); "
          f"pocket = sole lever for the reduction policy.")


if __name__ == "__main__":
    main()
