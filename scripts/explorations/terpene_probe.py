"""EXPLORATORY (Direction C): does the program-space abstraction fit terpene synthases?

Primary source (FETCHED): D.W. Christianson, "Structural and Chemical Biology of Terpenoid Cyclases,"
Chem. Rev. 2017, 117(17):11570-11648, doi:10.1021/acs.chemrev.7b00287 (PMID 28841019). Verbatim:
"Terpenoid cyclases catalyze the most complex chemical reactions in biology, in that more than half of
the substrate carbon atoms undergo changes in bonding and hybridization during a single enzyme-catalyzed
cyclization reaction." Class I monoterpene synthases ionize GPP (geranyl diphosphate, C10) to a
carbocation, which cyclizes and rearranges (hydride shifts, methyl/Wagner-Meerwein migrations, cation-pi
stabilization) and terminates by deprotonation or water capture.

This probe asks the architecture question, not the chemistry question: a PKS program is a COMPOSABLE
sequence of discrete operators (extend/reduce/methylate/cyclize) with per-operator mass deltas, which is
why we can enumerate and INFER it. Does a terpene synthase decompose that way? We test the gate case
(GPP -> the C10H16 monoterpenes) with exact-mass bookkeeping and report where the abstraction holds and
where it breaks.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# GPP and the canonical C10 monoterpene products of DIFFERENT class-I monoterpene synthases from the
# SAME substrate (Christianson 2017, sec. on monoterpene cyclization).
GPP = "CC(C)=CCC/C(C)=C/COP(=O)(O)OP(=O)(O)O"      # geranyl diphosphate
PPI = "OP(=O)(O)OP(=O)(O)O"                          # pyrophosphate (the leaving group)
MONOTERPENES = {
    "limonene":    "CC1=CCC(CC1)C(=C)C",
    "alpha-pinene": "CC1=CCC2CC1C2(C)C",
    "beta-pinene":  "C=C1CCC2CC1C2(C)C",
    "terpinolene":  "CC1=CCC(=C(C)C)CC1",
    "myrcene":      "C=CC(=C)CCC=C(C)C",            # acyclic, deprotonation without cyclization
}


def _mol(s):
    return Chem.MolFromSmiles(s)


def _mass(s):
    return Descriptors.ExactMolWt(_mol(s))


def _formula(s):
    return rdMolDescriptors.CalcMolFormula(_mol(s))


def main() -> None:
    print("Direction C -- does program-space fit terpenes? Gate case: GPP -> C10H16 monoterpenes\n")

    # (1) Mass bookkeeping: is the net transformation GPP -> product + PPi balanced (sound at the net level)?
    gpp_m, ppi_m = _mass(GPP), _mass(PPI)
    print(f"GPP   {_formula(GPP):12s} {gpp_m:8.3f}")
    print(f"PPi   {_formula(PPI):12s} {ppi_m:8.3f}\n")
    print(f"{'product':14s} {'formula':10s} {'mass':>8s}  GPP - (product + PPi)  [0 = mass-balanced net rxn]")
    print("-" * 84)
    for name, smi in MONOTERPENES.items():
        pm = _mass(smi)
        residual = gpp_m - (pm + ppi_m)
        print(f"{name:14s} {_formula(smi):10s} {pm:8.3f}  {residual:+.4f}")

    # (2) The abstraction test: are these distinguishable by the observables a sound enumerable grammar
    # would use (formula / mass)? And is there a COMPOSABLE inner program to enumerate/infer?
    formulas = {name: _formula(s) for name, s in MONOTERPENES.items()}
    distinct_formulas = set(formulas.values())
    print(f"\nAll {len(MONOTERPENES)} products from the SAME substrate GPP have formula(s): {distinct_formulas}")
    print("=> they are mass/formula-DEGENERATE: accurate mass and MS/MS-formula cannot distinguish them.")

    print("""
FINDING (architecture, not chemistry):
 * NET mass bookkeeping is sound and trivial: GPP -> product + PPi balances for every monoterpene
   (residual ~0 above) -- so a sound terpene PRODUCT VALIDATOR (per known synthase) is achievable.
 * But there is NO COMPOSABLE PROGRAM to enumerate or infer. In PKS, distinct products come from
   distinct OPERATOR SEQUENCES (a program over discrete, mass-changing steps) -- which is exactly what
   the executor enumerates and the verifier infers. In terpene cyclization, distinct products come from
   the SAME substrate via DIFFERENT carbocation-steering by the enzyme's 3D active-site template
   (Christianson 2017: ">half of carbons change bonding/hybridization in a SINGLE cyclization";
   cyclization position + rearrangement pathway + termination all set by the enzyme, not by composable
   operators). The intermediates are non-isolable carbocations; the "operators" (cyclization, hydride
   shift, Wagner-Meerwein) are mass-NEUTRAL isomerizations with no per-step observable.
 * Consequence: the program-space abstraction COLLAPSES to a 1:1 synthase->product lookup for terpenes.
   There is no inner program to enumerate, so the enumerate/infer machinery (the architecture's value)
   has nothing to operate on; and mass/MS-MS (the observables) cannot disambiguate the formula-degenerate
   isomers. A sound terpene predictor would need a per-synthase STRUCTURAL/templating model (or ML over
   sequence/structure), which is a fundamentally different abstraction from the sound enumerable grammar.
 * VERDICT: program-space fits PKS (compositional) and does NOT fit terpene cyclization (concerted,
   enzyme-templated). This is the finding, reported as such.
""")


if __name__ == "__main__":
    main()
