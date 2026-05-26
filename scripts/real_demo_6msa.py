"""Real Mining Demo v1 -- 6-methylsalicylic acid (6-MSA), the first outside-world case.

A blind, end-to-end run of the inverse compiler on a REAL fungal cluster with REAL data:

  cluster   : MIBiG BGC0001275 (6-MSA synthase, producer *Glarea lozoyensis*)
  grammar   : the literature 6-MSAS domain architecture (KS-AT-DH-KR-ACP) -- MIBiG 4.0 annotates
              this cluster only as class "PKS" (no per-module domains: the very gap the paper notes),
              so the domain alphabet is taken from the canonical 6-MSAS architecture (see SOURCES).
  observable: the real monoisotopic mass 152.047344 (MIBiG/PubChem CID 11279), as the [M-H]- ion.

The engine receives ONLY the domain alphabet and the mass. The product structure is withheld from
inference and revealed only afterwards for scoring. Honesty boundaries (see results/real_demo_6msa_sources.md):
  * the accurate mass is a real, citable database value; the [M-H]- m/z is computed from it;
  * a real EI-MS spectrum (MassBank MSBNK-Fac_Eng_Univ_Tokyo-JP008039) corroborates the molecular
    ion at m/z 152 -- it is a real-spectrum MASS sanity reference, NOT an LC-MS/MS validation rung;
  * no real LC-ESI-MS/MS peak list was sourced in this pass; it is also not needed here, because the
    mass alone already collapses the candidate set to a unique verified core. We do not fabricate
    peaks, so the MS/MS rung is reported as not-applied rather than simulated.
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.engine import contains
from lpi.grammars import PKS
from lpi.observe import ADDUCTS, MSObservables, ion_mz, infer_ms

RDLogger.DisableLog("rdApp.*")

# ---- REAL, documented inputs (sources in results/real_demo_6msa_sources.md) ----------
CLUSTER = "BGC0001275"                 # MIBiG 4.0; producer Glarea lozoyensis (fungal)
DOMAIN_SET = {"KS", "AT", "DH", "KR", "ACP"}   # canonical 6-MSAS architecture (literature)
REAL_MASS = 152.047344116              # neutral monoisotopic, C8H8O3 (MIBiG / PubChem CID 11279)
ADDUCT = "[M-H]-"
PPM = 5.0
CHAIN_BAND = (2, 6)                    # blind chain-length band (a tetraketide is 3 cycles)

# Product structure -- WITHHELD from inference; used only for the post-hoc reveal.
_TRUE_PRODUCT = "CC1=C(C(=CC=C1)O)C(=O)O"     # 6-MSA, from the MIBiG compound record


def main() -> None:
    print("Real Mining Demo v1 -- 6-methylsalicylic acid (MIBiG %s)\n" % CLUSTER)
    mz = ion_mz(REAL_MASS, ADDUCTS[ADDUCT])
    print(f"BLIND INPUT: domains={sorted(DOMAIN_SET)}  observed m/z={mz:.4f} ({ADDUCT}, {PPM:g} ppm)")
    print("            (product structure withheld from the engine)\n")

    alpha = PKS.alphabet_from_domains(DOMAIN_SET)
    res = infer_ms(alpha, MSObservables(observed_mz=mz, adducts=(ADDUCT,), ppm=PPM),
                   PKS, CHAIN_BAND[0], CHAIN_BAND[1])

    L = res.ladder
    print("CANDIDATE COLLAPSE")
    print(f"  genome (domain alphabet)        : {L['genome']}")
    print(f"  + accurate mass / adduct (5 ppm): {L['+mass']}   formulas: {', '.join(res.formulas)}")
    print(f"  + real MS/MS                    : n/a (no real LC-MS/MS sourced; unique at mass already)")
    print(f"\nVERDICT: {res.state.value.upper()}  --  {res.reason}\n")

    # ---- reveal and score (only now do we look at the known product) ----
    truth = M.canonical_smiles(M.mol_from_smiles(_TRUE_PRODUCT))
    rank = contains(res, truth)
    top = res.candidates[0].smiles if res.candidates else None
    print("REVEAL (post-inference)")
    print(f"  known product (MIBiG): {truth}")
    print(f"  engine top candidate : {top}")
    print(f"  rank of true product : {rank}")
    print(f"  match                : {'YES' if rank == 1 else ('in set' if rank else 'NOT FOUND')}")


if __name__ == "__main__":
    main()
