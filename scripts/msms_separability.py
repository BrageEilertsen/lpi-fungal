"""Try #1: can MS/MS fragmentation disambiguate the formula-isomers the mass cannot?

For the C12 cores where (generic alphabet + MS formula) leaves many isomers (olivetolic, BAB),
regenerate the formula-matched candidate set and compute a CRUDE single-bond-cleavage fragment-
mass fingerprint per isomer -- a rough proxy for CID MS/MS (no real spectra, no CFM-ID). If the
true isomer's fragment-fingerprint group is small, MS/MS would collapse the set and the pipeline
(alphabet -> mass -> MS/MS) completes. Honest scope: crude fragmenter, a separability lower bound,
NOT a validated spectral match. If isomers are fragmentation-degenerate, MS/MS is a dead end too.
"""
from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger

from lpi.chem.program import ReductionState as Rs, Release
from lpi.search.beam import _formula_cho
from lpi.search.generate import Alphabet, generate

RDLogger.DisableLog("rdApp.*")

GENERIC = Alphabet(
    (Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
    (Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
     Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE),
    ("acetyl", "propionyl", "hexanoyl"), True)
_PT = Chem.GetPeriodicTable()


def frag_fingerprint(smi: str) -> frozenset:
    """Multiset (as a set of rounded masses) of fragment masses from breaking each acyclic
    single bond; original per-atom H counts are kept (ignores H-transfer -- crude but consistent)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return frozenset()
    amass = {a.GetIdx(): _PT.GetAtomicWeight(a.GetAtomicNum()) + a.GetTotalNumHs() * 1.008
             for a in mol.GetAtoms()}
    out = set()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        em = Chem.RWMol(mol)
        em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for frag in Chem.GetMolFrags(em.GetMol()):
            out.add(round(sum(amass[i] for i in frag), 1))
    return frozenset(out)


def flat(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def analyze(name: str, smi: str):
    m = Chem.MolFromSmiles(smi)
    nC = sum(a.GetAtomicNum() == 6 for a in m.GetAtoms())
    estN = max(1, (nC - 2) // 2)
    res = generate(GENERIC, max(1, estN - 1), estN + 1, program_cap=60000, target_cho=_formula_cho(m))
    cands = [c.smiles for c in res.candidates]
    fps = {s: frag_fingerprint(s) for s in cands}
    tflat = flat(smi)
    true_s = next((s for s in cands if flat(s) == tflat), None)
    distinct = len(set(fps.values()))
    if true_s is None:
        print(f"  {name:20s}: true core not in candidate set (n={len(cands)})")
        return
    group = [s for s in cands if fps[s] == fps[true_s]]
    print(f"  {name:20s}: {len(cands):3d} formula-isomers -> {distinct:3d} distinct MS/MS "
          f"fingerprints; true isomer's group = {len(group)}  "
          f"(MS/MS narrows {len(cands)} -> {len(group)})")


def main():
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet").set_index("bgc_id")
    print("MS/MS separability of formula-isomers (crude single-cleavage fingerprint):")
    for bgc in ["BGC0002847", "BGC0002240"]:
        analyze(str(pairs.loc[bgc, "compound_name"]), pairs.loc[bgc, "product_smiles_canonical"])


if __name__ == "__main__":
    main()
