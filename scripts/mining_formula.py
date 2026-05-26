"""The realistic mining scenario: cluster alphabet + an MS-measured molecular formula.

In metabologenomics you pair a BGC with an untargeted MS feature whose accurate mass gives a
molecular formula. The formula pins the reduction COUNT (KR/ER add H2, DH removes H2O), which
is the combinatorial axis that blows up the candidate set -- so conditioning generation on the
observed formula should collapse it. Both inputs (domain alphabet, MS formula) are observable;
no program leakage. Two-walls prediction refined: formula-conditioning should make NR/PR
near-unique, while HR keeps residual ambiguity (different reduction PATTERNS can share a
formula) -- the grammar wall surviving even the mass constraint.
"""
from __future__ import annotations

from statistics import median

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from lpi.chem.program import ReductionState as Rs, Release
from lpi.data.curated import load_all
from lpi.search.generate import Alphabet, generate, rank_of

RDLogger.DisableLog("rdApp.*")

REDUCTIONS = {"NR": (Rs.KETO,), "PR": (Rs.KETO, Rs.KR),
              "HR": (Rs.KETO, Rs.KR, Rs.DH, Rs.ER)}
CMET = {"NR": True, "PR": False, "HR": True}
AROMATIC = (Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE)
LACTONE = (Release.LACTONIZATION,)
LINEAR = (Release.HYDROLYSIS,)


def group(rel):
    return AROMATIC if rel in AROMATIC else (LACTONE if rel in LACTONE else LINEAR)


def formula(smi):
    m = Chem.MolFromSmiles(smi)
    return rdMolDescriptors.CalcMolFormula(m) if m else None


def main():
    rows = []
    for e in load_all():
        sub = e.subclass if e.subclass in REDUCTIONS else None
        smi = e.expected_smiles
        if sub is None or not smi:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        nC = sum(a.GetAtomicNum() == 6 for a in m.GetAtoms())
        estN = max(1, (nC - 2) // 2)
        alpha = Alphabet(REDUCTIONS[sub], group(e.program.release),
                         (e.program.starter,), CMET[sub])
        res = generate(alpha, max(1, estN - 1), estN + 1)
        tgt_f = formula(smi)
        # condition on the MS-observable molecular formula
        kept = [c for c in res.candidates if formula(c.smiles) == tgt_f]
        from lpi.search.generate import GenResult
        rank = rank_of(GenResult(kept), smi)
        rows.append((sub, e.name[:24], len(res.candidates), len(kept), rank))

    print(f"{'sub':3s} {'entry':24s} {'#all':>6s} {'#formula':>9s} {'rank':>6s}")
    for sub, name, na, nk, r in sorted(rows):
        print(f"{sub:3s} {name:24s} {na:6d} {nk:9d} {('MISS' if r is None else str(r)):>6s}")

    print("\n  recall@K with cluster-alphabet + MS-formula conditioning:")
    fam = {"NR/PR aromatic": ("NR", "PR"), "HR reduced": ("HR",)}
    for name, subs in fam.items():
        grp = [r for r in rows if r[0] in subs]
        if not grp:
            continue
        n = len(grp)
        ks = "  ".join(f"@{K}:{sum(1 for g in grp if g[4] and g[4] <= K)}/{n}"
                       for K in (1, 3, 5))
        print(f"    {name:16s} {ks}   median #formula-cand {median([g[3] for g in grp]):.0f}")


if __name__ == "__main__":
    main()
