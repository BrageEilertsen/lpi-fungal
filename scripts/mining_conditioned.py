"""Does conditioning the generator on domain-OBSERVABLE features fix ranking for free?

The release/cyclization class (aromatizing cyclase vs thioesterase vs lactonizing) and the
reductive-domain content are read off a cluster's domains -- no hidden program needed. We
compare two alphabets per curated entry:
  * coarse     -- subclass reductions + all plausible releases (the earlier PoC);
  * conditioned -- subclass reductions + the release GROUP the cluster's domains imply
                   (aromatic / lactone / linear) + the observed starter.
Both use only domain-observable inputs (no program leakage). Prediction (two-walls): the
aromatic NR/PR bulk becomes high-recall once the release group is known, while HR stays low
even fully conditioned -- because HR's wall is per-cycle reduction combinatorics, not release.
"""
from __future__ import annotations

from statistics import median

from rdkit import Chem, RDLogger

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
ALL_REL = AROMATIC + LACTONE + LINEAR


def release_group(rel: Release):
    if rel in AROMATIC:
        return AROMATIC
    if rel in LACTONE:
        return LACTONE
    return LINEAR


def recall_table(rows, label):
    print(f"\n  {label}")
    fam = {"NR/PR aromatic": ("NR", "PR"), "HR reduced": ("HR",)}
    for name, subs in fam.items():
        grp = [r for r in rows if r[0] in subs]
        if not grp:
            continue
        n = len(grp)
        ks = "  ".join(f"@{K}:{sum(1 for g in grp if g[3] and g[3] <= K)}/{n}"
                       for K in (1, 5, 10))
        print(f"    {name:16s} {ks}   median #cand {median([g[2] for g in grp]):.0f}")


def main():
    coarse, cond = [], []
    for e in load_all():
        sub = e.subclass if e.subclass in REDUCTIONS else None
        smi = e.expected_smiles
        if sub is None or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        nC = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        estN = max(1, (nC - 2) // 2)
        lo, hi = max(1, estN - 1), estN + 1
        st = (e.program.starter,)
        a_coarse = Alphabet(REDUCTIONS[sub], ALL_REL, st, CMET[sub])
        a_cond = Alphabet(REDUCTIONS[sub], release_group(e.program.release), st, CMET[sub])
        rc = generate(a_coarse, lo, hi)
        rk = generate(a_cond, lo, hi)
        coarse.append((sub, e.name[:24], len(rc.candidates), rank_of(rc, smi)))
        cond.append((sub, e.name[:24], len(rk.candidates), rank_of(rk, smi)))

    print(f"{'sub':3s} {'entry':24s} {'coarse(#,rank)':>16s} {'conditioned(#,rank)':>20s}")
    for c, k in zip(sorted(coarse), sorted(cond)):
        print(f"{c[0]:3s} {c[1]:24s} {str((c[2], c[3])):>16s} {str((k[2], k[3])):>20s}")
    recall_table(coarse, "COARSE (all releases):")
    recall_table(cond, "CONDITIONED (domain-implied release group):")


if __name__ == "__main__":
    main()
