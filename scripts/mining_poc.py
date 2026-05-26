"""Genome-mining generator PoC: recall@K of the true core on the curated set, by subclass.

For each curated fungal program we (a) build the catalytic alphabet its subclass permits,
(b) bound the chain length from the product's carbon count (the proxy for an MS-measured mass
in real mining -- we do NOT use the known program), (c) enumerate + rank candidate cores, and
(d) record where the true structure lands. Aggregated by subclass, this tests the two-walls
prediction: NR/PR alphabets yield tiny candidate sets with the truth near the top; HR alphabets
yield combinatorial sets (the per-cycle-reduction grammar wall, made operational).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median

from rdkit import Chem, RDLogger

from lpi.chem.program import ReductionState as Rs, Release
from lpi.data.curated import load_all
from lpi.search.generate import Alphabet, generate, rank_of

RDLogger.DisableLog("rdApp.*")

ALPHA = {
    "NR": Alphabet(reductions=(Rs.KETO,),
                   releases=(Release.ALDOL_AROMATIC, Release.LACTONIZATION,
                             Release.PT_NAPHTHALENE, Release.HYDROLYSIS),
                   starters=("acetyl",), allow_cmet=True),
    "PR": Alphabet(reductions=(Rs.KETO, Rs.KR),
                   releases=(Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN,
                             Release.LACTONIZATION, Release.HYDROLYSIS),
                   starters=("acetyl",), allow_cmet=False),
    "HR": Alphabet(reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
                   releases=(Release.HYDROLYSIS, Release.LACTONIZATION),
                   starters=("acetyl", "hexanoyl"), allow_cmet=True),
}


def main():
    rows = []
    for e in load_all():
        sub = e.subclass if e.subclass in ALPHA else None
        smi = e.expected_smiles
        if sub is None or not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        nC = sum(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        estN = max(1, (nC - 2) // 2)  # chain-length band from carbon count (MS-mass proxy)
        res = generate(ALPHA[sub], max(1, estN - 1), estN + 1)
        r = rank_of(res, smi)
        rows.append((sub, e.name[:30], len(res.candidates), r, res.capped))

    print(f"{'sub':3s} {'entry':30s} {'#cand':>6s} {'rank':>6s}")
    for sub, name, nc, r, capped in sorted(rows):
        print(f"{sub:3s} {name:30s} {nc:6d} {('MISS' if r is None else str(r)):>6s}"
              f"{'  (capped)' if capped else ''}")

    print("\naggregate (recall@K = true core in top-K):")
    fam = {"NR/PR aromatic": ("NR", "PR"), "HR reduced": ("HR",)}
    for label, subs in fam.items():
        grp = [r for r in rows if r[0] in subs]
        if not grp:
            continue
        n = len(grp)
        for K in (1, 5, 10):
            hit = sum(1 for *_x, rank, _c in [(g[0], g[1], g[2], g[3], g[4]) for g in grp]
                      if rank is not None and rank <= K)
            print(f"  {label:16s} recall@{K}: {hit}/{n}", end="   ")
        med = median([g[2] for g in grp])
        print(f"| median #candidates: {med:.0f}")


if __name__ == "__main__":
    main()
