"""Stress-test the BGC+mass reframe off the curated home turf, with the formula PREFILTER.

Take the MIBiG-reachable products (not hand-curated) and a fully GENERIC alphabet (no subclass
hint), and prefilter enumeration on the MS-observable molecular formula (target_cho) so only
on-mass programs are executed. Finding: small cores (<=C10) collapse to a single isomer (rank 1),
but larger cores (olivetolic / BAB, C12) retain many formula-isomers that the parsimony prior
ranks poorly -- the mass ALONE is insufficient for larger cores. Near-unique needs the
domain-informed (tight) alphabet too, and/or a learned ranking prior; both observables matter.
"""
from __future__ import annotations

from statistics import median

import pandas as pd
from rdkit import Chem, RDLogger

from lpi.chem.program import ReductionState as Rs, Release
from lpi.search.beam import _formula_cho
from lpi.search.generate import Alphabet, generate, rank_of

RDLogger.DisableLog("rdApp.*")

GENERIC = Alphabet(
    reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
    releases=(Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
              Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE),
    starters=("acetyl", "propionyl", "hexanoyl"), allow_cmet=True)


def main():
    reach = pd.read_csv("results/reachability_v2.csv")
    pairs = pd.read_parquet("data/processed/fungal_pks_pairs.parquet").set_index("bgc_id")
    rows = []
    for _, r in reach[reach.status == "reachable"].iterrows():
        bgc = r.bgc_id
        if bgc not in pairs.index:
            continue
        smi = pairs.loc[bgc, "product_smiles_canonical"]
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            continue
        nC = sum(a.GetAtomicNum() == 6 for a in m.GetAtoms())
        estN = max(1, (nC - 2) // 2)
        # formula prefilter during enumeration: only on-mass programs are executed
        res = generate(GENERIC, max(1, estN - 1), estN + 1, program_cap=60000,
                       target_cho=_formula_cho(m))
        rk = rank_of(res, smi)
        rows.append((bgc, str(pairs.loc[bgc, "compound_name"])[:22], nC,
                     res.programs_tried, len(res.candidates), rk, res.capped))

    print(f"{'bgc':12s} {'name':22s} {'nC':>3s} {'#exec':>7s} {'#isomers':>8s} {'rank':>5s}")
    for bgc, name, nC, ne, nk, rk, cap in rows:
        print(f"{bgc:12s} {name:22s} {nC:3d} {ne:7d} {nk:8d} "
              f"{('MISS' if rk is None else str(rk)):>5s}{'  CAP' if cap else ''}")
    ranks = [x[5] for x in rows]
    n = len(rows)
    print(f"\nMIBiG-reachable, GENERIC alphabet + MS formula: n={n}")
    print(f"  recall@1={sum(1 for r in ranks if r == 1)}/{n}  "
          f"@3={sum(1 for r in ranks if r and r <= 3)}/{n}  "
          f"median #formula-candidates={median([x[4] for x in rows]):.0f}")


if __name__ == "__main__":
    main()
