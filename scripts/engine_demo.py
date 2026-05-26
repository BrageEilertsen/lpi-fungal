"""Engine v0 demo: the per-cluster reconstructibility ladder on representative cores.

For each core, condition the engine on a cluster-appropriate alphabet and watch the verified
candidate set collapse as observables are added (alphabet -> +MS mass -> +MS/MS), with the
reconstructibility regime the engine assigns. This is the observable-ladder of the paper,
instantiated as a runnable tool.
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.chem.program import ReductionState as R, Release
from lpi.engine import Observables, contains, frag_fingerprint, infer
from lpi.search.beam import _formula_cho
from lpi.search.generate import Alphabet

RDLogger.DisableLog("rdApp.*")

NR = Alphabet((R.KETO,), (Release.ALDOL_AROMATIC, Release.LACTONIZATION, Release.HYDROLYSIS),
              ("acetyl",), True)
PR = Alphabet((R.KETO, R.KR), (Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN,
                               Release.HYDROLYSIS), ("acetyl",), False)
HRAR = Alphabet((R.KETO, R.KR, R.DH, R.ER), (Release.ALDOL_AROMATIC, Release.HYDROLYSIS),
                ("acetyl", "hexanoyl"), True)

CASES = [
    ("orsellinic acid", "Cc1cc(O)cc(O)c1C(=O)O", NR, (3, 3)),
    ("(R)-mellein", "CC1Cc2cccc(O)c2C(=O)O1", PR, (3, 5)),
    ("olivetolic acid", "CCCCCc1cc(O)cc(O)c1C(=O)O", HRAR, (3, 6)),
]


def main():
    print("Verifier-Grounded Biosynthetic Program Engine v0 -- reconstructibility ladder\n")
    print(f"{'core':18s} {'alphabet':>9s} {'+mass':>6s} {'+MS/MS':>7s} {'rank':>5s}  regime")
    for name, smi, alpha, (lo, hi) in CASES:
        cho = _formula_cho(M.mol_from_smiles(smi))
        res = infer(Observables(alpha, lo, hi, target_cho=cho, msms=frag_fingerprint(smi)))
        L = res.ladder
        print(f"{name:18s} {L['alphabet']:9d} {L['alphabet+mass']:6d} {L['alphabet+mass+msms']:7d} "
              f"{str(contains(res, smi)):>5s}  {res.regime}")


if __name__ == "__main__":
    main()
