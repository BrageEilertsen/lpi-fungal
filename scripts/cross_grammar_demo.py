"""Cross-grammar engine demo: one interface, two biosynthetic families.

The architecture-generalization claim, made runnable. The same `engine.infer_cluster(domains,
observables, grammar=...)` call drives both the polyketide grammar (the paper's case study) and a
non-ribosomal peptide grammar. The accurate-mass filter, the MS/MS fragment matcher, the
observable ladder, and the three-state reconstructibility verdict (VERIFIED / UNDER-OBSERVED /
OUT-OF-GRAMMAR) are shared verbatim -- they read only candidate SMILES, never grammar internals.
"""
from __future__ import annotations

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.engine import contains, frag_fingerprint, infer_cluster
from lpi.executor.nrps import formula_chno
from lpi.grammars import NRPS, PKS
from lpi.search.beam import _formula_cho

RDLogger.DisableLog("rdApp.*")

# (label, grammar, cluster domains, product SMILES, formula-key fn, (min,max) steps, give mass?)
CASES = [
    ("orsellinic acid (PKS)", PKS, {"KS", "AT", "ACP", "PT", "TE"},
     "Cc1cc(O)cc(O)c1C(=O)O", _formula_cho, (3, 3), True),
    ("olivetolic acid (PKS)", PKS, {"KS", "AT", "ACP", "KR", "DH", "ER", "PT", "cMT"},
     "CCCCCc1cc(O)cc(O)c1C(=O)O", _formula_cho, (3, 6), True),
    ("cyclo(Gly-Gly) (NRPS)", NRPS, {"C", "A", "T", "TE"},
     "O=C1CNC(=O)CN1", formula_chno, (2, 2), True),
    ("glycylglycine (NRPS)", NRPS, {"C", "A", "T"},
     "OC(=O)CNC(=O)CN", formula_chno, (2, 3), True),
    ("cyclo(Phe-Tyr) (NRPS)", NRPS, {"C", "A", "T", "TE"},
     "O=C1NC(Cc2ccc(O)cc2)C(=O)NC1Cc1ccccc1", formula_chno, (2, 2), True),
    ("NRPS, no mass yet", NRPS, {"C", "A", "T", "TE"},
     "O=C1CNC(=O)CN1", formula_chno, (2, 3), False),  # under-observed: mass withheld
]


def main() -> None:
    print("Verifier-grounded engine -- one interface, two grammars\n")
    hdr = f"{'cluster / product':26s} {'gram':5s} {'alpha':>6s} {'+mass':>6s} {'+MSMS':>6s} {'rank':>5s}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for label, grammar, domains, smi, formula_fn, (lo, hi), give_mass in CASES:
        cho = formula_fn(M.mol_from_smiles(smi)) if give_mass else None
        msms = frag_fingerprint(smi) if give_mass else None
        res = infer_cluster(domains, lo, hi, target_cho=cho, msms=msms, grammar=grammar)
        L = res.ladder
        rank = contains(res, smi)
        verdict = res.state.value.upper()
        if res.next_observable:
            verdict += f"  (next: {res.next_observable})"
        print(f"{label:26s} {grammar.name:5s} {L['alphabet']:6d} {L['alphabet+mass']:6d} "
              f"{L['alphabet+mass+msms']:6d} {str(rank):>5s}  {verdict}")


if __name__ == "__main__":
    main()
