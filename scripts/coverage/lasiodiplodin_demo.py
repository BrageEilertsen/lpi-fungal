"""Demonstrator: the first genome -> position-sound DECORATED natural product.

Two stages, both sound, no fabricated chemistry:
  1. CORE (reachable, RECOVERED by the forward index): de-O-methyl-lasiodiplodin, rendered by the executor
     from a resorcylic-macrolactone PKS program (acetyl + KR/3xER/3xketo + resorcylic-macrolactone release).
  2. TAILORING (curation-conditional, POSITION-SOUND): one O-methylation at the curated register
     LASIODIPLODIN_OME -- the resorcinol OH ortho to the macrolactone ester. The register matches exactly
     one of the core's two aromatic OHs, so the position is pinned, not guessed.

The decorated product is checked against deposited lasiodiplodin (BGC0001245) by the SAME flat-canonical ~=
the core map uses. A match is therefore position-sound, not exact-by-budget.

VERDICT typing: "verified RELATIVE TO the methylation register" (curation-conditional, S4.1) -- NOT
observation-determined. Rung tag: rung-3-curated, pending the lasiodiplodin O-MT characterization
literature (substrate-directed -> promote to rung-2; enzyme-gated/silent -> stays rung-3-curated).

    make lasiodiplodin   /   python scripts/coverage/lasiodiplodin_demo.py
"""
from __future__ import annotations

import sys

import pandas as pd
from rdkit import Chem, RDLogger

from lpi.chem.program import Cycle, Program, Release, ReductionState as K
from lpi.data.mibig import PROCESSED
from lpi.executor import core as _core
from lpi.executor.tailoring_positional import LASIODIPLODIN_OME, apply_curated_edit

RDLogger.DisableLog("rdApp.*")

CORE_PROGRAM = Program(starter="acetyl", cycles=(
    Cycle(reduction=K.KR), Cycle(reduction=K.ER), Cycle(reduction=K.ER), Cycle(reduction=K.ER),
    Cycle(reduction=K.KETO), Cycle(reduction=K.KETO), Cycle(reduction=K.KETO),
), release=Release.RESORCYLIC_MACROLACTONE)


def _flat(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, isomericSmiles=False)


def main() -> None:
    core = _core.exec(CORE_PROGRAM)
    product = apply_curated_edit(core, LASIODIPLODIN_OME)

    df = pd.read_parquet(PROCESSED / "fungal_pks_pairs.parquet")
    dep_smi = df[df.bgc_id == LASIODIPLODIN_OME.witness_bgc].iloc[0]["product_smiles_canonical"]
    dep = Chem.MolFromSmiles(dep_smi)
    ok = _flat(product) == _flat(dep)

    print("=== Lasiodiplodin: genome -> position-sound decorated product ===\n")
    print(f"  [1] core  (executor-rendered; RECOVERED by forward index): {_flat(core)}")
    print(f"      program: acetyl + KR/ER/ER/ER/keto/keto/keto + resorcylic-macrolactone release")
    print(f"  [2] +edit ({LASIODIPLODIN_OME.name}, position-sound): {_flat(product)}")
    print(f"      register: {LASIODIPLODIN_OME.register_smarts}  (pins 1 of the core's 2 aromatic OHs)")
    print(f"  deposited lasiodiplodin ({LASIODIPLODIN_OME.witness_bgc}):              {_flat(dep)}\n")
    print(f"  VERDICT: {'VERIFIED relative to the methylation register' if ok else 'MISMATCH'} "
          f"-- curation-conditional, NOT observation-determined")
    print(f"  rung tag: {LASIODIPLODIN_OME.rung}  (pends the O-MT characterization literature: "
          f"substrate-directed -> rung-2; enzyme-gated/silent -> stays rung-3-curated)")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
