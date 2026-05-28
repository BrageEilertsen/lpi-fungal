"""Per-cycle ACP-tethered intermediate extractor (Phase A1).

Tracing wrapper around the executor's cycle loop. Does NOT modify executor/core.py
(critical code, workflow rule 4). The wrapper re-implements the same operator order
as ``executor.core.run`` -- extend -> optional C-MeT -> KR -> DH -> ER -- and snapshots
the intermediate SMILES at three checkpoints per cycle:

  post_extend   beta-keto thioester (the substrate C-MeT acts on, if present)
  post_cmet     after the optional alpha-methylation (= post_extend if no C-MeT)
  post_reduce   after the full reduction cascade for this cycle

We also compute the TE-hydrolysed linear form of ``post_reduce`` (the running released
chain, useful for chain-length / oxidation-state sanity).

Per-cycle training labels are taken directly from the curated Program. Output goes to
``data/policy/intermediates.parquet``. One row per (synthase, cycle).

Phase A is local-only -- no AF3, no docking yet. This is the substrate side of the
state s_t for the per-cycle policy pi_theta(s_t).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from lpi.chem.program import Cycle, Extender, ReductionState
from lpi.data.curated import CuratedEntry, load_all
from lpi.executor import operators as op
from lpi.executor.core import _apply_reduction  # private helper -- mirrors run() exactly

OUT = Path(__file__).resolve().parents[2] / "data" / "policy" / "intermediates.parquet"


def _smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def _linear_of(mol: Chem.Mol) -> str:
    """TE hydrolysis -> linear carboxylic acid form of the tethered intermediate."""
    return _smiles(op.hydrolyse(mol))


def trace(entry: CuratedEntry) -> list[dict]:
    """Walk one curated program; emit one dict per elongation cycle."""
    mol = op.load_starter(entry.program.starter)
    rows: list[dict] = []
    for t, cycle in enumerate(entry.program.cycles, start=1):
        # Extend (Claisen): + CH2-C(=O)- onto the active thioester.
        methylmalonyl = cycle.extender is Extender.METHYLMALONYL
        mol_post_extend = op.extend(mol, methylmalonyl=methylmalonyl)
        # Optional C-MeT on the alpha-CH2.
        mol_post_cmet = op.c_methylate(mol_post_extend) if cycle.c_methyl else mol_post_extend
        # Reductive cascade up to the declared state.
        mol_post_reduce = _apply_reduction(mol_post_cmet, cycle.reduction)

        rows.append({
            "synthase": entry.name,
            "bgc": entry.bgc,
            "subclass": entry.subclass,
            "starter": entry.program.starter,
            "release": entry.program.release.value,
            "n_cycles": entry.program.n_cycles,
            "cycle_t": t,
            "smiles_post_extend": _smiles(mol_post_extend),
            "smiles_post_cmet": _smiles(mol_post_cmet),
            "smiles_post_reduce": _smiles(mol_post_reduce),
            "smiles_linear_post_cycle": _linear_of(mol_post_reduce),
            "label_extender": cycle.extender.value,
            "label_cmet": bool(cycle.c_methyl),
            "label_reduction": cycle.reduction.value,
            "expected_final_smiles": entry.expected_final_smiles or "",
            "tier": entry.tier,
            "subclass_hr_pr": entry.subclass in ("HR", "PR"),
        })

        # Advance the running mol for the next cycle.
        mol = mol_post_reduce
    return rows


def main() -> None:
    entries = load_all()
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    for e in entries:
        try:
            rows.extend(trace(e))
        except Exception as exc:  # noqa: BLE001 -- report every failure, do not mask
            failures.append((e.name, repr(exc)))

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    n_syn = df["synthase"].nunique()
    n_hr_pr = int(df["subclass_hr_pr"].sum())
    print(f"wrote {OUT} -- {len(df)} rows, {n_syn} synthases")
    print(f"  by subclass: {df.groupby('subclass').size().to_dict()}")
    print(f"  HR+PR cycles (the substrate_state_probe scope): {n_hr_pr}")
    print(f"  reduction-label distribution: {df['label_reduction'].value_counts().to_dict()}")
    print(f"  C-MeT cycles: {int(df['label_cmet'].sum())}/{len(df)}")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
    # Spot check: 6-MSA cycle 2 should have a beta-OH after KR.
    msa = df[(df["synthase"].str.contains("6-methylsalicylic", case=False)) & (df["cycle_t"] == 2)]
    if len(msa):
        row = msa.iloc[0]
        print("\nSpot check -- 6-MSA cycle 2 (the KR cycle):")
        print(f"  post_extend (beta-keto):  {row.smiles_post_extend}")
        print(f"  post_reduce (beta-OH):    {row.smiles_post_reduce}")
        print(f"  linear so far:            {row.smiles_linear_post_cycle}")
        assert "O" in row.smiles_post_reduce, "expected hydroxyl after KR"


if __name__ == "__main__":
    main()
