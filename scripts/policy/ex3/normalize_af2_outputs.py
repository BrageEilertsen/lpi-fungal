"""Normalize ColabFold's verbose output tree -> data/policy/structures/{slug}/apo.pdb.

ColabFold writes a tree like:
  data/policy/structures/{slug}/
    {slug}_kr_acp_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
    {slug}_kr_acp_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb   (if amber)
    {slug}_kr_acp_scores_rank_001_alphafold2_ptm_model_1_seed_000.json
    log.txt
    ...

We pick the rank-1 relaxed PDB (or unrelaxed if no amber) as ``apo.pdb`` and record
its pLDDT to the kr_index. Run after the SLURM job finishes; idempotent.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STRUCT_DIR = ROOT / "data" / "policy" / "structures"
KR_INDEX = ROOT / "data" / "policy" / "sequences" / "_kr_index.parquet"


def pick_apo(slug_dir: Path) -> tuple[Path, dict] | None:
    """Find the rank-1 prediction in ``slug_dir``.

    ColabFold's naming depends on --num-models:
      multi-model: ``..._rank_001_..._{relaxed,unrelaxed}_..._model_{N}_seed_{S}.pdb``
      single-model: ``..._{relaxed,unrelaxed}_..._model_1_seed_000.pdb`` (no rank prefix)
    Prefer relaxed when present, fall back to unrelaxed.
    """
    # Multi-model: rank_001 prefix.
    rank1 = sorted(slug_dir.glob("*rank_001*relaxed*.pdb"))
    if not rank1:
        rank1 = sorted(slug_dir.glob("*rank_001*unrelaxed*.pdb"))
    # Single-model: no rank prefix. Prefer relaxed over unrelaxed (must filter to
    # avoid matching "unrelaxed" when looking for "relaxed").
    if not rank1:
        relaxed_only = [p for p in slug_dir.glob("*relaxed*.pdb")
                        if "unrelaxed" not in p.name]
        rank1 = sorted(relaxed_only)
    if not rank1:
        rank1 = sorted(slug_dir.glob("*unrelaxed*.pdb"))
    if not rank1:
        return None
    pdb = rank1[0]
    score_pat = re.sub(r"(relaxed|unrelaxed)", "scores", pdb.stem)
    score = slug_dir / f"{score_pat}.json"
    scores: dict = {}
    if score.exists():
        with score.open() as fh:
            scores = json.load(fh)
    return pdb, scores


def main() -> None:
    if not STRUCT_DIR.exists():
        raise SystemExit(f"{STRUCT_DIR} missing -- did AF2 finish?")
    rows = []
    for slug_dir in sorted(STRUCT_DIR.iterdir()):
        if not slug_dir.is_dir():
            continue
        slug = slug_dir.name
        picked = pick_apo(slug_dir)
        if picked is None:
            rows.append(dict(slug=slug, apo_pdb="", plddt_mean=0.0, status="NO_PDB"))
            continue
        pdb, scores = picked
        out = slug_dir / "apo.pdb"
        shutil.copy2(pdb, out)
        plddt = scores.get("plddt") or []
        plddt_mean = float(sum(plddt) / len(plddt)) if plddt else 0.0
        rows.append(dict(slug=slug, apo_pdb=str(out.relative_to(ROOT)),
                         plddt_mean=plddt_mean, status="OK"))

    df = pd.DataFrame(rows)
    out_idx = STRUCT_DIR / "_structures_index.parquet"
    df.to_parquet(out_idx, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out_idx.relative_to(ROOT)}")
    if (df.status == "OK").all():
        print("\nALL APO STRUCTURES READY.\n"
              "Next: python scripts/policy/ex3/identify_acp_serine.py")


if __name__ == "__main__":
    main()
