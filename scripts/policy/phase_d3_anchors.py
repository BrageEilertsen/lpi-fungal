"""Phase D3 ACP-Ser identification: extended over all 16 protein folds (6 Phase B + 10
new D3 didomain folds + 1 D3 full-length CitS).

Reuses the same motif scan as scripts/policy/ex3/identify_acp_serine.py:
  - Narrow:  G[A-Z]DS[LIVMA] / [GA][A-Z]DS  (canonical Type I PKS / FAS)
  - Broad:   [FYWAGILV]DS[LIVMA]            (covers PKS-NRPS / divergent variants)

For each fold's apo.pdb:
  1. Read the protein sequence from the corresponding _kr_acp.fasta input.
  2. Scan for ACP-Ser motifs in the C-terminal half (where ACP sits after KR).
  3. For each candidate Ser, check OG atom presence + pLDDT (B-factor) in the AF2 PDB.
  4. Pick the highest-pLDDT candidate; reject anchors below ACP_PLDDT_FLOOR=50.

Output: data/policy/structures/_d3_acp_serine.parquet -- one row per fold with
(slug, picked_resid, plddt, motif, tier, status). Phase B's 6 folds are RE-identified
to keep a single unified anchor table for Phase D3 downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[2]
SEQ_DIR = ROOT / "data" / "policy" / "sequences"
STRUCT_DIR = ROOT / "data" / "policy" / "structures"
OUT_PARQUET = STRUCT_DIR / "_d3_acp_serine.parquet"

NARROW_MOTIFS = [re.compile(r"G[A-Z]DS[LIVMA]"), re.compile(r"[GA][A-Z]DS")]
BROAD_MOTIFS = [re.compile(r"[FYWAGILV]DS[LIVMA]")]
ACP_PLDDT_FLOOR = 50.0


@dataclass
class AnchorResult:
    slug: str
    apo_pdb: str
    full_length: int
    picked_resid: int | None
    plddt: float
    motif: str
    tier: str
    n_candidates: int
    status: str


def load_fasta_seq(fasta_path: Path) -> str:
    text = fasta_path.read_text().strip().splitlines()
    return "".join(line.strip() for line in text[1:])


def find_acp_serine_candidates(seq: str) -> list[tuple[int, str, str]]:
    """Return (1-based S position, motif text, tier='narrow'|'broad'). C-terminal
    half preferred; falls back to full sequence if no C-terminal hit."""
    cands: list[tuple[int, str, str]] = []
    half = len(seq) // 2
    for tier_name, motifs in (("narrow", NARROW_MOTIFS), ("broad", BROAD_MOTIFS)):
        for motif in motifs:
            for m in motif.finditer(seq):
                s_offset = m.group().index("S")
                pos1 = m.start() + s_offset + 1
                cands.append((pos1, m.group(), tier_name))
    # Dedupe by S-position; narrow beats broad at the same position.
    by_pos: dict[int, tuple[int, str, str]] = {}
    for pos, mtxt, tier in cands:
        if pos not in by_pos or (by_pos[pos][2] == "broad" and tier == "narrow"):
            by_pos[pos] = (pos, mtxt, tier)
    deduped = sorted(by_pos.values())
    c_term = [c for c in deduped if c[0] >= half]
    return c_term or deduped


def og_at_resid(pdb_path: Path, resid: int) -> tuple[np.ndarray, float] | None:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    for atom in struct.get_atoms():
        if atom.get_parent().id[1] == resid and atom.name == "OG":
            return np.array(atom.coord), float(atom.bfactor)
    return None


def identify_anchor(fasta_path: Path, pdb_path: Path) -> AnchorResult:
    slug = fasta_path.stem.replace("_kr_acp", "")
    if not pdb_path.exists():
        return AnchorResult(slug, "", 0, None, 0.0, "", "", 0, "NO_PDB")
    seq = load_fasta_seq(fasta_path)
    cands = find_acp_serine_candidates(seq)
    if not cands:
        return AnchorResult(slug, str(pdb_path.relative_to(ROOT)), len(seq),
                            None, 0.0, "", "", 0, "NO_MOTIF")
    best: tuple[int, float, str, str] | None = None
    for pos, mtxt, tier in cands:
        info = og_at_resid(pdb_path, pos)
        if info is None:
            continue
        _, b = info
        narrow_pref = 1 if tier == "narrow" else 0
        score = (b, narrow_pref)
        if best is None or score > (best[1], 1 if best[3] == "narrow" else 0):
            best = (pos, b, mtxt, tier)
    if best is None:
        return AnchorResult(slug, str(pdb_path.relative_to(ROOT)), len(seq),
                            None, 0.0, "", "", len(cands), "NO_OG_FOUND")
    pos, b, mtxt, tier = best
    status = "OK" if b >= ACP_PLDDT_FLOOR else "LOW_PLDDT_EXCLUDED"
    return AnchorResult(slug, str(pdb_path.relative_to(ROOT)), len(seq),
                        pos, b, mtxt, tier, len(cands), status)


def main() -> None:
    rows: list[AnchorResult] = []
    for fasta in sorted(SEQ_DIR.glob("*_kr_acp.fasta")):
        slug = fasta.stem.replace("_kr_acp", "")
        pdb_path = STRUCT_DIR / slug / "apo.pdb"
        rows.append(identify_anchor(fasta, pdb_path))

    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET.relative_to(ROOT)} -- {len(df)} folds")
    print()
    print(df[["slug", "full_length", "picked_resid", "plddt", "motif", "tier",
              "n_candidates", "status"]].to_string(index=False, max_colwidth=40))

    n_ok = int((df.status == "OK").sum())
    n_excluded = int((df.status == "LOW_PLDDT_EXCLUDED").sum())
    n_failed = int(((df.status == "NO_MOTIF") | (df.status == "NO_PDB")
                    | (df.status == "NO_OG_FOUND")).sum())
    print(f"\nsummary: {n_ok} OK / {n_excluded} LOW_PLDDT_EXCLUDED / {n_failed} FAILED")


if __name__ == "__main__":
    main()
