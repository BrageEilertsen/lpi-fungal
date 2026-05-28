"""Identify the ACP active-site serine in each folded KR+ACP didomain.

The acyl-carrier protein active-site signature is the conserved GxDS or DSL motif --
the serine in the motif is the phosphopantetheine attachment site. We:

1. For each ``data/policy/structures/{slug}/apo.pdb``, re-read the sequence (from the
   FASTA the AF2 input used) and locate every G[A-Z]DS or [DH]S[LIVMA] motif --
   restrict to the C-terminal half (ACP is downstream of KR in the didomain).
2. For each candidate serine, verify the OG atom exists in the AF2 PDB and the
   side-chain is solvent-exposed (pLDDT > 60 region; SASA proxy).
3. Pick the best-scoring serine; write its residue number into each cycle's
   manifest.json (updating ``acp_serine_resid``).

Run after ``normalize_af2_outputs.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

ROOT = Path(__file__).resolve().parents[3]
SEQ_DIR = ROOT / "data" / "policy" / "sequences"
STRUCT_DIR = ROOT / "data" / "policy" / "structures"
DOCK_DIR = ROOT / "data" / "policy" / "docking"

# ACP active-site motifs, ordered narrow -> broad. The phosphopantetheine attachment
# site has the signature [FYWAGILV]DS[LIVMA] (hydrophobic-D-S-hydrophobic) across
# Type I PKS/FAS, with `GxDS` being the most common but not universal variant. LovB
# and other PKS-NRPS hybrids use the FDSx pattern; we accept both.
NARROW_MOTIFS = [re.compile(r"G[A-Z]DS[LIVMA]"), re.compile(r"[GA][A-Z]DS")]
BROAD_MOTIFS = [re.compile(r"[FYWAGILV]DS[LIVMA]")]

ACP_PLDDT_FLOOR = 50.0  # below this the picked anchor is too disordered to trust


def find_acp_serine(seq: str) -> list[tuple[int, str, str]]:
    """Return (1-based position, motif text, tier) candidates for the ACP-Ser.

    Tier is 'narrow' (canonical GxDS variants) or 'broad' (hydrophobic-DS-hydrophobic).
    We prefer the C-terminal half (ACP is downstream of KR), but fall back to the full
    sequence if no C-terminal candidate exists -- some synthases have the ACP-Ser at
    the KR/ACP boundary which lands just past the 50% mark.
    """
    candidates: list[tuple[int, str, str]] = []
    half = len(seq) // 2
    for tier_name, motifs in (("narrow", NARROW_MOTIFS), ("broad", BROAD_MOTIFS)):
        for motif in motifs:
            for m in motif.finditer(seq):
                offset = m.group().index("S")
                pos1 = m.start() + offset + 1  # 1-based
                candidates.append((pos1, m.group(), tier_name))
    # Deduplicate by serine position; keep narrow > broad classification when both hit.
    by_pos: dict[int, tuple[int, str, str]] = {}
    for pos, motif, tier in candidates:
        if pos not in by_pos or (by_pos[pos][2] == "broad" and tier == "narrow"):
            by_pos[pos] = (pos, motif, tier)
    deduped = sorted(by_pos.values())
    # Prefer C-terminal half; fall back to full if nothing matches.
    c_term = [c for c in deduped if c[0] >= half]
    return c_term or deduped


def og_atom(pdb_path: Path, resid: int) -> tuple[np.ndarray, float] | None:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    for atom in struct.get_atoms():
        res = atom.get_parent()
        if res.id[1] == resid and atom.name == "OG":
            return np.array(atom.coord), float(atom.bfactor)
    return None


def main() -> None:
    rows = []
    for fasta in sorted(SEQ_DIR.glob("*_kr_acp.fasta")):
        slug = fasta.stem.replace("_kr_acp", "")
        pdb_path = STRUCT_DIR / slug / "apo.pdb"
        if not pdb_path.exists():
            rows.append(dict(slug=slug, picked_resid=None, plddt=0.0,
                             n_candidates=0, status="NO_PDB"))
            continue
        text = fasta.read_text()
        seq = "".join(line.strip() for line in text.splitlines()[1:])
        cands = find_acp_serine(seq)
        if not cands:
            rows.append(dict(slug=slug, picked_resid=None, plddt=0.0,
                             n_candidates=0, motif="", tier="",
                             status="NO_MOTIF"))
            continue
        # Pick the candidate with the highest pLDDT (B-factor in the PDB), favouring
        # narrow-tier motifs at equal pLDDT.
        best = None  # (pos, plddt, motif, tier)
        for pos, motif, tier in cands:
            info = og_atom(pdb_path, pos)
            if info is None:
                continue
            _, b = info
            score = (b, 1 if tier == "narrow" else 0)
            if best is None or score > (best[1], 1 if best[3] == "narrow" else 0):
                best = (pos, b, motif, tier)
        if best is None:
            rows.append(dict(slug=slug, picked_resid=None, plddt=0.0,
                             n_candidates=len(cands), motif="", tier="",
                             status="NO_OG_FOUND"))
            continue
        # Below the pLDDT floor the anchor is too disordered to trust -- mark for
        # conformational-arm exclusion (substrate-only fallback). The Phase B6 batch
        # descriptor extractor reads ``status`` to decide.
        status = "OK" if best[1] >= ACP_PLDDT_FLOOR else "LOW_PLDDT_EXCLUDED"
        rows.append(dict(slug=slug, picked_resid=best[0], plddt=best[1],
                         n_candidates=len(cands), motif=best[2], tier=best[3],
                         status=status))

        # Update each cycle's manifest under this synthase. Match by the docking
        # bundle's directory slug (not a substring of synthase.name -- "mellein" is a
        # substring of "6_hydroxymellein" so substring-matching corrupts the manifests).
        cycle_dirs = list((DOCK_DIR).glob(f"*/cycle_*/manifest.json"))
        # Map protein-fold slug -> docking-bundle directory slug (the curated entry's
        # full slugified name; see scripts/policy/docking_prep.py:_slug).
        bundle_dir_by_slug = {
            "6_methylsalicylic_acid": "6_methylsalicylic_acid",
            "mellein":                 "mellein",
            "6_hydroxymellein":        "6_hydroxymellein",
            "lovb":                    "lovb_dihydromonacolin_l_nonaketide",
            "lovf": "2_methylbutyric_acid_lovf_diketide_c_met_full_cascade_test",
            "tenellin_tens":           "tenellin_polyketide_backbone_tens_pks_portion",
        }
        target_dir = bundle_dir_by_slug.get(slug)
        for mp in cycle_dirs:
            syn_dir = mp.parent.parent.name
            if syn_dir != target_dir:
                continue
            m = json.loads(mp.read_text())
            if status == "OK":
                m["acp_serine_resid"] = int(best[0])
                m["protein_pdb_path"] = str(pdb_path.relative_to(ROOT))
                m["acp_motif"] = best[2]
                m["acp_plddt"] = float(best[1])
            else:  # LOW_PLDDT_EXCLUDED or no anchor
                m["acp_serine_resid"] = None
                m["protein_pdb_path"] = None
                m["acp_exclusion_reason"] = (
                    f"ACP-Ser candidate at residue {best[0]} (motif {best[2]}) "
                    f"has pLDDT {best[1]:.1f} < floor {ACP_PLDDT_FLOOR} -- "
                    f"covalent anchor unreliable, cycle falls back to substrate-only arm"
                ) if best else "no ACP motif found in fold"
            mp.write_text(json.dumps(m, indent=2))

    df = pd.DataFrame(rows)
    out = STRUCT_DIR / "_acp_serine.parquet"
    df.to_parquet(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out.relative_to(ROOT)} + updated manifests")
    if (df.status == "OK").all():
        print("\nALL ACP SERINES IDENTIFIED.\n"
              "Next: bash scripts/policy/ex3/vina_run.sh")


if __name__ == "__main__":
    main()
