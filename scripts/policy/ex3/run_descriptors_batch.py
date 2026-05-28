"""Batch descriptor extraction over all docked poses (Phase C entry point).

Walks every cycle manifest, looks up the docked complex PDB, runs
``scripts/policy/descriptors.extract`` against it, and writes one row per cycle to
``data/policy/descriptors.parquet``.

The substrate-only proxies (3-HB / hexanoic / octanoic) get NaN feature values plus an
``arm`` column = "substrate_only" so the Phase C harness can stratify cleanly.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "policy"))
import descriptors  # noqa: E402  -- script-style import

DOCK_DIR = ROOT / "data" / "policy" / "docking"
POSE_DIR = ROOT / "data" / "policy" / "poses"
INTERMEDIATES = ROOT / "data" / "policy" / "intermediates.parquet"
OUT = ROOT / "data" / "policy" / "descriptors.parquet"


def main() -> None:
    rows = []
    for manifest_path in sorted(DOCK_DIR.glob("*/cycle_*/manifest.json")):
        cycle_t = int(manifest_path.parent.name.replace("cycle_", ""))
        m = json.loads(manifest_path.read_text())

        # Conformational arm: protein assigned + ACP-Ser passed pLDDT floor +
        # deterministic placement produced a placed.sdf.
        protein_pdb = m.get("protein_pdb_path")
        placed_sdf = m.get("placed_sdf_path")
        if not protein_pdb or not placed_sdf:
            rows.append(dict(
                synthase=m["synthase"], cycle_t=cycle_t, arm="substrate_only",
                source_pdb="", source_sdf="",
                **{f: float("nan") for f in (
                    "pocket_contact_count_5A", "pocket_contact_residues",
                    "pocket_polar_contacts", "pocket_hydrophobic_contacts",
                    "ligand_sasa_in_complex", "ligand_sasa_free",
                    "ligand_sasa_buried_fraction", "pocket_volume_A3",
                    "ligand_centroid_x", "ligand_centroid_y", "ligand_centroid_z")}))
            continue
        try:
            d = descriptors.extract(ROOT / protein_pdb, ROOT / placed_sdf)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(synthase=m["synthase"], cycle_t=cycle_t, arm="placement_error",
                             source_pdb=protein_pdb, source_sdf=placed_sdf,
                             error=repr(exc)))
            continue
        row = dict(synthase=m["synthase"], cycle_t=cycle_t, arm="conformational",
                   source_pdb=protein_pdb, source_sdf=placed_sdf)
        d_dict = asdict(d)
        cx, cy, cz = d_dict.pop("ligand_centroid_xyz")
        d_dict.pop("catalytic_distances", None)
        row.update(d_dict)
        row.update(dict(ligand_centroid_x=cx, ligand_centroid_y=cy, ligand_centroid_z=cz))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} -- {len(df)} rows")
    print(df.groupby("arm").size())
    print("\nConformational-arm descriptor summary:")
    conf = df[df.arm == "conformational"]
    if len(conf):
        print(conf[[
            "synthase", "cycle_t",
            "pocket_contact_count_5A", "ligand_sasa_buried_fraction",
            "pocket_volume_A3",
        ]].to_string(index=False, max_rows=30))


if __name__ == "__main__":
    main()
