"""Deterministic substrate placement at the ACP-Ser:OG anchor (Tier-1 alternative to Vina).

For each (synthase, cycle) bundle with a passing ACP-Ser anchor:

  1. Load apo.pdb. Extract ACP-Ser:OG coordinate.
  2. Compute the unit vector from ACP-Ser:OG to the protein centroid (pocket-inward
     direction).
  3. Load ligand.sdf. Identify the sentinel `*` atom (the ACP-linker attachment).
     Compute the ligand's long axis = direction from sentinel to the centroid of all
     other ligand atoms.
  4. Rotate the ligand so its long axis aligns with the pocket-inward direction
     (Rodrigues' formula).
  5. Translate the ligand so the sentinel atom coincides with ACP-Ser:OG.
  6. Write a new SDF (``placed.sdf``) next to ``ligand.sdf`` in each cycle dir, and a
     combined ``complex.pdb`` (protein + ligand as a HETATM block) for inspection.

This is the Tier-1 falsifier placement: same protocol every cycle, no Vina noise. The
per-cycle variation in pocket descriptors then comes purely from chain-length and
oxidation-state differences in the intermediate. If this lifts, Vina hardens it; if it
nulls, Vina (which optimizes binding energy, not per-cycle distinguishability) likely
won't save it -- Tier-3 MD is the next escalation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
DOCK_DIR = ROOT / "data" / "policy" / "docking"


def acp_serine_og(pdb_path: Path, resid: int) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    for atom in struct.get_atoms():
        if atom.get_parent().id[1] == resid and atom.name == "OG":
            return np.array(atom.coord)
    raise RuntimeError(f"no ACP-Ser:OG at residue {resid} in {pdb_path}")


def protein_centroid(pdb_path: Path) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    coords = np.array([a.coord for a in struct.get_atoms()])
    return coords.mean(axis=0)


def rotation_matrix(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rodrigues' formula -- rotation matrix that maps ``v_from`` onto ``v_to``."""
    a = v_from / np.linalg.norm(v_from)
    b = v_to / np.linalg.norm(v_to)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:  # already aligned or anti-parallel
        if c > 0:
            return np.eye(3)
        # anti-parallel: rotate 180 deg around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp -= a * float(np.dot(perp, a))
        perp /= np.linalg.norm(perp)
        K = np.array([
            [0.0, -perp[2], perp[1]],
            [perp[2], 0.0, -perp[0]],
            [-perp[1], perp[0], 0.0],
        ])
        return np.eye(3) + 2 * K @ K
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


def place_ligand(ligand_sdf: Path, og: np.ndarray, protein_com: np.ndarray) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False, sanitize=False)
    mol = Chem.Mol(supplier[0])  # writeable copy
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    sentinel_idx = next(i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() == 0)
    sentinel_xyz = coords[sentinel_idx]
    # Long axis: sentinel -> centroid of all other atoms (the chain extends away
    # from the linker endpoint).
    other = np.delete(coords, sentinel_idx, axis=0)
    other_centroid = other.mean(axis=0)
    v_lig = other_centroid - sentinel_xyz

    v_target = protein_com - og
    if np.linalg.norm(v_target) < 1e-6:
        v_target = np.array([1.0, 0.0, 0.0])  # degenerate fallback

    R = rotation_matrix(v_lig, v_target)
    # Rotate around the sentinel, then translate sentinel to OG.
    new_coords = (coords - sentinel_xyz) @ R.T + og

    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(new_coords[i]))
    return mol


class _ChainAFilter(Select):
    def accept_chain(self, chain):
        return chain.id != "X"  # placeholder; we don't actually filter, this is for hooks


def write_complex(pdb_path: Path, placed_mol: Chem.Mol, out: Path) -> None:
    """Append the placed ligand as a HETATM block to a copy of the protein PDB."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb_path))
    # Write protein with PDBIO, then append HETATM lines manually -- cleaner than
    # wrestling Bio.PDB into accepting a non-standard residue.
    io = PDBIO()
    io.set_structure(struct)
    io.save(str(out))

    conf = placed_mol.GetConformer()
    serial = 50000
    lines = []
    for i, atom in enumerate(placed_mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            element = "X"
            name = "X1"
        else:
            element = atom.GetSymbol()
            name = f"{element}{i}"[:4]
        x, y, z = conf.GetAtomPosition(i)
        lines.append(
            f"HETATM{serial + i:5d} {name:<4s} LIG L   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    # Insert before END
    text = out.read_text()
    text = text.replace("END\n", "") + "\n".join(lines) + "\nEND\n"
    out.write_text(text)


def main() -> None:
    rows = []
    for manifest_path in sorted(DOCK_DIR.glob("*/cycle_*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        syn_dir = manifest_path.parent.parent.name
        cycle_t = int(manifest_path.parent.name.replace("cycle_", ""))

        if not m.get("protein_pdb_path") or m.get("acp_serine_resid") is None:
            rows.append(dict(synthase=syn_dir, cycle_t=cycle_t,
                             status="SKIP_substrate_only",
                             reason=m.get("acp_exclusion_reason", "no protein assigned")))
            continue
        pdb = ROOT / m["protein_pdb_path"]
        og = acp_serine_og(pdb, m["acp_serine_resid"])
        com = protein_centroid(pdb)
        lig_in = ROOT / m["ligand_sdf_path"]
        placed = place_ligand(lig_in, og, com)
        out_sdf = manifest_path.parent / "placed.sdf"
        writer = Chem.SDWriter(str(out_sdf))
        writer.write(placed)
        writer.close()
        out_pdb = manifest_path.parent / "complex.pdb"
        write_complex(pdb, placed, out_pdb)

        # Update manifest with placed pose path so descriptors batch uses it.
        m["placed_sdf_path"] = str(out_sdf.relative_to(ROOT))
        m["complex_pdb_path"] = str(out_pdb.relative_to(ROOT))
        m["placement_method"] = "deterministic_acp_anchor_pocket_axis"
        manifest_path.write_text(json.dumps(m, indent=2))

        rows.append(dict(synthase=syn_dir, cycle_t=cycle_t,
                         status="OK",
                         og_xyz=tuple(float(x) for x in og),
                         com_offset=float(np.linalg.norm(com - og))))

    import pandas as pd
    df = pd.DataFrame(rows)
    out_idx = ROOT / "data" / "policy" / "_placement_index.parquet"
    df.to_parquet(out_idx, index=False)
    n_ok = int((df.status == "OK").sum())
    n_skip = int((df.status == "SKIP_substrate_only").sum())
    print(f"placed {n_ok} substrates; skipped {n_skip} (substrate-only / low-pLDDT ACP)")
    print(df.groupby("status").size())


if __name__ == "__main__":
    main()
