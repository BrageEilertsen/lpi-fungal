"""Pocket-substrate descriptor extractor (Phase A3).

Given a protein PDB and a 3D ligand SDF (a covalent-pose product of Phase B), emit a
fixed-length feature vector summarising the active-site context for the per-cycle
policy pi_theta(s_t). Phase A is dry-run only -- the extraction logic runs against a
synthetic PDB + the existing 6-MSA cycle-2 ligand SDF to validate the schema; real
poses arrive in Phase B.

Feature schema (v0; subject to ablation in Phase C):

  pocket_contact_count_5A           # protein atoms within 5 A of any ligand atom
  pocket_contact_residues           # distinct (chain, resname, resseq) tuples
  pocket_polar_contacts             # contacts via N/O/S atoms (charge / H-bond proxy)
  pocket_hydrophobic_contacts       # contacts via C atoms
  ligand_sasa_in_complex            # ligand solvent-accessible surface area (A^2)
                                    # inside the protein, via Bio.PDB.SASA
  ligand_sasa_free                  # SASA of ligand alone (free in solvent)
  ligand_sasa_buried_fraction       # 1 - in_complex/free  (1 = fully buried)
  pocket_volume_A3                  # void voxels within a 10 A cube around the ligand
                                    # centroid that are not within 2 A of protein
  ligand_centroid_xyz               # (x,y,z) of ligand mean atom position
  catalytic_distances               # dict[resid -> distance from residue C-alpha to
                                    # ligand centroid]  (Phase B fills resids; v0 empty)

The result is intended as the new s_t-feature variant fed into the
substrate_state_probe harness (Phase C). Static; MD adds a temporal dimension later.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
CONTACT_CUTOFF_A = 5.0
PROTEIN_PROBE_A = 1.4  # Shrake-Rupley standard water probe radius
VOXEL_GRID_A = 1.0
VOXEL_BOX_HALF_A = 5.0  # 10 A cube around ligand centroid
VOXEL_PROTEIN_CLEARANCE_A = 2.0


@dataclass
class Descriptors:
    pocket_contact_count_5A: int = 0
    pocket_contact_residues: int = 0
    pocket_polar_contacts: int = 0
    pocket_hydrophobic_contacts: int = 0
    ligand_sasa_in_complex: float = 0.0
    ligand_sasa_free: float = 0.0
    ligand_sasa_buried_fraction: float = 0.0
    pocket_volume_A3: float = 0.0
    ligand_centroid_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    catalytic_distances: dict[str, float] = field(default_factory=dict)


def _ligand_atoms(sdf: Path) -> tuple[np.ndarray, list[str]]:
    """Return (Nx3 coords, element symbols) for the first ligand in the SDF."""
    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError(f"could not load {sdf}")
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    return coords, elems


def _protein_atoms(pdb: Path) -> tuple[np.ndarray, list[tuple]]:
    """Return (Nx3 coords, residue-tags-per-atom)."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb))
    coords: list[list[float]] = []
    tags: list[tuple] = []  # (chain_id, resname, resseq, atom_name, element)
    for atom in struct.get_atoms():
        coords.append(list(atom.coord))
        res = atom.get_parent()
        ch = res.get_parent()
        tags.append((ch.id, res.resname, res.id[1], atom.name, atom.element))
    return np.array(coords), tags


def contacts(prot_coords: np.ndarray, prot_tags: list[tuple], lig_coords: np.ndarray,
             cutoff: float = CONTACT_CUTOFF_A) -> list[tuple]:
    """Return list of (chain, resname, resseq, atom_name, element, distance) for every
    protein atom within ``cutoff`` of any ligand atom."""
    # broadcast-distance, return min distance per protein atom
    diff = prot_coords[:, None, :] - lig_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    min_d = dist.min(axis=1)
    hits = np.where(min_d <= cutoff)[0]
    return [(*prot_tags[i], float(min_d[i])) for i in hits]


def pocket_volume(prot_coords: np.ndarray, lig_coords: np.ndarray,
                  grid: float = VOXEL_GRID_A, half: float = VOXEL_BOX_HALF_A,
                  clearance: float = VOXEL_PROTEIN_CLEARANCE_A) -> float:
    """Approximate pocket void volume = grid voxels in a box around ligand centroid
    not within ``clearance`` of any protein atom."""
    centroid = lig_coords.mean(axis=0)
    axis = np.arange(-half, half + grid * 0.5, grid)
    grid_xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    voxel_centers = centroid + grid_xyz
    if not len(prot_coords):
        return float(len(voxel_centers) * grid ** 3)
    # nearest-protein-atom distance per voxel; vectorised in chunks for memory safety
    keep = np.ones(len(voxel_centers), dtype=bool)
    CHUNK = 4096
    for i in range(0, len(voxel_centers), CHUNK):
        chunk = voxel_centers[i : i + CHUNK]
        d = np.sqrt(((chunk[:, None, :] - prot_coords[None, :, :]) ** 2).sum(-1)).min(axis=1)
        keep[i : i + CHUNK] = d > clearance
    return float(keep.sum() * grid ** 3)


def ligand_sasa(pdb: Path, lig_coords: np.ndarray, lig_elems: list[str]) -> tuple[float, float]:
    """Return (sasa_in_complex, sasa_free) for the ligand atoms in A^2.

    "In complex" = SASA computed on a structure containing protein + ligand;
    "free" = ligand atoms alone in vacuum (no protein occluding).

    We build a tiny in-memory PDB string for the ligand atoms (as HETATM HET) and
    feed both ligand-only and protein+ligand to Shrake-Rupley.
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("p", str(pdb))

    # Add ligand as a HETATM residue under a fresh chain. Bio.PDB lets us extend the
    # Structure object directly.
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom

    model = next(struct.get_models())
    lig_chain_id = "L"
    while lig_chain_id in model:
        lig_chain_id = chr(ord(lig_chain_id) + 1)
    chain = Chain(lig_chain_id)
    res = Residue((" ", 1, " "), "LIG", " ")
    for i, (xyz, el) in enumerate(zip(lig_coords, lig_elems)):
        atom = Atom(
            name=f"{el[:1]}{i}"[:4],
            coord=np.array(xyz, dtype=float),
            bfactor=0.0,
            occupancy=1.0,
            altloc=" ",
            fullname=f"{el[:1]}{i}"[:4],
            serial_number=i + 1,
            element=el if el != "*" else "C",  # SASA can't price '*'; treat as carbon
        )
        res.add(atom)
    chain.add(res)
    model.add(chain)

    sr = ShrakeRupley(probe_radius=PROTEIN_PROBE_A)
    sr.compute(struct, level="A")
    sasa_in_complex = float(sum(a.sasa for a in res.get_atoms()))

    # Now compute ligand-only by removing other chains -- easier to build a fresh structure.
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    s2 = Structure("lig")
    m2 = Model(0)
    c2 = Chain(lig_chain_id)
    r2 = Residue((" ", 1, " "), "LIG", " ")
    for atom in res.get_atoms():
        a2 = Atom(
            name=atom.name, coord=atom.coord, bfactor=0.0, occupancy=1.0,
            altloc=" ", fullname=atom.fullname, serial_number=atom.serial_number,
            element=atom.element,
        )
        r2.add(a2)
    c2.add(r2)
    m2.add(c2)
    s2.add(m2)
    sr.compute(s2, level="A")
    sasa_free = float(sum(a.sasa for a in r2.get_atoms()))
    return sasa_in_complex, sasa_free


def extract(pdb: Path, sdf: Path, catalytic_resids: list[int] | None = None) -> Descriptors:
    lig_coords, lig_elems = _ligand_atoms(sdf)
    prot_coords, prot_tags = _protein_atoms(pdb)

    hits = contacts(prot_coords, prot_tags, lig_coords)
    n_contacts = len(hits)
    polar = sum(1 for h in hits if h[4] in ("N", "O", "S"))
    hydrophobic = sum(1 for h in hits if h[4] == "C")
    resid_set = {(h[0], h[1], h[2]) for h in hits}

    pv = pocket_volume(prot_coords, lig_coords)
    sasa_in, sasa_free = ligand_sasa(pdb, lig_coords, lig_elems)
    buried = 1.0 - (sasa_in / sasa_free) if sasa_free > 0 else 0.0

    centroid = tuple(float(x) for x in lig_coords.mean(axis=0))

    cat: dict[str, float] = {}
    if catalytic_resids:
        # Distance from each catalytic-residue C-alpha to ligand centroid.
        ca_by_resid = {
            tag[2]: prot_coords[i] for i, tag in enumerate(prot_tags) if tag[3].strip() == "CA"
        }
        for r in catalytic_resids:
            if r in ca_by_resid:
                cat[str(r)] = float(np.linalg.norm(ca_by_resid[r] - np.array(centroid)))

    return Descriptors(
        pocket_contact_count_5A=int(n_contacts),
        pocket_contact_residues=int(len(resid_set)),
        pocket_polar_contacts=int(polar),
        pocket_hydrophobic_contacts=int(hydrophobic),
        ligand_sasa_in_complex=sasa_in,
        ligand_sasa_free=sasa_free,
        ligand_sasa_buried_fraction=float(buried),
        pocket_volume_A3=pv,
        ligand_centroid_xyz=centroid,
        catalytic_distances=cat,
    )


# -----------------------------------------------------------------------------------
# Phase-A dry run: synthetic PDB + a real intermediate's SDF -> assert outputs look OK.
# -----------------------------------------------------------------------------------

def synth_pdb(path: Path, ligand_centroid: np.ndarray) -> None:
    """Write a tiny synthetic 5-residue ALA cage around ``ligand_centroid``.

    Used to validate the extractor end-to-end without a real fold. Residues are placed
    on the corners of a small octahedron at radius 4 A from the centroid so the
    extractor sees both contacts and free voxels.
    """
    cx, cy, cz = ligand_centroid
    # 5 ALA residues at +/-4 A along x/y/z (omit -z so we don't fully cage the ligand).
    offsets = [(4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0), (0, 0, 4)]
    lines = ["REMARK   1 synthetic minimal cage for descriptor dry run", "MODEL        1"]
    serial = 1
    for ri, (dx, dy, dz) in enumerate(offsets, start=1):
        for name, el, ox, oy, oz in [
            ("N", "N", 0.0, 0.0, 0.0),
            ("CA", "C", 1.5, 0.0, 0.0),
            ("C", "C", 2.5, 1.0, 0.0),
            ("O", "O", 2.3, 2.2, 0.0),
            ("CB", "C", 1.5, -1.5, 0.0),
        ]:
            x = cx + dx + ox
            y = cy + dy + oy
            z = cz + dz + oz
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{ri:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {el}"
            )
            serial += 1
    lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines))


def main() -> None:
    sdf = ROOT / "data" / "policy" / "docking" / "6_methylsalicylic_acid" / "cycle_2" / "ligand.sdf"
    if not sdf.exists():
        raise SystemExit(f"missing {sdf} -- run docking_prep.py first")

    # Place the synthetic cage around the ligand's existing centroid so we get contacts.
    lig_coords, _ = _ligand_atoms(sdf)
    centroid = lig_coords.mean(axis=0)

    tmp_pdb = ROOT / "data" / "policy" / "_synthetic_dry_run.pdb"
    synth_pdb(tmp_pdb, centroid)

    d = extract(tmp_pdb, sdf, catalytic_resids=[1, 3])
    print("Dry-run descriptors against synthetic 5-residue cage + 6-MSA cycle-2 ligand:\n")
    print(json.dumps(asdict(d), indent=2))

    # Schema sanity checks
    assert d.pocket_contact_count_5A >= 0
    assert d.pocket_contact_residues <= 5
    assert 0.0 <= d.ligand_sasa_buried_fraction <= 1.0
    assert d.pocket_volume_A3 > 0
    print("\nschema sanity checks: PASS")


if __name__ == "__main__":
    main()
