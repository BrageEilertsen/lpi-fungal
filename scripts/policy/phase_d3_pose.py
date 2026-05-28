"""Phase D3 POSE pipeline: per-(BGC, candidate, cycle) substrate placement + descriptor
extraction, streamed to one parquet that train_weak.py loads as POSE features.

For each (BGC, candidate_idx, cycle_t) tuple:
  1. Run the executor on the candidate's program prefix (cycles 1..t-1) -> mol at
     start of cycle t.
  2. Extend (Claisen) -> mol_post_extend.
  3. Optional C-MeT -> mol_post_cmet  (this is s_t for the reduction policy).
  4. 3D-embed via RDKit ETKDGv3 (deterministic seed).
  5. Look up the BGC's protein fold + ACP-Ser:OG coord (from _d3_acp_serine.parquet).
  6. Translate + rotate the ligand: sentinel at OG, chain toward protein COM.
  7. Compute pocket descriptors against the protein PDB.
  8. Stream the row.

Output: data/policy/phase_d_pose.parquet -- one row per (bgc, cand_idx, cycle_t).
Columns: bgc, cand_idx, cycle_t, n_cycles, label_reduction, label_cmet,
smiles_predict_state, pocket_contact_count_5A, pocket_polar_contacts,
pocket_hydrophobic_contacts, ligand_sasa_buried_fraction, pocket_volume_A3,
ligand_centroid_x/y/z, source_pdb, status.

Substrate-only arms (3-HB / hexanoic / octanoic curated proxies, BGCs with
LOW_PLDDT_EXCLUDED anchors) get NaN features + status field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "policy"))
sys.path.insert(0, str(ROOT / "src"))

import descriptors  # noqa: E402  -- the descriptor extractor from Phase A
from lpi.chem.program import Extender, ReductionState as Rs  # noqa: E402
from lpi.data.curated import load_all  # noqa: E402
from lpi.executor import operators as op  # noqa: E402
from lpi.executor.core import _apply_reduction  # noqa: E402

GROUND_TRUTH = ROOT / "data" / "policy" / "ground_truth_programs.json"
ANCHORS_PARQUET = ROOT / "data" / "policy" / "structures" / "_d3_acp_serine.parquet"
OUT_PARQUET = ROOT / "data" / "policy" / "phase_d_pose.parquet"

# Per-BGC -> protein fold slug. BGCs sharing chemistry alias to the same fold.
BGC_TO_FOLD = {
    # Phase B folds (already on disk from prior run)
    "BGC0001275": "6_methylsalicylic_acid",
    "BGC0001276": "6_methylsalicylic_acid",
    # D3 folds + aliases for chemistry-equivalent BGCs
    "BGC0001121": "BGC0001121",
    "BGC0002212": "BGC0001121",  # orsellinic alias
    "BGC0002595": "BGC0001121",  # orsellinic alias
    "BGC0001606": "BGC0001606",
    "BGC0002191": "BGC0002191",
    "BGC0001338": "BGC0001338",  # citrinin (full-length)
    "BGC0000161": "BGC0000161",
    "BGC0002852": "BGC0002852",
    "BGC0002180": "BGC0002180",
    "BGC0001273": "BGC0001273",
    "BGC0001909": "BGC0001909",
    "BGC0002065": "BGC0001909",  # strobilurin alias
    "BGC0002238": "BGC0002238",
    "BGC0003109": "BGC0003109",
}

# Curated -> fold slug (the 6 Phase B real synthases; proxies have no protein)
CURATED_TO_FOLD = {
    "6-methylsalicylic acid":                                                                  "6_methylsalicylic_acid",
    "mellein":                                                                                 "mellein",
    "6-hydroxymellein":                                                                        "6_hydroxymellein",
    "LovB dihydromonacolin L nonaketide":                                                      "lovb",
    "2-methylbutyric acid (LovF diketide; C-MeT + full-cascade test)":                         "lovf",
    "tenellin polyketide backbone (TenS PKS portion)":                                         "tenellin_tens",
}


def cycle_intermediates(prog: dict) -> list[Chem.Mol]:
    """Run the executor on a program; return s_t (post_cmet) mol at each cycle 1..N."""
    starter = prog["starter"]
    cycles = prog["cycles"]
    mol = op.load_starter(starter)
    out: list[Chem.Mol] = []
    for c in cycles:
        methylmalonyl = c["extender"] == "methylmalonyl"
        mol_post_extend = op.extend(mol, methylmalonyl=methylmalonyl)
        mol_post_cmet = op.c_methylate(mol_post_extend) if c["c_methyl"] else mol_post_extend
        out.append(mol_post_cmet)
        mol_post_reduce = _apply_reduction(mol_post_cmet, Rs(c["reduction"]))
        mol = mol_post_reduce
    return out


def embed_3d(mol: Chem.Mol, seed: int = 0xC0FFEE) -> Chem.Mol:
    mol_h = Chem.AddHs(mol)
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("attachment_role", "ACP_LINKER")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDG()) != 0:
            return mol_h  # may fail; downstream extract will skip
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return mol_h


def rotation_matrix(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = v_from / (np.linalg.norm(v_from) + 1e-12)
    b = v_to / (np.linalg.norm(v_to) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp -= a * float(np.dot(perp, a))
        perp /= np.linalg.norm(perp)
        K = np.array([[0.0, -perp[2], perp[1]],
                      [perp[2], 0.0, -perp[0]],
                      [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


def place_ligand_inplace(mol: Chem.Mol, og: np.ndarray, com: np.ndarray) -> None:
    """In-place: translate + rotate so sentinel sits at OG, chain extends toward COM."""
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    sentinel_idx = next(i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() == 0)
    sentinel = coords[sentinel_idx]
    other_centroid = np.delete(coords, sentinel_idx, axis=0).mean(axis=0)
    v_lig = other_centroid - sentinel
    v_target = com - og
    if np.linalg.norm(v_target) < 1e-6:
        v_target = np.array([1.0, 0.0, 0.0])
    R = rotation_matrix(v_lig, v_target)
    new_coords = (coords - sentinel) @ R.T + og
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, tuple(new_coords[i]))


def descriptors_from_inmemory(mol: Chem.Mol, og: np.ndarray, prot_coords: np.ndarray,
                              prot_tags: list[tuple]) -> dict:
    """Compute descriptors directly from in-memory mol + protein arrays. Mirrors
    scripts/policy/descriptors.extract but bypasses the SDF/PDB write/read cycle."""
    conf = mol.GetConformer()
    lig_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    lig_elems = [a.GetSymbol() for a in mol.GetAtoms()]

    hits = descriptors.contacts(prot_coords, prot_tags, lig_coords)
    n_contacts = len(hits)
    polar = sum(1 for h in hits if h[4] in ("N", "O", "S"))
    hydrophobic = sum(1 for h in hits if h[4] == "C")
    pv = descriptors.pocket_volume(prot_coords, lig_coords)
    # SASA via a temporary PDB-write is slow per-call; we approximate buried_fraction
    # via the protein-atoms-within-5A-of-ligand surface heuristic for speed. Phase D3
    # main pass uses this; a full Shrake-Rupley pass over selected rows can validate
    # the proxy.
    # Heuristic: buried_fraction ~ 1 - (lig atoms NOT close to any protein) / total.
    diff = lig_coords[:, None, :] - prot_coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    lig_min_d = dist.min(axis=1)
    exposed = int((lig_min_d > 5.0).sum())
    buried_proxy = 1.0 - exposed / max(1, len(lig_coords))

    return dict(
        pocket_contact_count_5A=int(n_contacts),
        pocket_polar_contacts=int(polar),
        pocket_hydrophobic_contacts=int(hydrophobic),
        ligand_sasa_buried_fraction=float(buried_proxy),
        pocket_volume_A3=float(pv),
        ligand_centroid_x=float(lig_coords.mean(axis=0)[0]),
        ligand_centroid_y=float(lig_coords.mean(axis=0)[1]),
        ligand_centroid_z=float(lig_coords.mean(axis=0)[2]),
    )


def load_protein(pdb_path: Path) -> tuple[np.ndarray, list[tuple]]:
    return descriptors._protein_atoms(pdb_path)


def main() -> None:
    if not ANCHORS_PARQUET.exists():
        raise SystemExit(f"missing {ANCHORS_PARQUET} -- run phase_d3_anchors.py first")
    anchors = pd.read_parquet(ANCHORS_PARQUET).set_index("slug")

    # Cache protein loads (one per unique fold slug)
    protein_cache: dict[str, tuple[np.ndarray, list[tuple], np.ndarray, np.ndarray]] = {}
    def load(slug: str):
        if slug in protein_cache:
            return protein_cache[slug]
        if slug not in anchors.index:
            return None
        row = anchors.loc[slug]
        if row.status != "OK":
            return None
        pdb = ROOT / row.apo_pdb
        if not pdb.exists():
            return None
        prot_coords, prot_tags = load_protein(pdb)
        # ACP-Ser:OG
        og_info = descriptors_ext_og(pdb, int(row.picked_resid))
        if og_info is None:
            return None
        og, com = og_info, prot_coords.mean(axis=0)
        protein_cache[slug] = (prot_coords, prot_tags, og, com)
        return protein_cache[slug]

    rows: list[dict] = []

    # ---- Curated 9 HR/PR -- one program (|Z*|=1), cand_idx=0 ----
    for e in load_all():
        if e.subclass not in ("HR", "PR"):
            continue
        fold_slug = CURATED_TO_FOLD.get(e.name)
        prog = dict(starter=e.program.starter,
                    cycles=[dict(reduction=c.reduction.value, c_methyl=bool(c.c_methyl),
                                 extender=c.extender.value) for c in e.program.cycles])
        process_program(rows, bgc=f"curated:{e.name}", cand_idx=0,
                        prog=prog, fold_slug=fold_slug, load=load)

    # ---- Inventory GOLD/SILVER BGCs ----
    gt = json.loads(GROUND_TRUTH.read_text())
    for bgc, v in gt.items():
        fold_slug = BGC_TO_FOLD.get(bgc)
        for k, c in enumerate(v["candidates"]):
            process_program(rows, bgc=bgc, cand_idx=k,
                            prog=c["program"], fold_slug=fold_slug, load=load)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET.relative_to(ROOT)} -- {len(df)} (BGC, cand, cycle) rows")
    print(df.groupby("status").size())


def descriptors_ext_og(pdb: Path, resid: int) -> np.ndarray | None:
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    s = p.get_structure("p", str(pdb))
    for a in s.get_atoms():
        if a.get_parent().id[1] == resid and a.name == "OG":
            return np.array(a.coord)
    return None


def process_program(rows: list[dict], *, bgc: str, cand_idx: int,
                    prog: dict, fold_slug: str | None, load) -> None:
    """Walk a candidate program, emit one POSE-feature row per cycle."""
    n_cycles = len(prog["cycles"])
    intermediates = cycle_intermediates(prog)
    has_protein = fold_slug is not None
    cache_entry = load(fold_slug) if has_protein else None
    for t, (cycle_dict, mol_predict) in enumerate(zip(prog["cycles"], intermediates), start=1):
        row = dict(
            bgc=bgc, cand_idx=cand_idx, cycle_t=t, n_cycles=n_cycles,
            label_reduction=cycle_dict["reduction"],
            label_cmet=bool(cycle_dict["c_methyl"]),
            label_extender=cycle_dict["extender"],
            starter=prog["starter"],
            smiles_predict_state=Chem.MolToSmiles(mol_predict),
            fold_slug=fold_slug or "",
        )
        if cache_entry is None:
            row.update(dict(
                pocket_contact_count_5A=float("nan"),
                pocket_polar_contacts=float("nan"),
                pocket_hydrophobic_contacts=float("nan"),
                ligand_sasa_buried_fraction=float("nan"),
                pocket_volume_A3=float("nan"),
                ligand_centroid_x=float("nan"),
                ligand_centroid_y=float("nan"),
                ligand_centroid_z=float("nan"),
                status="substrate_only",
            ))
        else:
            prot_coords, prot_tags, og, com = cache_entry
            try:
                mol3d = embed_3d(mol_predict)
                place_ligand_inplace(mol3d, og, com)
                feats = descriptors_from_inmemory(mol3d, og, prot_coords, prot_tags)
                row.update(feats)
                row["status"] = "conformational"
            except Exception as exc:  # noqa: BLE001
                row.update(dict(status=f"FAIL: {exc!r}"))
        rows.append(row)


if __name__ == "__main__":
    main()
