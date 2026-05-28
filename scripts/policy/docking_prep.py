"""Covalent docking input prep (Phase A2).

Per (synthase, cycle), emit a docking bundle that a real covalent-docking run
(AutoDock-Vina with covalent restraint, AF3 covalent co-fold, or OpenMM constrained
minimization) can consume the moment AF3 structures land on EX3.

Each bundle contains:
  ligand.sdf       3D-embedded intermediate; the sentinel '*' atom (ACP linker stub)
                   is preserved as the explicit covalent attachment point.
  manifest.json    metadata: synthase, cycle, intermediate SMILES, predict-decision
                   state, training label, AF3-stub paths, covalent-bond definition
                   (sentinel atom -> ACP serine OG), active catalytic-domain hint.

All AF3 paths are STUBS for Phase A -- the schema is real, the file pointers are
placeholders. Phase B drops the real AF3 outputs into the same paths.

Scope: HR + PR cycles only (the per-cycle reduction-prediction policy's scope; n=30).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "policy" / "intermediates.parquet"
OUT_DIR = ROOT / "data" / "policy" / "docking"
SENTINEL_LABEL = "ACP_LINKER"  # human-readable name attached to the '*' atom in SDF


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def embed_ligand(smiles: str) -> Chem.Mol:
    """3D-embed the intermediate; keep the '*' sentinel as an explicit atom.

    The sentinel marks the covalent attachment point -- a docking engine will
    place a covalent bond between this atom and the ACP serine OG of the protein.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"could not parse {smiles!r}")
    mol = Chem.AddHs(mol)
    # Sentinel '*' becomes Xe in the SDF (atomic num 0 doesn't embed cleanly); we
    # tag it via an atom property so docking tools can find it after writing.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetProp("attachment_role", SENTINEL_LABEL)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE  # deterministic embedding
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Some sentinel-bearing intermediates fail ETKDGv3 with default options; try
        # a basic distance-geometry fallback before giving up.
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise RuntimeError(f"3D embedding failed for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:  # noqa: BLE001 -- MMFF may not parameterize the * sentinel
        pass
    return mol


def find_attachment_idx(mol: Chem.Mol) -> int:
    """Return the atom index of the '*' sentinel (the ACP-linker attachment point)."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            return atom.GetIdx()
    raise RuntimeError("no sentinel '*' atom found in intermediate")


@dataclass
class Manifest:
    synthase: str
    bgc: str
    subclass: str
    cycle_t: int
    n_cycles: int
    label_reduction: str
    label_cmet: bool
    label_extender: str
    starter: str
    # The state the per-cycle reduction policy actually conditions on: after
    # extension + optional C-MeT, before reduction.
    smiles_predict_state: str
    smiles_post_reduce: str  # for chain-of-state reconstruction
    ligand_sdf_path: str  # relative to repo root
    ligand_attachment_atom_idx: int  # the '*' sentinel atom in the SDF
    # Stubbed Phase-B AF3 outputs.
    protein_pdb_path: str | None  # apo synthase structure (Phase B)
    covalent_complex_pdb_path: str | None  # AF3 co-fold w/ phosphopantetheine + intermediate
    acp_serine_resid: int | None  # the ACP active-site serine OG (Phase B)
    active_catalytic_domain: str  # KR by default for HR/PR reduction prediction
    covalent_bond: dict  # {ligand_atom: sentinel idx, protein_atom: "ACP_SER:OG"}
    notes: str


def build_manifest(row: pd.Series, ligand_sdf_path: Path, attachment_idx: int) -> Manifest:
    syn_slug = _slug(row["synthase"])
    return Manifest(
        synthase=row["synthase"],
        bgc=row["bgc"],
        subclass=row["subclass"],
        cycle_t=int(row["cycle_t"]),
        n_cycles=int(row["n_cycles"]),
        label_reduction=row["label_reduction"],
        label_cmet=bool(row["label_cmet"]),
        label_extender=row["label_extender"],
        starter=row["starter"],
        smiles_predict_state=row["smiles_post_cmet"],
        smiles_post_reduce=row["smiles_post_reduce"],
        ligand_sdf_path=str(ligand_sdf_path.relative_to(ROOT)),
        ligand_attachment_atom_idx=attachment_idx,
        protein_pdb_path=f"data/policy/structures/{syn_slug}/apo.pdb",  # STUB
        covalent_complex_pdb_path=(
            f"data/policy/poses/{syn_slug}/cycle_{int(row['cycle_t'])}.pdb"
        ),  # STUB
        acp_serine_resid=None,  # Phase B: identify ACP active-site serine from AF3
        active_catalytic_domain="KR",  # default for HR/PR reduction prediction
        covalent_bond={
            "ligand_atom_idx": attachment_idx,
            "protein_atom": "ACP_SER:OG",
            "bond_order": 1,
            "notes": "covalent attachment via phosphopantetheine arm; sentinel '*' "
            "stands in for the pant arm endpoint that bonds to ACP serine OG",
        },
        notes=(
            f"Phase A stub: protein and pose PDB paths are placeholders. Drop AF3 "
            f"apo prediction into {{protein_pdb_path}} and the covalent co-fold (or "
            f"docked complex) into {{covalent_complex_pdb_path}} to activate."
        ),
    )


def write_bundle(row: pd.Series) -> dict:
    syn_slug = _slug(row["synthase"])
    cycle_t = int(row["cycle_t"])
    bundle_dir = OUT_DIR / syn_slug / f"cycle_{cycle_t}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Ligand = the substrate the reduction operator would act on = post_cmet state.
    mol = embed_ligand(row["smiles_post_cmet"])
    attachment_idx = find_attachment_idx(mol)
    ligand_sdf = bundle_dir / "ligand.sdf"
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()

    manifest = build_manifest(row, ligand_sdf, attachment_idx)
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    return {
        "synthase": row["synthase"],
        "cycle_t": cycle_t,
        "ligand_sdf": str(ligand_sdf.relative_to(ROOT)),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "attachment_atom_idx": attachment_idx,
    }


def main() -> None:
    df = pd.read_parquet(IN)
    scope = df[df["subclass_hr_pr"]].copy()  # HR+PR only -- reduction-prediction scope
    print(f"prep scope: {len(scope)} cycles across {scope['synthase'].nunique()} synthases")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    failures: list[tuple[str, int, str]] = []
    for _, row in scope.iterrows():
        try:
            index.append(write_bundle(row))
        except Exception as exc:  # noqa: BLE001 -- report every failure
            failures.append((row["synthase"], int(row["cycle_t"]), repr(exc)))

    idx_df = pd.DataFrame(index)
    idx_path = OUT_DIR / "_index.parquet"
    idx_df.to_parquet(idx_path, index=False)
    print(f"wrote {len(index)} docking bundles + {idx_path.relative_to(ROOT)}")
    print(f"sample manifest: {index[0]['manifest'] if index else '(none)'}")
    if failures:
        print("\nFAILURES:")
        for syn, t, exc in failures:
            print(f"  {syn} cycle {t}: {exc}")


if __name__ == "__main__":
    main()
