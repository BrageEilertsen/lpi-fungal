#!/bin/bash
# Covalent AutoDock Vina docking across all 24 (synthase, cycle) bundles in
# data/policy/docking/, using the AF2 apo + identified ACP-Ser:OG as anchor.
# Requires: AutoDock Vina 1.2.5+, Meeko (ligand prep), prepare_receptor4.py (AutoDock
# Tools). Output: data/policy/poses/{slug}/cycle_{t}.pdb (the rank-1 docked complex).
#
# Single-pose default; bump --num_modes if you want top-K for ensemble descriptors.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# --- ADJUST these for EX3 ----------------------------------------------------
# module load vina/1.2.5
# module load meeko
# module load mgltools/1.5.7
# ----------------------------------------------------------------------------

REPO="$(pwd)"
DOCK_DIR="${REPO}/data/policy/docking"
POSE_DIR="${REPO}/data/policy/poses"
mkdir -p "${POSE_DIR}" "logs"

VINA_BIN="${VINA_BIN:-vina}"

for manifest in "${DOCK_DIR}"/*/cycle_*/manifest.json; do
  syn_slug="$(basename "$(dirname "$(dirname "${manifest}")")")"
  cycle_dir="$(dirname "${manifest}")"
  cycle_t="$(basename "${cycle_dir}" | sed 's/^cycle_//')"
  out="${POSE_DIR}/${syn_slug}"
  mkdir -p "${out}"

  ligand_sdf="$(python -c "import json; print(json.load(open('${manifest}'))['ligand_sdf_path'])")"
  protein_pdb="$(python -c "import json; print(json.load(open('${manifest}'))['protein_pdb_path'] or '')")"
  acp_resid="$(python -c "import json; print(json.load(open('${manifest}'))['acp_serine_resid'] or 0)")"

  if [[ -z "${protein_pdb}" || ! -f "${REPO}/${protein_pdb}" || "${acp_resid}" -eq 0 ]]; then
    echo "SKIP ${syn_slug}/cycle_${cycle_t}: no protein/ACP serine (substrate-only synthase or pre-fold)"
    continue
  fi

  echo "=== ${syn_slug}/cycle_${cycle_t}  protein=${protein_pdb}  ACP-Ser=${acp_resid} ==="

  # Receptor prep -- PDBQT with covalent residue flag on ACP serine.
  prepare_receptor4.py -r "${REPO}/${protein_pdb}" -o "${out}/cycle_${cycle_t}_rec.pdbqt" -A "hydrogens"

  # Ligand prep with covalent attachment marker -- Meeko's --covalent path:
  # uses the sentinel atom in ligand.sdf (the * dummy) as the covalent bond endpoint.
  mk_prepare_ligand.py \
    -i "${REPO}/${ligand_sdf}" \
    -o "${out}/cycle_${cycle_t}_lig.pdbqt" \
    --covalent --covalent_smarts '[#0]' --covalent_smarts_indices 0

  # Box around the ACP serine OG (the covalent anchor) -- 20 A box.
  CENTER=$(python -c "
from Bio.PDB import PDBParser
s = PDBParser(QUIET=True).get_structure('p','${REPO}/${protein_pdb}')
for a in s.get_atoms():
    if a.get_parent().id[1] == ${acp_resid} and a.name == 'OG':
        print(f'{a.coord[0]:.3f} {a.coord[1]:.3f} {a.coord[2]:.3f}'); break
")
  read CX CY CZ <<< "${CENTER}"

  # Covalent docking.
  ${VINA_BIN} \
    --receptor "${out}/cycle_${cycle_t}_rec.pdbqt" \
    --ligand   "${out}/cycle_${cycle_t}_lig.pdbqt" \
    --center_x "${CX}" --center_y "${CY}" --center_z "${CZ}" \
    --size_x 22 --size_y 22 --size_z 22 \
    --exhaustiveness 16 \
    --num_modes 5 \
    --seed 0 \
    --out "${out}/cycle_${cycle_t}_docked.pdbqt" \
    --log "${out}/cycle_${cycle_t}.log"

  # Convert top-pose PDBQT -> PDB for descriptors.
  obabel "${out}/cycle_${cycle_t}_docked.pdbqt" -O "${out}/cycle_${cycle_t}.pdb" -f 1 -l 1
done

echo
echo "Vina done. Next: python scripts/policy/ex3/run_descriptors_batch.py"
