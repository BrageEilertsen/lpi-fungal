#!/bin/bash
#SBATCH --job-name=lpi-af2-kr
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=08:00:00
#SBATCH --output=af2_%j.out
#SBATCH --error=af2_%j.err
#
# Fold 6 KR+ACP didomain sequences (315-371 aa each) with ColabFold / LocalColabFold.
# Self-contained: runs in whichever directory it's submitted from; expects the
# *_kr_acp.fasta files in the same dir; writes results to ./structures/.
#
# Estimated runtime on 1x A100: ~5-10 min per sequence x 6 ~= 30-60 min total.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# EX3 module env: standard pattern (per the login banner).
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm/21.08.8

# ColabFold reachability -- the SLURM tries three paths in order:
#   1. module load colabfold (whatever the EX3 module is called)
#   2. ~/venvs/colabfold/bin/activate (user-installed; recommended fallback)
#   3. /work/$USER/colabfold-venv (scratch install, fastest weights I/O)
# Run check_colabfold.sh interactively on the login node FIRST to see which path
# is live; that script also bootstraps option (2) if neither (1) nor (2) exists.

if module avail colabfold 2>&1 | grep -qi colabfold; then
    module load colabfold
    echo "Using colabfold module: $(which colabfold_batch)"
elif [ -d "${HOME}/venvs/colabfold" ]; then
    source "${HOME}/venvs/colabfold/bin/activate"
    echo "Using user venv: $(which colabfold_batch)"
elif [ -d "/work/${USER}/colabfold-venv" ]; then
    source "/work/${USER}/colabfold-venv/bin/activate"
    echo "Using scratch venv: $(which colabfold_batch)"
else
    echo "ERROR: no colabfold reachable. Run check_colabfold.sh first." >&2
    exit 1
fi

mkdir -p structures

for fasta in *_kr_acp.fasta; do
  slug="$(basename "${fasta}" _kr_acp.fasta)"
  outdir="structures/${slug}"
  # Skip if a rank-1 PDB already exists in the structures dir (lets us layer
  # incremental D3 folds onto Phase B's existing 6 didomain outputs without
  # re-folding them).
  if compgen -G "${outdir}/*_rank_001_*.pdb" > /dev/null; then
    echo "=== SKIP ${slug} (already folded; rank-1 PDB present) ==="
    continue
  fi
  mkdir -p "${outdir}"
  echo "=== folding ${slug} -> ${outdir} ==="
  # No --templates: hhsearch isn't in the venv's path; no --amber/--use-gpu-relax:
  # avoids the pdbfixer dep crash and the unrelaxed PDB is fine for Vina docking.
  colabfold_batch \
    --num-models 1 \
    --num-recycle 3 \
    --rank plddt \
    --random-seed 0 \
    "${fasta}" "${outdir}"
done

echo
echo "AF2 done. Tar and download:"
echo "  tar -czf af2_results.tgz structures/"
echo "Then on your laptop:"
echo "  scp ex3:~/lpi-af2/af2_results.tgz ."
echo "  tar -xzf af2_results.tgz -C data/policy/"
echo "  python scripts/policy/ex3/normalize_af2_outputs.py"
