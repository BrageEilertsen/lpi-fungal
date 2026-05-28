#!/bin/bash
# Run this ONCE on the EX3 login node (srl-login1) before submitting af2_run.slurm.
#
# It probes three reachability paths in order:
#   1. EX3 colabfold module
#   2. ~/venvs/colabfold/ (user-installed)
#   3. /work/$USER/colabfold-venv/ (scratch-installed; recommended for big weight I/O)
#
# If none exist, it offers to bootstrap option (2) -- a ~5-10 min one-time install.
# After this script reports OK, sbatch af2_run.slurm will Just Work.

set -uo pipefail

module use /cm/shared/ex3-modules/latest/modulefiles 2>/dev/null || true

echo "=== probing colabfold reachability on $(hostname) ==="

# Path 1: module
if module avail colabfold 2>&1 | grep -qi colabfold; then
    echo "OK  module: colabfold available"
    module load colabfold 2>/dev/null && {
        echo "    which colabfold_batch: $(which colabfold_batch 2>/dev/null || echo missing)"
        echo "    -> SLURM will pick path 1"
        exit 0
    }
fi
echo "    (no colabfold module found)"

# Path 2: user venv
if [ -d "${HOME}/venvs/colabfold" ] && [ -x "${HOME}/venvs/colabfold/bin/colabfold_batch" ]; then
    echo "OK  user venv: ~/venvs/colabfold/ already set up"
    echo "    -> SLURM will pick path 2"
    exit 0
fi
echo "    (no ~/venvs/colabfold/ found)"

# Path 3: scratch venv
if [ -d "/work/${USER}/colabfold-venv" ] && [ -x "/work/${USER}/colabfold-venv/bin/colabfold_batch" ]; then
    echo "OK  scratch venv: /work/${USER}/colabfold-venv/ set up"
    echo "    -> SLURM will pick path 3"
    exit 0
fi
echo "    (no /work/${USER}/colabfold-venv/ found)"

echo
echo "NONE OF THE THREE PATHS ARE LIVE. Offering bootstrap..."
echo
read -r -p "Bootstrap ~/venvs/colabfold/ with LocalColabFold? [y/N] " ans
case "${ans:-N}" in
    y|Y|yes)
        set -e
        # EX3's login node runs Ubuntu 22.04 -> python3 is on PATH by default.
        # We don't need a module load. Verify version first (colabfold needs >=3.10).
        PY="$(command -v python3)"
        if [ -z "${PY}" ]; then
            echo "ERROR: no python3 on PATH. Check: module avail 2>&1 | grep -iE 'python|conda'" >&2
            exit 1
        fi
        PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        echo "using ${PY} (Python ${PYVER})"
        case "${PYVER}" in
            3.10|3.11|3.12) ;;  # colabfold-compatible
            *)
                echo "WARNING: Python ${PYVER} may not be colabfold-compatible (needs 3.10+)." >&2
                echo "         If pip install fails, run: module avail 2>&1 | grep -iE 'python|conda'" >&2
                echo "         and pick a newer one to load. Continuing anyway." >&2
                ;;
        esac

        mkdir -p "${HOME}/venvs"
        python3 -m venv "${HOME}/venvs/colabfold"
        # shellcheck disable=SC1090
        source "${HOME}/venvs/colabfold/bin/activate"
        pip install --upgrade pip wheel

        # Install colabfold *without* its bundled jax (which defaults to CPU-only
        # on pypi). We install GPU jax separately so the SLURM job uses the A100.
        pip install --no-cache-dir 'colabfold[alphafold-minus-jax]'

        # GPU jax matching CUDA 12.x (EX3 a100q nodes are typically CUDA 12).
        # If this picks the wrong CUDA, the SLURM job will fall back to CPU
        # gracefully -- ~30 min per sequence instead of ~5; total ~3 hours.
        pip install --no-cache-dir --upgrade \
            "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
            || pip install --no-cache-dir --upgrade "jax[cuda12_pip]" \
            || echo "WARNING: GPU jax install failed; colabfold will run on CPU."

        echo
        echo "Bootstrap done. Verify:"
        echo "  source ~/venvs/colabfold/bin/activate"
        echo "  colabfold_batch --help | head -5"
        echo "  python -c 'import jax; print(jax.devices())'"
        echo
        echo "If jax.devices() shows '[CudaDevice(...)]' -> GPU path live."
        echo "Then: sbatch af2_run.slurm"
        ;;
    *)
        echo "Aborted. Options:"
        echo "  - ask EX3 admin which colabfold setup is canonical"
        echo "  - manually create ~/venvs/colabfold/ and install"
        echo "  - re-run this script with 'y' to bootstrap"
        exit 1
        ;;
esac
