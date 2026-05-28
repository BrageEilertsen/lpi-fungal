#!/bin/bash
# Minimal up/down transfer for the Tier-1 AF2 step on EX3.
#
# What goes UP: 6 KR+ACP didomain FASTAs (~15 KB total) + the SLURM script.
# What comes DOWN: 6 AF2 folds + ColabFold's scores JSONs (~1-2 MB).
#
# Everything else (Vina docking, descriptor extraction, Phase C eval) stays local
# on your laptop -- no GPU needed.
#
# USAGE (defaults to brageei@dnat.simula.no:60441; override via env):
#   bash scripts/policy/ex3/transfer.sh up
#   # ... wait for SLURM job to finish on EX3 ...
#   bash scripts/policy/ex3/transfer.sh down

set -euo pipefail

EX3_USER="${EX3_USER:-brageei}"
EX3_HOST="${EX3_HOST:-dnat.simula.no}"
EX3_PORT="${EX3_PORT:-60441}"
EX3_DIR="${EX3_DIR:-lpi-af2}"   # relative to home; expanded server-side

SSH="ssh -p ${EX3_PORT}"
SCP="scp -P ${EX3_PORT}"
RSYNC_E="-e 'ssh -p ${EX3_PORT}'"

REPO="$(git rev-parse --show-toplevel)"
cd "${REPO}"

case "${1:-}" in
  up)
    BUNDLE="$(mktemp -d)/af2_bundle"
    mkdir -p "${BUNDLE}"
    cp data/policy/sequences/*_kr_acp.fasta "${BUNDLE}/"
    cp scripts/policy/ex3/af2_run.slurm "${BUNDLE}/"
    cp scripts/policy/ex3/check_colabfold.sh "${BUNDLE}/" 2>/dev/null || true
    ${SSH} "${EX3_USER}@${EX3_HOST}" "mkdir -p ${EX3_DIR}"
    ${SCP} "${BUNDLE}"/* "${EX3_USER}@${EX3_HOST}:${EX3_DIR}/"
    echo
    echo "Uploaded $(ls "${BUNDLE}" | wc -l) files to ${EX3_HOST}:~/${EX3_DIR}/"
    echo
    echo "Next: SSH in and check that colabfold is reachable, then submit:"
    echo "  ssh -p ${EX3_PORT} ${EX3_USER}@${EX3_HOST}"
    echo "  cd ~/${EX3_DIR}"
    echo "  bash check_colabfold.sh         # verify colabfold module / venv"
    echo "  sbatch af2_run.slurm            # ~30-60 min on a100q"
    ;;
  down)
    DEST="${REPO}/data/policy/structures"
    mkdir -p "${DEST}"
    rsync -av --progress -e "ssh -p ${EX3_PORT}" \
      "${EX3_USER}@${EX3_HOST}:${EX3_DIR}/structures/" "${DEST}/"
    echo
    echo "Downloaded to ${DEST}"
    echo "Run locally:"
    echo "  PYTHONPATH=src .venv/bin/python scripts/policy/ex3/normalize_af2_outputs.py"
    echo "  PYTHONPATH=src .venv/bin/python scripts/policy/ex3/identify_acp_serine.py"
    ;;
  *)
    echo "usage: $0 {up|down}"
    exit 1
    ;;
esac
