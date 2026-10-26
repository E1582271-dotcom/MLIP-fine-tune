#!/bin/bash
# Runs inside the GPU container (called by generation/hpc/run_md.pbs): puts the package tree of the
# selected potential on the Python path and runs generation/md/run_md.py from the repository root.
# Parameters arrive as environment variables: MLIP TEMPS STEPS EQUILIB LOG_EVERY SUPERCELL_TAG TRAJ_TAG
# MACE_MODEL CIF PKG_BASE PKG_PYTHON (TEMPS uses '_' as separator; CIF, if set, overrides the
# config0<SUPERCELL_TAG>.cif cell).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [ -n "${PKG_BASE:-}" ]; then
  export PYTHONUSERBASE="$PKG_BASE"
  export PYTHONPATH="$PKG_BASE/lib/${PKG_PYTHON:-python3.10}/site-packages:${PYTHONPATH:-}"
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python3 -c "import torch; print('[gpu]', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

TEMPS_CSV="${TEMPS:-800}"
TEMPS_CSV="${TEMPS_CSV//_/,}"

python3 generation/md/run_md.py \
  --mlip "${MLIP:-mace}" \
  --device cuda \
  --temps "$TEMPS_CSV" \
  --steps "${STEPS:-2000}" \
  --equilib "${EQUILIB:-500}" \
  --log-every "${LOG_EVERY:-50}" \
  --supercell-tag "${SUPERCELL_TAG:-}" \
  --mace-model "${MACE_MODEL:-small}" \
  --traj-tag "${TRAJ_TAG:-}" \
  ${CIF:+--cif "$CIF"}
