#!/bin/bash
# Step 6 -- fine-tune MACE-MP-0 small on the DFT-labelled Li6PS5Cl snapshots (mace_run_train).
# Runs inside the GPU container (generation/finetune/finetune_mace.pbs); mace-torch 0.3.16 provides
# mace.cli.run_train. --foundation_model makes this a fine-tune of the foundation model rather
# than training from scratch. The arguments below are those of both July fine-tunes (training logs:
# generation/README.md, step 6); FT_NAME, TRAIN_FILE and VALID_FILE select the label set, the other
# environment overrides exist for experiments only.
#
# Defaults train on the published Gamma split (data/dft/training_sets/ftG_train.xyz, ftG_valid.xyz);
# relative TRAIN_FILE/VALID_FILE/FOUNDATION paths are taken from the repository root. Outputs, in
# generation/output/finetune/ (ignored by git): models/ (li6ps5cl_ft.model = stage one,
# li6ps5cl_ft_stagetwo.model = stage two/SWA, the model named for the Gamma fine-tuned MD),
# checkpoints/, results/ (training curves) and logs/.
# SWA starts at MACE's default epoch (3/4 of 120 = 90) with MACE's default stage-two weights
# (energy 1000, forces 100) and learning rate (1e-3), as the training log records.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -n "${PKG_BASE:-}" ]; then
  export PYTHONUSERBASE="$PKG_BASE"
  export PYTHONPATH="$PKG_BASE/lib/${PKG_PYTHON:-python3.10}/site-packages:${PYTHONPATH:-}"
fi

# 'small' downloads MACE-MP-0 small; offline nodes: point FOUNDATION at a copy fetched beforehand.
FOUNDATION="${FOUNDATION:-small}"
FT_NAME="${FT_NAME:-li6ps5cl_ft}"
TRAIN_FILE="${TRAIN_FILE:-data/dft/training_sets/ftG_train.xyz}"
VALID_FILE="${VALID_FILE:-data/dft/training_sets/ftG_valid.xyz}"
case "$TRAIN_FILE" in /*) ;; *) TRAIN_FILE="$REPO/$TRAIN_FILE" ;; esac
case "$VALID_FILE" in /*) ;; *) VALID_FILE="$REPO/$VALID_FILE" ;; esac
if [ -f "$REPO/$FOUNDATION" ]; then FOUNDATION="$REPO/$FOUNDATION"; fi
# MACE writes models/, checkpoints/, results/ and logs/ below the working directory.
mkdir -p "$REPO/generation/output/finetune"
cd "$REPO/generation/output/finetune"
echo "[ft] name=$FT_NAME train=$TRAIN_FILE valid=$VALID_FILE"
echo "[ft] foundation=$FOUNDATION epochs=${EPOCHS:-120} fweight=${FWEIGHT:-10} lr=${LR:-0.0001} batch=${BATCH:-4}"

python3 -m mace.cli.run_train \
    --name="$FT_NAME" \
    --foundation_model="$FOUNDATION" \
    --train_file="$TRAIN_FILE" \
    --valid_file="$VALID_FILE" \
    --energy_key="dft_energy" \
    --forces_key="dft_forces" \
    --E0s="average" \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight="${FWEIGHT:-10}" \
    --lr="${LR:-0.0001}" \
    --batch_size="${BATCH:-4}" \
    --max_num_epochs="${EPOCHS:-120}" \
    --swa \
    --default_dtype="float32" \
    --device="cuda" \
    --seed=0 \
    --model_dir="models" \
    --checkpoints_dir="checkpoints" \
    --results_dir="results"

echo "[ft] done -> generation/output/finetune/models/${FT_NAME}.model (stage one) and ${FT_NAME}_stagetwo.model (SWA)"
echo "[ft] MD with the fine-tuned model: see generation/md/run_md.py and generation/README.md (step 7)"
