#!/bin/bash
# Run all mechanistic interpretability experiments inside Singularity.
# Llama-3.1-8B-Instruct fits on a single L40S (46 GB) in BF16 (~16 GB).

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/../config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/../config.env"
module load singularity

SIF="${SIF:?Set SIF to your Singularity image path — see scripts/config.env.example}"
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-$PROJ_DIR}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

LOG="$PROJ_DIR/results/mechinterp/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJ_DIR/results/mechinterp"
exec > >(tee -a "$LOG") 2>&1

echo "=== Mechanistic Interpretability Pipeline ==="
echo "Log: $LOG"

singularity exec --nv --cleanenv \
    -B "$PROJ_DIR:$PROJ_DIR" \
    -B "$SCRATCH_DIR:$SCRATCH_DIR" \
    -B "$HF_DIR:$HF_DIR" \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONNOUSERSITE=1 \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_HOME="$HF_DIR" \
    "$SIF" bash -c "cd $PROJ_DIR && python scripts/mechinterp/run_all.py $*"

echo "=== Done. Results in results/mechinterp/ ==="
