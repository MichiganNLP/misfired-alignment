[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

module load singularity

SIF="${SIF:?Set SIF to your Singularity image path — see scripts/config.env.example}"
GPFS_PROJ="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_PROJ="${SCRATCH_DIR:-$GPFS_PROJ}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

singularity exec --nv --cleanenv \
  -B "$GPFS_PROJ:$GPFS_PROJ" \
  -B "$SCRATCH_PROJ:$SCRATCH_PROJ" \
  -B "$HF_HOME:$HF_HOME" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --env HF_HOME="$HF_HOME" \
  "$SIF" bash -lc '
    set -euo pipefail
    export PATH=/usr/local/bin:/usr/bin:/bin:$PATH
    nvidia-cuda-mps-control -d || true
    which python3 && python3 --version
    nvidia-smi || true
    exec bash
  '
