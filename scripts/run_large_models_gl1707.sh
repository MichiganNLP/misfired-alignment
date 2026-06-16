#!/bin/bash
set -e
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

SINGULARITY="${SINGULARITY:-singularity}"
SIF="${SIF:?Set SIF to your Singularity image path — see scripts/config.env.example}"
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
BASE_PAIRS="$PROJ_DIR/data/prompt_pairs_bbq.json"
TRIG_PAIRS="$PROJ_DIR/data/prompt_pairs_bbq_trigger.json"
LOG_DIR="$PROJ_DIR/results/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3

SING="$SINGULARITY exec --nv \
    --env PYTHONNOUSERSITE=1 \
    --env HF_HOME=$HF_DIR \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --bind $PROJ_DIR,$HF_DIR \
    $SIF"

# ── Download Qwen3-32B ────────────────────────────────────────────
echo "[$(date)] Downloading Qwen3-32B..."
$SING python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-32B', cache_dir='$HF_DIR/hub')
print('Qwen3-32B download complete')
"

# ── Qwen3-32B ────────────────────────────────────────────────────
echo "[$(date)] Qwen3-32B base..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model Qwen/Qwen3-32B --provider hf \
    --pairs_file $BASE_PAIRS --tag bbq \
    2>&1 | tee "$LOG_DIR/qwen3-32b_base.log"

echo "[$(date)] Qwen3-32B trigger..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model Qwen/Qwen3-32B --provider hf \
    --pairs_file $TRIG_PAIRS --tag bbq_trigger \
    2>&1 | tee "$LOG_DIR/qwen3-32b_trigger.log"

# ── Llama-3.1-70B ────────────────────────────────────────────────
echo "[$(date)] Llama-3.1-70B base..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model meta-llama/Llama-3.1-70B-Instruct --provider hf \
    --pairs_file $BASE_PAIRS --tag bbq \
    2>&1 | tee "$LOG_DIR/llama70b_base.log"

echo "[$(date)] Llama-3.1-70B trigger..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model meta-llama/Llama-3.1-70B-Instruct --provider hf \
    --pairs_file $TRIG_PAIRS --tag bbq_trigger \
    2>&1 | tee "$LOG_DIR/llama70b_trigger.log"

# ── Qwen2.5-72B ──────────────────────────────────────────────────
echo "[$(date)] Qwen2.5-72B base..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model Qwen/Qwen2.5-72B-Instruct --provider hf \
    --pairs_file $BASE_PAIRS --tag bbq \
    2>&1 | tee "$LOG_DIR/qwen25-72b_base.log"

echo "[$(date)] Qwen2.5-72B trigger..."
$SING python $PROJ_DIR/scripts/evaluate.py \
    --model Qwen/Qwen2.5-72B-Instruct --provider hf \
    --pairs_file $TRIG_PAIRS --tag bbq_trigger \
    2>&1 | tee "$LOG_DIR/qwen25-72b_trigger.log"

echo "[$(date)] All done."
