#!/bin/bash
# Wait until GPUs are free, then run the full pipeline inside Singularity.
#
# Steps:
#   1. build_from_bbq.py  (Qwen3.5-27B synthesis, needs all 4 GPUs)
#   2. generate_prompts.py
#   3. evaluate.py for each HF model (sequentially)
#   4. analyze.py

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

# Singularity is only available via the modules system on this cluster
module load singularity

SIF="${SIF:?Set SIF to your Singularity image path — see scripts/config.env.example}"
PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_DIR="${SCRATCH_DIR:-$PROJ_DIR}"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$PROJ_DIR/results"
LOG="$PROJ_DIR/results/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== fairness-logic pipeline ==="
echo "Project: $PROJ_DIR"
echo "SIF:     $SIF"
echo "Log:     $LOG"

# ── Helper: run a python command inside Singularity ───────────────────────────
singularity_python() {
    singularity exec --nv --cleanenv \
        -B "$PROJ_DIR:$PROJ_DIR" \
        -B "$SCRATCH_DIR:$SCRATCH_DIR" \
        -B "$HF_DIR:$HF_DIR" \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONNOUSERSITE=1 \
        --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        --env HF_HOME="$HF_DIR" \
        "$SIF" bash -c "cd $PROJ_DIR && $*"
}

# ── GPU polling ───────────────────────────────────────────────────────────────
# Need all 4 L40S GPUs mostly free for Qwen3.5-27B (~54 GB BF16 total).
FREE_THRESHOLD_MB=2000
POLL_INTERVAL=60

echo ""
echo "=== Waiting for GPUs to be free (< ${FREE_THRESHOLD_MB} MB used per GPU) ==="
while true; do
    mapfile -t USED < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    ALL_FREE=true
    for mem in "${USED[@]}"; do
        mem=$(echo "$mem" | tr -d ' ')
        if [ "$mem" -ge "$FREE_THRESHOLD_MB" ]; then
            ALL_FREE=false
            break
        fi
    done

    if $ALL_FREE; then
        echo "[$(date '+%H:%M:%S')] GPUs are free. Starting pipeline."
        break
    else
        echo "[$(date '+%H:%M:%S')] GPUs busy (${USED[*]} MB used). Retrying in ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    fi
done

nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# ── Step 1: Synthesize stereotypes ───────────────────────────────────────────
echo ""
echo "=== Step 1: build_from_bbq.py (Qwen/Qwen3.5-27B) ==="
singularity_python "python scripts/build_from_bbq.py --from-raw"

# ── Step 2: Generate prompt pairs ────────────────────────────────────────────
echo ""
echo "=== Step 2: generate_prompts.py ==="
singularity_python "python scripts/generate_prompts.py"

# ── Step 3: Evaluate HF models ───────────────────────────────────────────────
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=== Step 3: evaluate.py — $MODEL ==="
    singularity_python "python scripts/evaluate.py --model '$MODEL' --provider hf"
done

# ── Step 4: Analyze ───────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: analyze.py ==="
singularity_python "python scripts/analyze.py results/*.json --plot"

echo ""
echo "=== All done. Log: $LOG ==="
