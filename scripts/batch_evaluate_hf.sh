#!/bin/bash
# Run all HF models × 4 conditions (bbq/bbq_trigger × direct/cot) via Singularity.
#
# Conditions:
#   bbq           — prompt_pairs_bbq.json,         direct  (stereo + contrast)
#   bbq_cot       — prompt_pairs_bbq.json,         CoT     (stereo + contrast)
#   bbq_trigger   — prompt_pairs_bbq_trigger.json, direct  (stereo only analyzed)
#   bbq_trigger_cot — prompt_pairs_bbq_trigger.json, CoT   (stereo only analyzed)
#
# All models use Singularity (transformers 5.3.0) — the conda env has a
# torch/torchvision version mismatch that prevents importing transformers.pipeline.
#
# Models are looked up first in scratch HF cache, then in ~/.cache/huggingface.
# Skip-if-exists: if the output JSON already exists, the run is skipped so the
# script can be safely re-run after interruption.

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"
module load singularity

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_SCRATCH="${HF_SCRATCH:-${HF_HOME:-$HOME/.cache/huggingface}}"
HF_DEFAULT="$HOME/.cache/huggingface"
SIF="${SIF:?Set SIF to your Singularity image path — see scripts/config.env.example}"
BBQ_FILE="$PROJ_DIR/data/prompt_pairs_bbq.json"
BBQ_TRIGGER_FILE="$PROJ_DIR/data/prompt_pairs_bbq_trigger.json"

LOG="$PROJ_DIR/results/batch_hf_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJ_DIR/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== fairness-logic HF batch evaluation ==="
echo "Log: $LOG"
echo "Started: $(date)"

# ── Locate the HF cache dir that has a given model ────────────────────────────
get_hf_home() {
    local cache_key="models--$(echo "$1" | tr '/' '--')"
    if [[ -d "$HF_SCRATCH/hub/$cache_key" ]]; then
        echo "$HF_SCRATCH"
    elif [[ -d "$HF_DEFAULT/hub/$cache_key" ]]; then
        echo "$HF_DEFAULT"
    else
        echo "$HF_SCRATCH"   # fallback: will download here on first use
    fi
}

# ── Run one condition via Singularity (skip if output already exists) ─────────
run_sif() {
    local model="$1" tag="$2" pairs_file="$3" cot_flag="${4:-}"
    local safe_name hf_home out_file
    safe_name=$(echo "$model" | tr '/' '_' | tr ':' '_')
    out_file="$PROJ_DIR/results/${safe_name}_${tag}_results.json"
    hf_home=$(get_hf_home "$model")

    if [[ -f "$out_file" ]]; then
        echo "  [skip] $tag — $(basename "$out_file") already exists"
        return 0
    fi

    echo ""
    echo "── $model | $tag (HF_HOME=$hf_home) ──"
    singularity exec --nv --cleanenv \
        -B "$PROJ_DIR:$PROJ_DIR" \
        -B "$HF_SCRATCH:$HF_SCRATCH" \
        -B "$HF_DEFAULT:$HF_DEFAULT" \
        --env PYTHONUNBUFFERED=1 \
        --env PYTHONNOUSERSITE=1 \
        --env CUDA_VISIBLE_DEVICES=0,1 \
        --env HF_HOME="$hf_home" \
        "$SIF" bash -c "cd '$PROJ_DIR' && python scripts/evaluate.py \
            --model '$model' \
            --provider hf \
            --tag '$tag' \
            --pairs_file '$pairs_file' \
            $cot_flag"
}

# ── Run all 4 conditions for a model ─────────────────────────────────────────
run_all4() {
    local model="$1"
    run_sif "$model" "bbq"             "$BBQ_FILE"         ""
    run_sif "$model" "bbq_trigger"     "$BBQ_TRIGGER_FILE" ""
    run_sif "$model" "bbq_cot"         "$BBQ_FILE"         "--cot"
    run_sif "$model" "bbq_trigger_cot" "$BBQ_TRIGGER_FILE" "--cot"
}

# ── All models ────────────────────────────────────────────────────────────────
ALL_MODELS=(
    # Standard models (transformers 4.52.4 would work but SIF is more reliable)
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-27b-it"
    # Newer models (require transformers 5.3.0)
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "google/gemma-4-31b-it"
)

for MODEL in "${ALL_MODELS[@]}"; do
    echo ""
    echo "=== Model: $MODEL ==="
    run_all4 "$MODEL"
done

echo ""
echo "=== All done. $(date) ==="
echo "Results in: $PROJ_DIR/results/"
