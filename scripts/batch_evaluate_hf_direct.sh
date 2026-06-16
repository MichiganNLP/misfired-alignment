#!/bin/bash
# Run all HF models × 4 conditions directly (no Singularity).
# Uses the base conda Python which has CUDA, transformers, and accelerate.
#
# Conditions:
#   bbq             — prompt_pairs_bbq.json,         direct
#   bbq_cot         — prompt_pairs_bbq.json,         CoT
#   bbq_trigger     — prompt_pairs_bbq_trigger.json, direct
#   bbq_trigger_cot — prompt_pairs_bbq_trigger.json, CoT
#
# Skip-if-exists: if the output JSON already exists, the run is skipped.

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HF_SCRATCH="${HF_SCRATCH:-${HF_HOME:-$HOME/.cache/huggingface}}"
HF_DEFAULT="$HOME/.cache/huggingface"
BBQ_FILE="$PROJ_DIR/data/prompt_pairs_bbq.json"
BBQ_TRIGGER_FILE="$PROJ_DIR/data/prompt_pairs_bbq_trigger.json"
PYTHON="${PYTHON:-python}"

LOG="$PROJ_DIR/results/batch_hf_direct_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJ_DIR/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== fairness-logic HF batch evaluation (direct) ==="
echo "Python: $PYTHON"
echo "Log: $LOG"
echo "Started: $(date)"

# ── Locate the HF cache dir that has a given model ───────────────────────────
get_hf_home() {
    local cache_key="models--$(echo "$1" | tr '/' '--')"
    if [[ -d "$HF_SCRATCH/hub/$cache_key" ]]; then
        echo "$HF_SCRATCH"
    elif [[ -d "$HF_DEFAULT/hub/$cache_key" ]]; then
        echo "$HF_DEFAULT"
    else
        echo "$HF_SCRATCH"
    fi
}

# ── Run one condition directly (skip if output already exists) ────────────────
run_direct_py() {
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
    PYTHONUNBUFFERED=1 HF_HOME="$hf_home" CUDA_VISIBLE_DEVICES=0,1 \
        "$PYTHON" "$PROJ_DIR/scripts/evaluate.py" \
            --model "$model" \
            --provider hf \
            --tag "$tag" \
            --pairs_file "$pairs_file" \
            $cot_flag \
    || echo "  [WARN] evaluate.py exited with error for $model/$tag — continuing batch"
}

# ── Run 2 direct conditions only (no CoT) ────────────────────────────────────
run_no_cot() {
    local model="$1"
    run_direct_py "$model" "bbq"         "$BBQ_FILE"         ""
    run_direct_py "$model" "bbq_trigger" "$BBQ_TRIGGER_FILE" ""
}

# ── All models ────────────────────────────────────────────────────────────────
# Llama-3.2-3B: also run bbq_cot (in progress, resumes from JSONL checkpoint).
# All other models: bbq + bbq_trigger only.

echo ""
echo "=== Model: meta-llama/Llama-3.2-3B-Instruct ==="
run_no_cot "meta-llama/Llama-3.2-3B-Instruct"
run_direct_py "meta-llama/Llama-3.2-3B-Instruct" "bbq_cot" "$BBQ_FILE" "--cot"

ALL_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "google/gemma-3-27b-it"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "google/gemma-4-31b-it"
)

for MODEL in "${ALL_MODELS[@]}"; do
    echo ""
    echo "=== Model: $MODEL ==="
    run_no_cot "$MODEL"
done

echo ""
echo "=== All done. $(date) ==="
echo "Results in: $PROJ_DIR/results/"
