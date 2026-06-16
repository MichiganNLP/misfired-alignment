#!/bin/bash
# Run all closed-source API models × their conditions.
#
# Providers:
#   openrouter — closed-source models via OpenRouter (set OPENROUTER_API_KEY)
#   deepseek   — DeepSeek models via DeepSeek API   (set DEEPSEEK_API_KEY)
#
# Conditions per model:
#   4 conditions (bbq/bbq_trigger × direct/cot):
#     gpt-5.4-nano, gpt-5.4-mini, gemini-3.1-flash-lite, grok-4.20, deepseek-chat
#   2 direct conditions (bbq / bbq_trigger only):
#     claude-4.7-opus, claude-4.6-sonnet, gpt-5.4, gemini-3.1-pro, deepseek-reasoner
#     (Claude / GPT flagship: expensive output; reasoner: CoT = direct anyway)
#
# For bbq_trigger runs, only the stereotyped condition results are analyzed.
#
# Skip logic: if the output file already exists, the run is skipped automatically.
# This makes the script safe to re-run after partial failures.
#
# Usage:
#   export OPENROUTER_API_KEY="sk-or-v1-..."
#   export DEEPSEEK_API_KEY="sk-..."
#   bash scripts/batch_evaluate_api.sh

set -euo pipefail
[ -f "$(dirname "${BASH_SOURCE[0]}")/config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/config.env"

PROJ_DIR="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BBQ_FILE="$PROJ_DIR/data/prompt_pairs_bbq.json"
BBQ_TRIGGER_FILE="$PROJ_DIR/data/prompt_pairs_bbq_trigger.json"

LOG="$PROJ_DIR/results/batch_api_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJ_DIR/results"
exec > >(tee -a "$LOG") 2>&1

echo "=== fairness-logic API batch evaluation ==="
echo "Log: $LOG"
echo "Started: $(date)"

# ── Validate API keys are set ─────────────────────────────────────────────────
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY is not set. Export it before running this script."
    exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is not set. Export it before running this script."
    exit 1
fi

# ── Core run function ─────────────────────────────────────────────────────────
run_api() {
    local provider="$1" model="$2" tag="$3" pairs_file="$4" cot_flag="${5:-}"

    # Build expected output path to support skip-if-exists
    local safe_name
    safe_name=$(echo "$model" | tr '/' '_' | tr ':' '_')
    local out_file="$PROJ_DIR/results/${safe_name}_${tag}_results.json"

    if [[ -f "$out_file" ]]; then
        echo "  [skip] $tag — output already exists: $(basename "$out_file")"
        return 0
    fi

    echo ""
    echo "── [$provider] $model | $tag ──"
    python "$PROJ_DIR/scripts/evaluate.py" \
        --model "$model" \
        --provider "$provider" \
        --tag "$tag" \
        --pairs_file "$pairs_file" \
        $cot_flag \
    || echo "  [WARN] evaluate.py exited with error for $model/$tag — continuing batch"
}

# Convenience wrappers
run_direct() { run_api "$1" "$2" "bbq" "$BBQ_FILE" ""; run_api "$1" "$2" "bbq_trigger" "$BBQ_TRIGGER_FILE" ""; }
run_cot()    { run_api "$1" "$2" "bbq_cot" "$BBQ_FILE" "--cot"; run_api "$1" "$2" "bbq_trigger_cot" "$BBQ_TRIGGER_FILE" "--cot"; }
run_all4()   { run_direct "$1" "$2"; run_cot "$1" "$2"; }

# ── OpenRouter models ─────────────────────────────────────────────────────────
echo ""
echo "=== OPENROUTER ==="

# Flagship models — 2 direct conditions (expensive CoT output)
echo "--- Anthropic ---"
run_direct openrouter "anthropic/claude-4.7-opus-20260416"
run_direct openrouter "anthropic/claude-4.6-sonnet-20260217"

echo "--- OpenAI flagship ---"
run_direct openrouter "openai/gpt-5.4-20260305"

echo "--- Google flagship ---"
run_direct openrouter "google/gemini-3.1-pro-preview-20260219"

# Cheaper models — direct only (no CoT)
echo "--- OpenAI scaled ---"
run_direct openrouter "openai/gpt-5.4-mini-20260317"
run_direct openrouter "openai/gpt-5.4-nano-20260317"

echo "--- Google fast ---"
run_direct openrouter "google/gemini-3.1-flash-lite-preview-20260303"

echo "--- xAI ---"
run_direct openrouter "x-ai/grok-4.20-20260309"

# ── DeepSeek models ───────────────────────────────────────────────────────────
echo ""
echo "=== DEEPSEEK ==="

# deepseek-chat (V3): direct only (no CoT)
run_direct deepseek "deepseek-chat"

# deepseek-reasoner (R1): always thinks natively, CoT = direct → 2 conditions only
run_direct deepseek "deepseek-reasoner"

echo ""
echo "=== All done. $(date) ==="
echo "Results in: $PROJ_DIR/results/"
