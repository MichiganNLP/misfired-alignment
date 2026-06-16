#!/bin/bash
# Run evaluation across all models in sequence.
# Set API keys in environment before running:
#   export OPENAI_API_KEY=...
#   export ANTHROPIC_API_KEY=...

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PAIRS_FILE="data/prompt_pairs.json"

# ── OpenAI ──────────────────────────────────────────
echo "=== GPT-4o ==="
python scripts/evaluate.py --model gpt-4o --provider openai --pairs_file $PAIRS_FILE

echo "=== GPT-4o-mini ==="
python scripts/evaluate.py --model gpt-4o-mini --provider openai --pairs_file $PAIRS_FILE

echo "=== GPT-3.5-turbo ==="
python scripts/evaluate.py --model gpt-3.5-turbo --provider openai --pairs_file $PAIRS_FILE

# ── Anthropic ───────────────────────────────────────
echo "=== Claude claude-opus-4-7 ==="
python scripts/evaluate.py --model claude-opus-4-7 --provider anthropic --pairs_file $PAIRS_FILE

echo "=== Claude claude-sonnet-4-6 ==="
python scripts/evaluate.py --model claude-sonnet-4-6 --provider anthropic --pairs_file $PAIRS_FILE

echo "=== Claude claude-haiku-4-5-20251001 ==="
python scripts/evaluate.py --model claude-haiku-4-5-20251001 --provider anthropic --pairs_file $PAIRS_FILE

# ── Analysis ────────────────────────────────────────
echo "=== Analyzing all results ==="
python scripts/analyze.py results/*.json --plot
