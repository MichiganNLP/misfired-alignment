#!/bin/bash
# Overnight queue dispatcher.
# Waits for all currently-running cross-family runs to finish, then
# sequentially fires the queued phases on whatever GPUs are free.
#
# Each phase is fully self-contained: spawns its jobs, waits for them, logs.
# Failures in one phase do NOT abort the dispatcher — we log and continue.

set -u  # NOT -e: a phase failing should not kill subsequent phases
[ -f "$(dirname "${BASH_SOURCE[0]}")/../config.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/../config.env"

PROJ="${PROJ_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-${PYTHON:-python}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PROJ_DIR="$PROJ"
DIR=$PROJ/results/mechinterp
TS=$(date +%Y%m%d_%H%M%S)
DISPATCH_LOG=$DIR/overnight_${TS}.log
exec > >(tee -a "$DISPATCH_LOG") 2>&1

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== overnight queue dispatcher started ==="
log "Logfile: $DISPATCH_LOG"

# ── Helpers ───────────────────────────────────────────────────────────────────

wait_for_log() {
    # Wait until the latest log matching pattern $1 has 'All done' OR a Traceback.
    local pat="$1"
    while true; do
        local f=$(ls -t "$DIR"/${pat} 2>/dev/null | head -1)
        if [[ -n "$f" ]]; then
            if grep -q "All done" "$f" 2>/dev/null; then
                log "  ✅ $(basename "$f") complete"; return 0
            fi
            if grep -qE "Traceback" "$f" 2>/dev/null; then
                log "  ⚠ $(basename "$f") errored"; return 1
            fi
        fi
        sleep 60
    done
}

count_idle_gpus() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null \
        | awk '{gsub(/MiB/,""); gsub(/ /,""); if ($0+0 < 1000) c++} END{print c+0}'
}

wait_for_idle() {
    local need=${1:-1}
    while true; do
        local n=$(count_idle_gpus)
        if (( n >= need )); then return; fi
        sleep 30
    done
}

# ── Phase 0: wait for current cross-family runs to finish ────────────────────
log ""
log "── Phase 0: blocking on currently-running cross-family runs ──"
for j in wave1_qwen3_8b_ wave2_mistral_base_ wave2_qwen3_8b_base_ wave2_qwen3.5_9b_base_; do
    wait_for_log "${j}*.log"
done
log "  ✅ Phase 0 done"

# ── Phase 1: cross-family path patching (4 GPUs parallel) ────────────────────
log ""
log "── Phase 1: cross-family path patching (Mistral I/B, Qwen3 I/B) ──"
wait_for_idle 4
PIDS=()
launch_pp() {
    local gpu=$1; local model="$2"; local tokenizer="$3"; local pairs="$4"; local tag=$5
    local plog=$DIR/pp_${tag}_${TS}.log
    log "  GPU $gpu: path patching $tag"
    HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES=$gpu \
        $PY -u $PROJ/scripts/mechinterp/path_patching.py \
            --model "$model" \
            ${tokenizer:+--tokenizer "$tokenizer"} \
            --pairs_file "$pairs" \
            --role failure > "$plog" 2>&1 &
    PIDS+=($!)
}
launch_pp 0 "mistralai/Mistral-7B-Instruct-v0.3" ""                          $PROJ/data/mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json mistral_instruct
launch_pp 1 "mistralai/Mistral-7B-v0.3"          "mistralai/Mistral-7B-Instruct-v0.3" $PROJ/data/mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json mistral_base
launch_pp 2 "Qwen/Qwen3-8B"                      ""                          $PROJ/data/mechinterp_pairs_Qwen_Qwen3-8B.json                                qwen3_instruct
launch_pp 3 "Qwen/Qwen3-8B-Base"                 "Qwen/Qwen3-8B"             $PROJ/data/mechinterp_pairs_Qwen_Qwen3-8B.json                                qwen3_base
for pid in "${PIDS[@]}"; do wait "$pid"; log "  pp pid=$pid exit=$?"; done
log "  ✅ Phase 1 done"

# ── Phase 2: Llama path patching ──────────────────────────────────────────────
log ""
log "── Phase 2: Llama-3.1-8B path patching ──"
wait_for_idle 2
PIDS=()
launch_pp 0 "meta-llama/Llama-3.1-8B-Instruct" "" $PROJ/data/mechinterp_pairs.json llama_instruct
launch_pp 1 "meta-llama/Llama-3.1-8B"          "" $PROJ/data/mechinterp_pairs.json llama_base
for pid in "${PIDS[@]}"; do wait "$pid"; log "  pp pid=$pid exit=$?"; done
log "  ✅ Phase 2 done"

# ── Phase 3: build no-trigger pair set + run mechinterp on Llama ──────────────
log ""
log "── Phase 3: building no-trigger pair set ──"
$PY - <<'PY'
import json
import os
from pathlib import Path
P = Path(os.environ.get("PROJ_DIR", ".")) / "data"
trig = json.load(open(P/"mechinterp_pairs.json"))
bbq_pairs = {p["id"]: p for p in json.load(open(P/"prompt_pairs_bbq.json"))}
out_pairs = []
for ex in trig["pairs"]:
    src = bbq_pairs.get(ex["id"])
    if not src: continue
    out_pairs.append({
        **{k: v for k, v in ex.items() if k not in ("stereotyped_user", "contrast_user")},
        "stereotyped_user": src["prompts"]["stereotyped"]["text"],
        "contrast_user":    src["prompts"]["contrast"]["text"],
    })
json.dump({**{k: v for k, v in trig.items() if k != "pairs"},
           "reference_tag_override": "bbq_notrigger_same_ids",
           "pairs": out_pairs},
          open(P/"mechinterp_pairs_notrigger.json", "w"), indent=2)
print(f"  wrote {len(out_pairs)} no-trigger pairs to {P}/mechinterp_pairs_notrigger.json")
PY

log "  launching no-trigger mechinterp on Llama-Instruct + base"
NTLOG_I=$DIR/notrigger_llama_instruct_${TS}.log
NTLOG_B=$DIR/notrigger_llama_base_${TS}.log
HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES=0 \
    $PY -u $PROJ/scripts/mechinterp/run_all.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --pairs_file $PROJ/data/mechinterp_pairs_notrigger.json \
        --output-suffix notrigger \
        --skip heads > "$NTLOG_I" 2>&1 &
PI=$!
HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES=1 \
    $PY -u $PROJ/scripts/mechinterp/run_all.py \
        --model meta-llama/Llama-3.1-8B \
        --pairs_file $PROJ/data/mechinterp_pairs_notrigger.json \
        --output-suffix notrigger \
        --skip heads > "$NTLOG_B" 2>&1 &
PB=$!
wait "$PI"; log "  notrigger Instruct exit=$?"
wait "$PB"; log "  notrigger Base exit=$?"
log "  ✅ Phase 3 done"

# ── Phase 4: CoT-simple mechinterp on Qwen3-8B ────────────────────────────────
log ""
log "── Phase 4: CoT-simple mechinterp on Qwen3-8B (enable_thinking=True) ──"
wait_for_idle 1
COTLOG=$DIR/cot_qwen3_8b_${TS}.log
HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES=0 \
    $PY -u $PROJ/scripts/mechinterp/run_all.py \
        --model Qwen/Qwen3-8B \
        --pairs_file $PROJ/data/mechinterp_pairs_Qwen_Qwen3-8B.json \
        --enable-thinking \
        --output-suffix cot \
        --skip heads > "$COTLOG" 2>&1
log "  ✅ Phase 4 done (or errored — check $COTLOG)"

# ── Phase 5: aggregate everything (best-effort, models that exist) ───────────
log ""
log "── Phase 5: aggregator over all models ──"
$PY $PROJ/scripts/mechinterp/aggregate.py \
    --models \
        meta-llama/Llama-3.1-8B-Instruct \
        meta-llama/Llama-3.1-8B \
        mistralai/Mistral-7B-Instruct-v0.3 \
        mistralai/Mistral-7B-v0.3 \
        Qwen/Qwen3-8B \
        Qwen/Qwen3-8B-Base \
        Qwen/Qwen3.5-9B \
        Qwen/Qwen3.5-9B-Base 2>&1 | tail -80

log ""
log "=== overnight queue dispatcher DONE ==="
log "Full log: $DISPATCH_LOG"
