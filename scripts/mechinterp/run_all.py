"""
Master script: run all mechanistic interpretability experiments.

Loads the model once and runs:
  1. Logit lens
  2. Activation patching (layer sweep + group token sweep; head-patching opt-in)
  3. Attention analysis
  4. Head ablation

Usage:
  # Default: data-driven 60-pair set on Llama-3.1-8B-Instruct
  python scripts/mechinterp/run_all.py

  # Run on the BASE model with the same pair set (alignment-causation test)
  python scripts/mechinterp/run_all.py --model meta-llama/Llama-3.1-8B

  # Subset by id, or skip slow steps
  python scripts/mechinterp/run_all.py --example secretary_women_men
  python scripts/mechinterp/run_all.py --skip heads
  python scripts/mechinterp/run_all.py --pairs_file data/mechinterp_pairs.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import utils
from utils import load_model, build_prompt_pairs, set_run_context
from logit_lens import run_logit_lens
from activation_patching import run_activation_patching
from attention_analysis import run_attention_analysis
from head_ablation import run_head_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tokenizer", default=None,
                        help="Override tokenizer source (e.g. point a base model to its "
                             "Instruct sibling for chat-template compatibility)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs_file", default=None,
                        help="Path to pair JSON. Defaults to data/mechinterp_pairs.json "
                             "if it exists, else falls back to legacy ANALYSIS_EXAMPLES.")
    parser.add_argument("--example", nargs="+", help="Subset of pair ids")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["logit_lens", "patching", "attention", "ablation", "heads"],
                        help="Experiments to skip ('heads' = activation-patching-by-head, slowest)")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N heads for ablation")
    parser.add_argument("--output-suffix", default="",
                        help="Append to per-model output dir name "
                             "(e.g. 'notrigger' → results/mechinterp/{model}_notrigger/)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Pass enable_thinking=True to the chat template "
                             "(Qwen3 / thinking-mode models)")
    args = parser.parse_args()

    out_dir = set_run_context(args.model, suffix=args.output_suffix)

    print("=" * 60)
    print("Mechanistic Interpretability: Fairness-Induced Logic Failure")
    print(f"Model:       {args.model}")
    print(f"Pairs file:  {args.pairs_file or utils.DEFAULT_PAIRS_FILE}")
    print(f"Output dir:  {out_dir}")
    print("=" * 60)

    t0 = time.time()
    model, tokenizer = load_model(args.model, args.device, tokenizer_name=args.tokenizer)
    pairs = build_prompt_pairs(tokenizer, pairs_file=args.pairs_file,
                               enable_thinking=args.enable_thinking)

    if args.example:
        print(f"Restricting to ids: {args.example}")
    print(f"Analyzing {len(pairs) if not args.example else len(args.example)} pairs")

    # 1. Logit lens
    if "logit_lens" not in args.skip:
        print("\n" + "─" * 40)
        print(f"[{time.time()-t0:.0f}s] Experiment 1: Logit Lens")
        print("─" * 40)
        run_logit_lens(model, tokenizer, pairs, example_ids=args.example)

    # 2. Activation patching
    if "patching" not in args.skip:
        print("\n" + "─" * 40)
        print(f"[{time.time()-t0:.0f}s] Experiment 2: Activation Patching")
        print("─" * 40)
        modes = ["final", "group"]
        if "heads" not in args.skip:
            modes.append("heads")
        run_activation_patching(model, tokenizer, pairs,
                                example_ids=args.example, modes=modes)

    # 3. Attention analysis
    if "attention" not in args.skip:
        print("\n" + "─" * 40)
        print(f"[{time.time()-t0:.0f}s] Experiment 3: Attention Analysis")
        print("─" * 40)
        run_attention_analysis(model, tokenizer, pairs, example_ids=args.example)

    # 4. Head ablation
    if "ablation" not in args.skip:
        print("\n" + "─" * 40)
        print(f"[{time.time()-t0:.0f}s] Experiment 4: Head Ablation")
        print("─" * 40)
        run_head_ablation(model, tokenizer, pairs,
                          example_ids=args.example, top_n=args.top_n)

    print("\n" + "=" * 60)
    print(f"All done in {time.time()-t0:.0f}s. Results: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
