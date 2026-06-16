"""
Extended head-ablation run for top-K analysis at K ∈ {1, 3, 5, 10, 20}.

Differs from the standard `head_ablation.py` in two ways:
  (1) Saves the full (n_layers × n_heads) single-head specificity grid as
      `<pair_id>_specificity.npz` per pair, so any future top-K analysis
      can be done from cache without re-running the slow specificity sweep.
  (2) Multi-head ablation is run at K = 1, 3, 5, 10, 20 instead of
      [1, 3, 5, top_n], giving a continuous K-curve for the "is the circuit
      surgical or distributed?" question.

By default this targets Gemma-3-27B-IT failure pairs only (the case where
top-10 only recovered 17%). Outputs land in
    results/mechinterp/google_gemma-3-27b-it/head_ablation_topk/
so the existing top-10 results are not clobbered.

Usage:
  python scripts/mechinterp/head_ablation_topk.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from head_ablation import head_ablation_sweep, multi_head_ablation
from utils import (
    build_prompt_pairs, get_decoder_handles, get_logit_diff,
    get_yes_no_ids, load_model, set_run_context,
)

MODEL       = "google/gemma-3-27b-it"
PAIRS_FILE  = "data/mechinterp_pairs_gemma3.json"
OUT_SUBDIR  = "head_ablation_topk"
K_VALUES    = [1, 3, 5, 10, 20]
N_TOP_SAVED = 50   # also save top-50 ranked head list for quick analysis


def main():
    # Redirect outputs to results/mechinterp/<model>/<OUT_SUBDIR>/
    set_run_context(MODEL, suffix="")
    out_dir = utils.RESULTS_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load failure pair ids from the family-specific pair file
    pair_dicts = json.load(open(PAIRS_FILE))["pairs"]
    failure_ids = {p["id"] for p in pair_dicts if p.get("role") == "failure"}
    print(f"Failure pairs to process: {len(failure_ids)}")

    # Load the model (sharded across GPUs visible to CUDA_VISIBLE_DEVICES)
    t0 = time.time()
    model, tokenizer = load_model(MODEL, device="auto")
    print(f"Model loaded in {time.time() - t0:.0f}s")

    pairs = build_prompt_pairs(tokenizer, pairs_file=PAIRS_FILE)
    pairs = [p for p in pairs if p.id in failure_ids]
    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    print(f"yes ids: {yes_ids}\nno  ids: {no_ids}\n")

    H = get_decoder_handles(model)
    in_dev = H.embed.weight.device
    print(f"Model layers: {H.n_layers}, heads: {H.n_heads}, head_dim: {H.head_dim}")

    results: dict[str, dict] = {}
    summary_path = out_dir / "ablation_topk_results.json"
    if summary_path.exists():
        results = json.load(open(summary_path))
        already_done = set(results.keys())
        print(f"Resuming — {len(already_done)} pairs already complete")
    else:
        already_done = set()

    pairs_to_run = [p for p in pairs if p.id not in already_done]
    print(f"Running {len(pairs_to_run)} pairs\n" + "=" * 60)

    for i, pair in enumerate(pairs_to_run, 1):
        t_pair = time.time()
        print(f"\n[{i}/{len(pairs_to_run)}] {pair.id}  ({pair.category}, "
              f"{pair.stereotyped_group} vs {pair.contrast_group})")

        # ── Step 1: full single-head specificity sweep ──
        delta_s, delta_c, specificity = head_ablation_sweep(
            model, tokenizer,
            pair.stereotyped_prompt, pair.contrast_prompt,
            yes_ids, no_ids,
        )
        # Persist full grid for any future top-K extension
        np.savez(out_dir / f"{pair.id}_specificity.npz",
                 delta_s=delta_s, delta_c=delta_c, specificity=specificity)

        # ── Step 2: rank top-N heads ──
        n_layers, n_heads = specificity.shape
        flat = []
        for li in range(n_layers):
            for hi in range(n_heads):
                flat.append((li, hi, float(specificity[li, hi]),
                             float(delta_s[li, hi]), float(delta_c[li, hi])))
        flat.sort(key=lambda x: -x[2])
        top_records = [
            {"layer": li, "head": hi, "specificity": s,
             "delta_stereo": ds, "delta_contrast": dc}
            for (li, hi, s, ds, dc) in flat[:N_TOP_SAVED]
        ]
        head_list = [(r["layer"], r["head"]) for r in top_records]

        # ── Step 3: baseline logit diff (no ablation) ──
        s_inputs = tokenizer(pair.stereotyped_prompt, return_tensors="pt").to(in_dev)
        with torch.no_grad():
            base_ld = get_logit_diff(model(**s_inputs).logits, yes_ids, no_ids)
        base_ans = "yes" if base_ld > 0 else "no"
        print(f"  baseline: ans='{base_ans}'  logit_diff={base_ld:+.3f}")

        # ── Step 4: multi-head ablation at each K ──
        recoveries = {}
        for k in K_VALUES:
            if k > len(head_list):
                continue
            new_ld = multi_head_ablation(
                model, tokenizer, pair.stereotyped_prompt,
                yes_ids, no_ids, head_list[:k],
            )
            new_ans = "yes" if new_ld > 0 else "no"
            recovered = (base_ld < 0) and (new_ld > 0)
            recoveries[str(k)] = {
                "logit_diff": float(new_ld),
                "answer":     new_ans,
                "recovered":  bool(recovered),
            }
            mark = "✓ RECOVERED" if recovered else "✗"
            print(f"  top-{k:>2} ablation: ans='{new_ans}'  ld={new_ld:+.3f}  {mark}")

        results[pair.id] = {
            "category":        pair.category,
            "stereotyped_group": pair.stereotyped_group,
            "contrast_group":    pair.contrast_group,
            "base_answer":     base_ans,
            "base_logit_diff": float(base_ld),
            "top_alignment_heads": top_records,
            "multi_head_recovery": recoveries,
        }
        # Atomic incremental save (crash recovery)
        tmp = summary_path.with_suffix(".json.tmp")
        json.dump(results, open(tmp, "w"), indent=2)
        tmp.replace(summary_path)

        elapsed = time.time() - t_pair
        print(f"  pair elapsed: {elapsed:.0f}s")

    # Final recovery summary
    print("\n" + "=" * 60)
    print("Recovery rate summary (failure pairs):")
    for k in K_VALUES:
        recs = [r["multi_head_recovery"].get(str(k), {}).get("recovered")
                for r in results.values()]
        recs = [v for v in recs if v is not None]
        if recs:
            print(f"  top-{k:>2}: {sum(recs)}/{len(recs)} = "
                  f"{100*sum(recs)/len(recs):.0f}% recovered")
    print(f"\nDone. Total pairs: {len(results)}")


if __name__ == "__main__":
    main()
