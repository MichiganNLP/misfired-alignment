"""
Extend the top-K head ablation curve on Gemma-3-27B-IT failure pairs to
larger K values, reusing the cached specificity grids saved by
`head_ablation_topk.py` (so we skip the expensive single-head sweep).

For each of the 30 failure pairs:
  1. Load <pair_id>_specificity.npz (cached 62×32 specificity grid)
  2. Re-rank all 1,984 heads by specificity, keep top-300
  3. Run a single forward pass per K with the top-K heads ablated
  4. Record whether the stereotyped answer flips from incorrect to correct

K values: [1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]

Output: results/mechinterp/google_gemma-3-27b-it/head_ablation_topk/
        ablation_extended_K_results.json
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
from head_ablation import multi_head_ablation
from utils import (
    build_prompt_pairs, get_decoder_handles, get_logit_diff,
    get_yes_no_ids, load_model, set_run_context,
)

MODEL       = "google/gemma-3-27b-it"
PAIRS_FILE  = "data/mechinterp_pairs_gemma3.json"
OUT_SUBDIR  = "head_ablation_topk"
K_VALUES    = [30, 50, 75, 100, 150, 200]   # K=1/3/5/10/20 already in ablation_topk_results.json
N_TOP_RANK  = 300


def main():
    set_run_context(MODEL, suffix="")
    out_dir = utils.RESULTS_DIR / OUT_SUBDIR
    npz_files = sorted(out_dir.glob("*_specificity.npz"))
    print(f"Found {len(npz_files)} cached specificity grids in {out_dir}")
    if not npz_files:
        raise SystemExit("No cached grids — run head_ablation_topk.py first.")

    # Load failure-pair ids from the family pair file
    pair_dicts = json.load(open(PAIRS_FILE))["pairs"]
    failure_ids = {p["id"] for p in pair_dicts if p.get("role") == "failure"}

    t0 = time.time()
    model, tokenizer = load_model(MODEL, device="auto")
    print(f"Model loaded in {time.time() - t0:.0f}s")

    pairs = build_prompt_pairs(tokenizer, pairs_file=PAIRS_FILE)
    pairs = [p for p in pairs if p.id in failure_ids]
    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    H = get_decoder_handles(model)
    in_dev = H.embed.weight.device
    print(f"Failure pairs: {len(pairs)}\n" + "=" * 60)

    summary_path = out_dir / "ablation_extended_K_results.json"
    results: dict[str, dict] = {}
    if summary_path.exists():
        results = json.load(open(summary_path))
        print(f"Resuming — {len(results)} pairs already processed")

    pairs_to_run = [p for p in pairs if p.id not in results]
    print(f"Running {len(pairs_to_run)} pairs at K = {K_VALUES}")

    for i, pair in enumerate(pairs_to_run, 1):
        t_pair = time.time()
        npz_path = out_dir / f"{pair.id}_specificity.npz"
        if not npz_path.exists():
            print(f"[{i}/{len(pairs_to_run)}] {pair.id}: SKIP (no .npz)"); continue

        print(f"\n[{i}/{len(pairs_to_run)}] {pair.id}")
        with np.load(npz_path) as zf:
            specificity = zf["specificity"]   # (n_layers, n_heads)
        n_layers, n_heads = specificity.shape

        # Rank top-N heads by specificity (no need to re-compute single-head ablations)
        flat = []
        for li in range(n_layers):
            for hi in range(n_heads):
                flat.append((float(specificity[li, hi]), li, hi))
        flat.sort(reverse=True)   # specificity descending
        head_list = [(li, hi) for spec, li, hi in flat[:N_TOP_RANK]]

        # Baseline logit-diff (no ablation)
        s_inputs = tokenizer(pair.stereotyped_prompt, return_tensors="pt").to(in_dev)
        with torch.no_grad():
            base_ld = get_logit_diff(model(**s_inputs).logits, yes_ids, no_ids)
        base_ans = "yes" if base_ld > 0 else "no"

        recoveries = {}
        for k in K_VALUES:
            if k > len(head_list): continue
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
            print(f"  top-{k:>3}: ans='{new_ans}'  ld={new_ld:+.3f}  "
                  f"{'✓' if recovered else '✗'}")

        results[pair.id] = {
            "base_answer":         base_ans,
            "base_logit_diff":     float(base_ld),
            "multi_head_recovery": recoveries,
        }
        # Atomic incremental save
        tmp = summary_path.with_suffix(".json.tmp")
        json.dump(results, open(tmp, "w"), indent=2)
        tmp.replace(summary_path)
        print(f"  pair elapsed: {time.time() - t_pair:.0f}s")

    # Final recovery summary
    print("\n" + "=" * 60)
    print("Recovery rate by K (failure pairs):")
    print(f"{'K':>5}  {'recovered':>10}  {'%':>6}")
    print("-" * 30)
    for k in K_VALUES:
        recs = [r["multi_head_recovery"].get(str(k), {}).get("recovered")
                for r in results.values()]
        recs = [v for v in recs if v is not None]
        if recs:
            print(f"{k:>5}  {sum(recs):>4}/{len(recs):<5}  {100*sum(recs)/len(recs):>5.0f}%")


if __name__ == "__main__":
    main()
