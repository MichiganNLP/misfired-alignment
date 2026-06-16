"""
Late residual substitution (a path-patching variant for hypothesis B).

For each failure pair on the STEREOTYPED prompt:
  1. Find L_max  = layer where logit-lens diff is most positive ("factual peak").
  2. For each L_target in {L_max+1, ..., final_layer}, run the model and
     replace the L_target residual at the final-token position with the
     L_max residual (from the same forward pass). Continue forward.
  3. Record the final logit_diff.

Interpretation:
  - If substitution at L_target flips the final answer to "yes", L_target is
    downstream of where the factual circuit reached the correct answer and
    upstream of (or coincident with) where the suppression vetoed it.
  - The earliest L_target that produces "yes" identifies the boundary at which
    re-injecting the factual residual is sufficient to escape the suppressor.

This is a direct test of hypothesis B-yes (factual circuit intact, suppression
acts late and is bypassable by injecting the early signal forward).

Usage:
  python scripts/mechinterp/path_patching.py
  python scripts/mechinterp/path_patching.py --model meta-llama/Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import (
    build_prompt_pairs, get_logit_diff, get_yes_no_ids,
    load_model, set_run_context, get_decoder_handles,
)


def cache_all_residuals(model, tokenizer, prompt: str):
    """Run model on `prompt`, cache the residual stream after each decoder
    layer. Returns (residuals, logits) where residuals is a dict {layer: tensor}."""
    cached = {}

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cached[idx] = h.detach().clone()
        return hook

    H = get_decoder_handles(model)
    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(H.layers)]
    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()
    return cached, out.logits, inputs


def patched_forward(model, tokenizer, prompt: str, target_layer: int,
                    replacement_residual: torch.Tensor) -> torch.Tensor:
    """Run forward; at target_layer's output, overwrite the FINAL-TOKEN
    residual with replacement_residual (only at the final position, only at
    the stated layer). Return the final logits."""

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # h: [1, seq, d_model]; replacement_residual is from the same prompt's
        # earlier layer at every position. Take final-token slice only.
        h[..., -1, :] = replacement_residual[..., -1, :]
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    H = get_decoder_handles(model)
    target = H.layers[target_layer]
    handle = target.register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs)
    handle.remove()
    return out.logits


def find_peak_pos_layer(per_layer_diffs: list[float]) -> int:
    """Layer index where logit_diff is most positive (factual peak)."""
    return int(np.argmax(per_layer_diffs))


def run_path_patching(model, tokenizer, pairs, example_ids=None,
                       only_role: str | None = "failure",
                       pairs_file: str | None = None):
    """For each (failure-)pair, do late residual substitution and report
    the recovery curve over target layers. `pairs_file` lets us look up
    role labels from the same JSON the prompts were loaded from (otherwise
    `load_pair_dicts()` would fall back to the default Llama set)."""
    yes_id, no_id = get_yes_no_ids(tokenizer)
    out_dir = utils.RESULTS_DIR / "path_patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    H = get_decoder_handles(model)
    n_layers = H.n_layers

    # Recover role labels keyed by pair_id from the SAME file used to build
    # the prompt pairs (so cross-family runs don't fall back to default).
    pair_dicts = {p["id"]: p for p in utils.load_pair_dicts(pairs_file)}

    all_results = {}
    role_curves = defaultdict(list)  # role -> list of [logit_diff per L_target]

    for pair in pairs:
        if example_ids and pair.id not in example_ids:
            continue
        role = pair_dicts.get(pair.id, {}).get("role", "unknown")
        if only_role and role != only_role:
            continue

        print(f"\n[{pair.id}] role={role}  Late residual substitution ...")

        # Cache all residuals on stereo prompt
        residuals, logits_clean, _ = cache_all_residuals(model, tokenizer, pair.stereotyped_prompt)
        baseline_diff = get_logit_diff(logits_clean, yes_id, no_id)

        # Per-layer logit lens to find L_max
        per_layer_diffs = []
        with torch.no_grad():
            for L in range(n_layers):
                projected = H.lm_head(H.norm(residuals[L].to(H.norm.weight.device)))
                per_layer_diffs.append(get_logit_diff(projected, yes_id, no_id))
        L_max = find_peak_pos_layer(per_layer_diffs)
        L_max_diff = per_layer_diffs[L_max]

        # Sweep L_target = L_max+1 ... n_layers-1
        sweep = []
        flip_layer = None
        for L_target in range(L_max + 1, n_layers):
            patched_logits = patched_forward(
                model, tokenizer, pair.stereotyped_prompt,
                target_layer=L_target,
                replacement_residual=residuals[L_max],
            )
            d = get_logit_diff(patched_logits, yes_id, no_id)
            sweep.append((L_target, d))
            if flip_layer is None and d > 0 and baseline_diff < 0:
                flip_layer = L_target

        # Pad sweep to full layer range with NaN for missing entries (so
        # cross-pair averaging works by aligning on absolute layer index).
        sweep_padded = [np.nan] * n_layers
        for L_target, d in sweep:
            sweep_padded[L_target] = d
        role_curves[role].append(sweep_padded)

        flip_str = f"flips→yes at L{flip_layer}" if flip_layer is not None else "never flips"
        print(f"  baseline diff={baseline_diff:+.2f}  L_max=L{L_max} (diff={L_max_diff:+.2f})  {flip_str}")

        all_results[pair.id] = {
            "role": role,
            "baseline_diff": float(baseline_diff),
            "L_max": int(L_max),
            "L_max_diff": float(L_max_diff),
            "sweep": [[L, float(d)] for L, d in sweep],
            "first_flip_layer": flip_layer,
        }

    with open(out_dir / "path_patching_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved per-pair results: {out_dir}/path_patching_results.json")

    # ── Aggregate: mean curve, flip-rate at each L_target ──
    n_total = len(all_results)
    if n_total == 0:
        return
    flips = sum(1 for r in all_results.values() if r["first_flip_layer"] is not None)
    print(f"\nSubstitution recovery (across {n_total} pairs):")
    print(f"  Pairs with at least one L_target flipping wrong→right: {flips}/{n_total} = {flips/n_total:.0%}")

    # Plot mean recovery curve
    if role_curves:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        x = np.arange(n_layers)
        for role, curves in role_curves.items():
            arr = np.array(curves, dtype=float)  # [N, L]
            mean = np.nanmean(arr, axis=0)
            color = "purple" if role == "failure" else "gray"
            ax.plot(x, mean, color=color, lw=2,
                    label=f"{role} (n={arr.shape[0]})")
            n_per_layer = np.sum(~np.isnan(arr), axis=0)
            sem = np.where(n_per_layer >= 2,
                            np.nanstd(arr, axis=0, ddof=1) /
                            np.sqrt(np.maximum(n_per_layer, 1)),
                            np.nan)
            ax.fill_between(x, mean - sem, mean + sem, alpha=0.2, color=color)
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Substitution layer L_target")
        ax.set_ylabel("Final logit-diff (yes − no) after substitution")
        ax.set_title("Late residual substitution: replacing L_target with L_max residual")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "path_patching_recovery.png", dpi=150)
        plt.close()
        print(f"Saved recovery plot: {out_dir}/path_patching_recovery.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--pairs_file", default=None)
    parser.add_argument("--example", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--role", default="failure",
                        choices=["failure", "control", "all"],
                        help="Restrict to pairs of this role")
    args = parser.parse_args()

    set_run_context(args.model)
    model, tokenizer = load_model(args.model, args.device, tokenizer_name=args.tokenizer)
    pairs = build_prompt_pairs(tokenizer, pairs_file=args.pairs_file)
    role_filter = None if args.role == "all" else args.role
    run_path_patching(model, tokenizer, pairs,
                      example_ids=args.example, only_role=role_filter,
                      pairs_file=args.pairs_file)


if __name__ == "__main__":
    main()
