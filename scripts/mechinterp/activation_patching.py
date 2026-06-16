"""
Activation patching (causal tracing) to locate the alignment circuit.

Two experiments:
  A) Layer sweep at the FINAL TOKEN position:
     - Run "clean" (contrast) prompt, cache residual stream at each layer.
     - Run "corrupted" (stereotyped) prompt, patching in one layer at a time from clean.
     - Recovery score = how much does patching layer L restore the correct answer?
     - High recovery at layer L → that layer is where the alignment suppression happens.

  B) GROUP TOKEN position sweep:
     - Same as A, but patch only at the position of the group identity token
       (e.g., "gay" vs "straight", "women" vs "men").
     - If a specific layer's group token representation carries the alignment signal,
       patching there will restore the correct answer.
     - This directly tests whether the group identity token is the locus of the failure.

Interpretation:
  - If A shows recovery at early/mid layers → the suppression happens early; the
    alignment circuit acts on the answer from the start.
  - If B shows recovery at the same layers as A → the group token representation
    is the mechanism (the alignment signal flows through how the group token is
    represented, not through later integration).
  - If B shows no recovery even where A does → the suppression doesn't come from
    the group token alone; the alignment circuit integrates multiple signals.

Usage:
  python scripts/mechinterp/activation_patching.py
  python scripts/mechinterp/activation_patching.py --example hiv_gay_straight --mode final group
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import (
    build_prompt_pairs, find_group_token_position,
    get_logit_diff, get_yes_no_ids, load_model, get_decoder_handles,
)


# ── Core patching logic ───────────────────────────────────────────────────────

def cache_residuals(model, tokenizer, prompt: str) -> dict[int, torch.Tensor]:
    """Run prompt, return {layer_idx: residual_stream_output [1, seq, d]}."""
    cache: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[idx] = h.detach().clone()
        return fn

    H = get_decoder_handles(model)
    hooks = [l.register_forward_hook(make_hook(i))
             for i, l in enumerate(H.layers)]
    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()

    logits = out.logits
    return cache, inputs, logits


def patch_one_layer(
    model,
    corrupt_inputs,
    patch_layer: int,
    clean_cache: dict[int, torch.Tensor],
    patch_position: int,       # sequence position to patch (-1 = last)
    clean_seq_len: int,        # needed to map "last" correctly across seq lengths
) -> torch.Tensor:
    """Run corrupt forward pass with one layer's residual patched from clean at `patch_position`."""

    def make_hook(target_layer):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            src = clean_cache[target_layer]   # [1, clean_seq, d]

            # Resolve actual sequence indices
            corrupt_seq = h.shape[1]
            src_seq = src.shape[1]

            if patch_position == -1:
                # Align from the right (both prompts end with the same suffix)
                c_pos = corrupt_seq - 1
                s_pos = src_seq - 1
            else:
                c_pos = patch_position
                s_pos = patch_position
                if c_pos >= corrupt_seq or s_pos >= src_seq:
                    return out   # position out of range, skip

            patched_h = h.clone()
            patched_h[0, c_pos, :] = src[0, s_pos, :]

            if isinstance(out, tuple):
                return (patched_h,) + out[1:]
            return patched_h

        return fn

    H = get_decoder_handles(model)
    hook = H.layers[patch_layer].register_forward_hook(make_hook(patch_layer))
    with torch.no_grad():
        out = model(**corrupt_inputs)
    hook.remove()
    return out.logits


def layer_sweep_patching(
    model,
    tokenizer,
    clean_prompt: str,
    corrupt_prompt: str,
    yes_id,
    no_id,
    patch_position: int = -1,   # -1 = last (final) token
    position_label: str = "final token",
    group_pos_clean: int | None = None,
    group_pos_corrupt: int | None = None,
) -> tuple[list[float], float, float]:
    """
    Sweep over all layers, patching the residual stream from clean → corrupt.
    Returns (recovery_scores, clean_ld, corrupt_ld).
    """
    clean_cache, clean_inputs, clean_logits = cache_residuals(model, tokenizer, clean_prompt)
    corrupt_cache, corrupt_inputs, corrupt_logits = cache_residuals(model, tokenizer, corrupt_prompt)

    clean_ld = get_logit_diff(clean_logits, yes_id, no_id)
    corrupt_ld = get_logit_diff(corrupt_logits, yes_id, no_id)
    denom = clean_ld - corrupt_ld

    print(f"  Clean  logit diff (contrast):    {clean_ld:+.3f}")
    print(f"  Corrupt logit diff (stereotyped): {corrupt_ld:+.3f}")
    print(f"  Patching position: {position_label}")

    recovery = []
    n_layers = get_decoder_handles(model).n_layers

    for L in range(n_layers):
        # Determine which position to patch
        if patch_position == -1:
            pos = -1
        elif patch_position == "group" and group_pos_corrupt is not None:
            pos = group_pos_corrupt
        else:
            pos = patch_position

        patched_logits = patch_one_layer(
            model, corrupt_inputs, L, clean_cache, pos, clean_inputs.input_ids.shape[1]
        )
        patched_ld = get_logit_diff(patched_logits, yes_id, no_id)

        r = (patched_ld - corrupt_ld) / denom if abs(denom) > 1e-6 else 0.0
        recovery.append(r)

    return recovery, clean_ld, corrupt_ld


# ── Attention head patching ───────────────────────────────────────────────────

def head_patch_sweep(
    model,
    tokenizer,
    clean_prompt: str,
    corrupt_prompt: str,
    yes_id,
    no_id,
) -> np.ndarray:
    """
    For each (layer, head), patch that head's output from clean → corrupt at the final token.
    Returns a [n_layers, n_heads] array of recovery scores.
    """
    H = get_decoder_handles(model)
    n_layers = H.n_layers
    n_heads = H.n_heads
    head_dim = H.head_dim

    # Cache head outputs from clean run
    head_outputs: dict[int, torch.Tensor] = {}

    def make_attn_hook(layer_idx):
        def fn(module, inp, out):
            # out[0] is the attention output [1, seq, d_model]
            head_outputs[layer_idx] = out[0].detach().clone()
        return fn

    attn_hooks = [
        H.layers[i].self_attn.register_forward_hook(make_attn_hook(i))
        for i in range(n_layers)
    ]
    in_dev = H.embed.weight.device
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(in_dev)
    with torch.no_grad():
        _ = model(**clean_inputs)
    for h in attn_hooks:
        h.remove()

    corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(in_dev)
    with torch.no_grad():
        corrupt_logits = model(**corrupt_inputs).logits
    corrupt_ld = get_logit_diff(corrupt_logits, yes_id, no_id)

    clean_logits_out = None
    with torch.no_grad():
        clean_logits_out = model(**clean_inputs).logits
    clean_ld = get_logit_diff(clean_logits_out, yes_id, no_id)
    denom = clean_ld - corrupt_ld

    results = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        clean_attn_out = head_outputs[layer_idx]   # [1, seq, d_model]
        clean_seq = clean_attn_out.shape[1]

        for head_idx in range(n_heads):
            # Dimensions: d = head_idx * head_dim : (head_idx+1) * head_dim
            # We patch only those dimensions in the attention output at the last token
            start = head_idx * head_dim
            end = start + head_dim

            corrupt_seq_len = corrupt_inputs.input_ids.shape[1]

            def make_head_hook(li, hi, s, e, src, c_seq, src_seq):
                def fn(module, inp, out):
                    patched = out[0].clone()
                    # Align from right
                    c_pos = c_seq - 1
                    s_pos = src_seq - 1
                    patched[0, c_pos, s:e] = src[0, s_pos, s:e]
                    return (patched,) + out[1:]
                return fn

            hook = H.layers[layer_idx].self_attn.register_forward_hook(
                make_head_hook(layer_idx, head_idx, start, end,
                               clean_attn_out, corrupt_seq_len, clean_seq)
            )
            with torch.no_grad():
                patched_logits = model(**corrupt_inputs).logits
            hook.remove()

            patched_ld = get_logit_diff(patched_logits, yes_id, no_id)
            r = (patched_ld - corrupt_ld) / denom if abs(denom) > 1e-6 else 0.0
            results[layer_idx, head_idx] = r

    return results, clean_ld, corrupt_ld


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_layer_sweep(recovery_final, recovery_group, pair_id, stereo_group, contrast_group, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    layers = list(range(len(recovery_final)))

    for ax, recovery, title in zip(
        axes,
        [recovery_final, recovery_group],
        ["Patch at Final Token Position", "Patch at Group Identity Token Position"],
    ):
        colors = ["tomato" if r < 0 else "steelblue" for r in recovery]
        ax.bar(layers, recovery, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="Full recovery")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Recovery score")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Activation Patching — {pair_id}\n"
        f"Clean: '{contrast_group}' prompt → Corrupt: '{stereo_group}' prompt",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_head_heatmap(head_results, pair_id, stereo_group, contrast_group, out_path):
    n_layers, n_heads = head_results.shape
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.5), max(6, n_layers * 0.25)))
    im = ax.imshow(head_results, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Recovery score")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer")
    ax.set_title(
        f"Head-level Patching — {pair_id}\n"
        f"Clean: '{contrast_group}' → Corrupt: '{stereo_group}' (final token position)",
        fontsize=10,
    )
    # Mark top heads
    flat = head_results.flatten()
    top_k = min(5, len(flat))
    top_indices = np.argpartition(flat, -top_k)[-top_k:]
    for idx in top_indices:
        li, hi = divmod(idx, n_heads)
        ax.add_patch(plt.Rectangle((hi - 0.5, li - 0.5), 1, 1,
                                    fill=False, edgecolor="gold", linewidth=2))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_activation_patching(model, tokenizer, pairs, example_ids=None, modes=("final", "group", "heads")):
    yes_id, no_id = get_yes_no_ids(tokenizer)
    out_dir = utils.RESULTS_DIR / "activation_patching"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        if example_ids and pair.id not in example_ids:
            continue

        print(f"\n[{pair.id}] Activation patching ...")

        # Locate group identity tokens
        s_inputs = tokenizer(pair.stereotyped_prompt, return_tensors="pt")
        c_inputs = tokenizer(pair.contrast_prompt, return_tensors="pt")
        group_pos_s = find_group_token_position(s_inputs.input_ids, tokenizer, pair.stereotyped_group)
        group_pos_c = find_group_token_position(c_inputs.input_ids, tokenizer, pair.contrast_group)
        print(f"  Group token pos — stereotyped: {group_pos_s}, contrast: {group_pos_c}")

        recovery_final, clean_ld, corrupt_ld = None, None, None
        recovery_group = None

        if "final" in modes:
            print("  Running final-token layer sweep ...")
            recovery_final, clean_ld, corrupt_ld = layer_sweep_patching(
                model, tokenizer,
                pair.contrast_prompt,    # clean = contrast (correct)
                pair.stereotyped_prompt, # corrupt = stereotyped (wrong)
                yes_id, no_id,
                patch_position=-1,
                position_label="final token",
            )

        if "group" in modes and group_pos_s is not None and group_pos_c is not None:
            print("  Running group-token layer sweep ...")
            # For group patching we use corrupt_group_pos in the corrupt prompt
            recovery_group, _, _ = layer_sweep_patching(
                model, tokenizer,
                pair.contrast_prompt,
                pair.stereotyped_prompt,
                yes_id, no_id,
                patch_position=group_pos_s,
                position_label=f"group token (pos {group_pos_s})",
                group_pos_clean=group_pos_c,
                group_pos_corrupt=group_pos_s,
            )
        else:
            recovery_group = [0.0] * get_decoder_handles(model).n_layers

        if recovery_final is not None:
            plot_layer_sweep(
                recovery_final, recovery_group,
                pair.id, pair.stereotyped_group, pair.contrast_group,
                out_dir / f"{pair.id}_layer_sweep.png",
            )
            print(f"  → Saved layer sweep plot")
            top_final = int(np.argmax(recovery_final))
            top_group = int(np.argmax(recovery_group)) if recovery_group else None
            print(f"  Top recovery layer (final token): {top_final}  ({recovery_final[top_final]:.2f})")
            if top_group is not None:
                print(f"  Top recovery layer (group token): {top_group}  ({recovery_group[top_group]:.2f})")

            # Per-pair JSON for cross-pair aggregation
            with open(out_dir / f"{pair.id}_layer_sweep.json", "w") as f:
                json.dump({
                    "pair_id": pair.id,
                    "category": pair.category,
                    "stereotyped_group": pair.stereotyped_group,
                    "contrast_group": pair.contrast_group,
                    "clean_logit_diff": float(clean_ld),
                    "corrupt_logit_diff": float(corrupt_ld),
                    "recovery_per_layer_final": [float(x) for x in recovery_final],
                    "recovery_per_layer_group": [float(x) for x in (recovery_group or [])],
                    "group_pos_stereotyped": int(group_pos_s) if group_pos_s is not None else None,
                    "group_pos_contrast": int(group_pos_c) if group_pos_c is not None else None,
                }, f, indent=2)

        if "heads" in modes:
            print("  Running head-level sweep (this takes a while) ...")
            head_results, _, _ = head_patch_sweep(
                model, tokenizer,
                pair.contrast_prompt,
                pair.stereotyped_prompt,
                yes_id, no_id,
            )
            plot_head_heatmap(
                head_results, pair.id,
                pair.stereotyped_group, pair.contrast_group,
                out_dir / f"{pair.id}_head_heatmap.png",
            )
            top_heads = np.dstack(np.unravel_index(
                np.argsort(head_results.flatten())[-5:], head_results.shape
            ))[0][::-1]
            print(f"  Top 5 heads (layer, head): {[(int(l), int(h)) for l, h in top_heads]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example", nargs="+")
    parser.add_argument("--mode", nargs="+", default=["final", "group", "heads"],
                        choices=["final", "group", "heads"])
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    pairs = build_prompt_pairs(tokenizer)
    run_activation_patching(model, tokenizer, pairs, example_ids=args.example, modes=args.mode)


if __name__ == "__main__":
    main()
