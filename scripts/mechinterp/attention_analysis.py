"""
Attention pattern analysis: does the model attend differently to the group
identity token depending on which group is named?

For each attention head at each layer, we measure:
  A) attention_to_group[layer, head] = attention weight from the final token
     to the group identity token (e.g., "gay" vs "straight").
  B) The difference (stereotyped - contrast) highlights heads that respond
     specifically to the stereotyped group label.

If the alignment circuit hypothesis is right, we expect specific heads in
specific layers to attend MORE to the group token in the stereotyped condition.
These are the candidate "alignment heads" — they detect the group identity and
suppress the factual reasoning signal.

We also visualize the full attention pattern (all source → final token) for
a qualitative view of what the model is "looking at" when forming its answer.

Usage:
  python scripts/mechinterp/attention_analysis.py
  python scripts/mechinterp/attention_analysis.py --example hiv_gay_straight
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import (
    build_prompt_pairs, find_group_token_position,
    get_yes_no_ids, get_answer, load_model, get_decoder_handles,
)


def get_attention_weights(model, tokenizer, prompt: str) -> list[torch.Tensor]:
    """
    Run model with output_attentions=True.
    Returns list of [n_heads, seq, seq] tensors, one per layer.
    (averaged over GQA groups to give n_heads=32 for Llama-3.1-8B)
    """
    H = get_decoder_handles(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    # out.attentions: tuple of [1, n_kv_heads, seq, seq] per layer
    # For GQA models, n_kv_heads < n_heads; we tile to match query heads
    attn_patterns = []
    n_heads = H.n_heads
    for layer_attn in out.attentions:
        a = layer_attn[0]  # [n_kv_heads, seq, seq]
        n_kv = a.shape[0]
        if n_kv < n_heads:
            # Repeat KV groups to match query heads
            reps = n_heads // n_kv
            a = a.repeat_interleave(reps, dim=0)  # [n_heads, seq, seq]
        attn_patterns.append(a.cpu())
    return attn_patterns, inputs.input_ids


def compute_group_attention(
    attn_patterns: list[torch.Tensor],
    group_pos: int,
) -> np.ndarray:
    """
    For each (layer, head), return attention weight FROM the last token TO group_pos.
    Returns [n_layers, n_heads].
    """
    n_layers = len(attn_patterns)
    n_heads = attn_patterns[0].shape[0]
    result = np.zeros((n_layers, n_heads))
    for l, a in enumerate(attn_patterns):
        # a: [n_heads, seq, seq]; a[h, -1, pos] = attention from last token to pos
        result[l, :] = a[:, -1, group_pos].float().numpy()
    return result


def run_attention_analysis(model, tokenizer, pairs, example_ids=None):
    yes_id, no_id = get_yes_no_ids(tokenizer)
    out_dir = utils.RESULTS_DIR / "attention_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        if example_ids and pair.id not in example_ids:
            continue
        print(f"\n[{pair.id}] Attention analysis ...")

        # Get attention patterns for both conditions
        attn_s, ids_s = get_attention_weights(model, tokenizer, pair.stereotyped_prompt)
        attn_c, ids_c = get_attention_weights(model, tokenizer, pair.contrast_prompt)

        # Find group token positions
        group_pos_s = find_group_token_position(ids_s, tokenizer, pair.stereotyped_group)
        group_pos_c = find_group_token_position(ids_c, tokenizer, pair.contrast_group)
        print(f"  Group pos — stereo: {group_pos_s} ('{pair.stereotyped_group}'), "
              f"contrast: {group_pos_c} ('{pair.contrast_group}')")

        if group_pos_s is None or group_pos_c is None:
            print("  WARNING: could not locate group token, skipping")
            continue

        # Attention to group token from final token
        attn_to_group_s = compute_group_attention(attn_s, group_pos_s)   # [L, H]
        attn_to_group_c = compute_group_attention(attn_c, group_pos_c)   # [L, H]
        diff = attn_to_group_s - attn_to_group_c                         # positive = stereo > contrast

        # ── Plot 1: attention difference heatmap ──────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(attn_s) * 0.3)))
        for ax, data, title, cmap, vlim in zip(
            axes,
            [attn_to_group_s, attn_to_group_c, diff],
            [
                f"Attn → '{pair.stereotyped_group}' token",
                f"Attn → '{pair.contrast_group}' token",
                "Difference (stereo − contrast)",
            ],
            ["Reds", "Blues", "RdBu_r"],
            [(0, None), (0, None), (-0.1, 0.1)],
        ):
            vmin, vmax = vlim
            im = ax.imshow(data, aspect="auto", cmap=cmap,
                           vmin=vmin if vmin else data.min(),
                           vmax=vmax if vmax else data.max())
            plt.colorbar(im, ax=ax, shrink=0.7)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_title(title, fontsize=9)

        # Highlight top differential heads
        top_k = 5
        flat_diff = diff.flatten()
        top_indices = np.argpartition(flat_diff, -top_k)[-top_k:]
        for idx in top_indices:
            li, hi = divmod(idx, diff.shape[1])
            axes[2].add_patch(plt.Rectangle(
                (hi - 0.5, li - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1.5,
            ))

        fig.suptitle(
            f"Attention to Group Token — {pair.id} ({pair.category})\n"
            f"Final token attends to group identity token",
            fontsize=11,
        )
        plt.tight_layout()
        out = out_dir / f"{pair.id}_group_attn_heatmap.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  → Saved {out}")

        # Per-pair JSON for cross-pair aggregation
        with open(out_dir / f"{pair.id}_group_attn.json", "w") as f:
            json.dump({
                "pair_id": pair.id,
                "category": pair.category,
                "stereotyped_group": pair.stereotyped_group,
                "contrast_group": pair.contrast_group,
                "group_pos_stereotyped": int(group_pos_s),
                "group_pos_contrast": int(group_pos_c),
                "attn_to_group_stereo":   attn_to_group_s.tolist(),  # [L,H]
                "attn_to_group_contrast": attn_to_group_c.tolist(),  # [L,H]
                "attn_diff_stereo_minus_contrast": diff.tolist(),    # [L,H]
            }, f)

        # Report top differential heads
        top_heads = [(int(li), int(hi), diff[li, hi])
                     for li, hi in (divmod(i, diff.shape[1]) for i in top_indices)]
        top_heads.sort(key=lambda x: -x[2])
        print(f"  Top differential heads (stereo > contrast attention to group token):")
        for li, hi, d in top_heads:
            print(f"    Layer {li:2d}, Head {hi:2d}: diff = {d:+.4f}  "
                  f"(stereo: {attn_to_group_s[li,hi]:.4f}, contrast: {attn_to_group_c[li,hi]:.4f})")

        # ── Plot 2: full attention pattern for last token (selected layers) ──
        # Show the top differential layer's full attention pattern from final token
        top_layer = top_heads[0][0]
        seq_len_s = attn_s[0].shape[-1]
        tokens_s = [tokenizer.decode([t]) for t in ids_s[0].tolist()]

        # Truncate for readability
        max_display = 40
        display_len = min(seq_len_s, max_display)
        display_tokens = tokens_s[-display_len:]

        n_heads = attn_s[top_layer].shape[0]
        fig, axes = plt.subplots(4, 8, figsize=(20, 10), sharey=True)
        axes = axes.flatten()

        for h in range(min(n_heads, len(axes))):
            ax = axes[h]
            a_s = attn_s[top_layer][h, -1, -display_len:].float().numpy()
            a_c = attn_c[top_layer][h, -1, -display_len:].float().numpy()
            x = np.arange(display_len)
            ax.bar(x, a_s, alpha=0.6, color="tomato", label="stereo")
            ax.bar(x, a_c, alpha=0.6, color="steelblue", label="contrast")
            ax.set_title(f"H{h}", fontsize=7)
            ax.set_xticks([])
            ax.tick_params(labelsize=6)
            if h == 0:
                ax.legend(fontsize=6)

        fig.suptitle(
            f"Layer {top_layer} — Final Token Attention Distribution\n"
            f"Last {display_len} tokens of '{pair.stereotyped_group}' (red) vs '{pair.contrast_group}' (blue) prompt",
            fontsize=10,
        )
        plt.tight_layout()
        out2 = out_dir / f"{pair.id}_layer{top_layer}_attn_pattern.png"
        plt.savefig(out2, dpi=120)
        plt.close()
        print(f"  → Saved {out2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example", nargs="+")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    pairs = build_prompt_pairs(tokenizer)
    run_attention_analysis(model, tokenizer, pairs, example_ids=args.example)


if __name__ == "__main__":
    main()
