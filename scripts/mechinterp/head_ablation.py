"""
Head ablation experiment: directly tests the "two circuits" hypothesis.

Hypothesis: specific attention heads form the "alignment circuit" that suppresses
the correct factual answer when a stereotyped group is named. Ablating (zeroing out)
these heads on stereotyped prompts should recover the correct "yes" answer.

Experiments:
  1. Individual head ablation sweep: for each head, zero its output and measure
     the effect on logit(yes) - logit(no) for BOTH conditions.
     - Heads that increase the answer toward "yes" on the stereotyped prompt
       but have little effect on the contrast prompt → alignment circuit heads.
     - Heads that decrease toward "no" on both → general factual reasoning heads.

  2. Targeted ablation of top-N alignment heads: ablate the heads identified in
     experiment 1 together and verify recovery of correct answer.

  3. Cross-condition specificity: for each head, compute:
       specificity = Δlogit_stereo - Δlogit_contrast
     High specificity → head has a group-conditioned suppression effect.

Usage:
  python scripts/mechinterp/head_ablation.py
  python scripts/mechinterp/head_ablation.py --example hiv_gay_straight --top-n 10
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
    build_prompt_pairs, get_logit_diff,
    get_yes_no_ids, get_answer, load_model, get_decoder_handles,
)


def ablate_head(model, inputs, layer_idx: int, head_idx: int, head_dim: int) -> torch.Tensor:
    """Zero out one attention head's contribution to the output at all positions."""
    start = head_idx * head_dim
    end = start + head_dim

    def hook_fn(module, inp, out):
        patched = out[0].clone()
        patched[:, :, start:end] = 0.0
        if isinstance(out, tuple):
            return (patched,) + out[1:]
        return patched

    H = get_decoder_handles(model)
    hook = H.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(**inputs)
    hook.remove()
    return out.logits


def head_ablation_sweep(
    model,
    tokenizer,
    stereo_prompt: str,
    contrast_prompt: str,
    yes_id,
    no_id,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (layer, head), ablate and measure Δlogit_diff for both prompts.
    Returns (delta_stereo, delta_contrast, specificity) each [n_layers, n_heads].
    """
    H = get_decoder_handles(model)
    n_layers = H.n_layers
    n_heads = H.n_heads
    head_dim = H.head_dim
    in_dev = H.embed.weight.device

    s_inputs = tokenizer(stereo_prompt, return_tensors="pt").to(in_dev)
    c_inputs = tokenizer(contrast_prompt, return_tensors="pt").to(in_dev)

    with torch.no_grad():
        base_ld_s = get_logit_diff(model(**s_inputs).logits, yes_id, no_id)
        base_ld_c = get_logit_diff(model(**c_inputs).logits, yes_id, no_id)

    print(f"  Baseline logit diff — stereotyped: {base_ld_s:+.3f}, contrast: {base_ld_c:+.3f}")

    delta_s = np.zeros((n_layers, n_heads))
    delta_c = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            ablated_s = ablate_head(model, s_inputs, layer, head, head_dim)
            ablated_c = ablate_head(model, c_inputs, layer, head, head_dim)
            delta_s[layer, head] = get_logit_diff(ablated_s, yes_id, no_id) - base_ld_s
            delta_c[layer, head] = get_logit_diff(ablated_c, yes_id, no_id) - base_ld_c

    # Specificity: ablating this head helps the stereotyped case more than the contrast
    # Positive specificity → alignment head (its suppression is group-conditioned)
    specificity = delta_s - delta_c

    return delta_s, delta_c, specificity


def multi_head_ablation(
    model,
    tokenizer,
    stereo_prompt: str,
    yes_id,
    no_id,
    head_list: list[tuple[int, int]],
) -> float:
    """Ablate multiple heads simultaneously; return the new logit diff."""
    H = get_decoder_handles(model)
    head_dim = H.head_dim
    # Group by layer
    by_layer: dict[int, list[int]] = {}
    for layer, head in head_list:
        by_layer.setdefault(layer, []).append(head)

    def make_multi_hook(heads):
        def hook_fn(module, inp, out):
            patched = out[0].clone()
            for h in heads:
                s = h * head_dim
                e = s + head_dim
                patched[:, :, s:e] = 0.0
            return (patched,) + out[1:]
        return hook_fn

    hooks = []
    for layer, heads in by_layer.items():
        hooks.append(
            H.layers[layer].self_attn.register_forward_hook(make_multi_hook(heads))
        )

    inputs = tokenizer(stereo_prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks:
        h.remove()

    return get_logit_diff(out.logits, yes_id, no_id)


def run_head_ablation(model, tokenizer, pairs, example_ids=None, top_n=10):
    yes_id, no_id = get_yes_no_ids(tokenizer)
    out_dir = utils.RESULTS_DIR / "head_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for pair in pairs:
        if example_ids and pair.id not in example_ids:
            continue
        print(f"\n[{pair.id}] Head ablation sweep ...")

        delta_s, delta_c, specificity = head_ablation_sweep(
            model, tokenizer,
            pair.stereotyped_prompt, pair.contrast_prompt,
            yes_id, no_id,
        )

        n_layers, n_heads = specificity.shape

        # ── Plot ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(20, max(5, n_layers * 0.3)))
        for ax, data, title, cmap in zip(
            axes,
            [delta_s, delta_c, specificity],
            [
                f"Δlogit (stereotyped: '{pair.stereotyped_group}')\n+ve = ablation helps 'yes'",
                f"Δlogit (contrast: '{pair.contrast_group}')\n+ve = ablation helps 'yes'",
                "Specificity (Δstereo − Δcontrast)\n+ve = alignment head",
            ],
            ["RdBu", "RdBu", "PuOr"],
        ):
            vmax = max(abs(data).max(), 0.01)
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, shrink=0.6)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_title(title, fontsize=9)

        # Highlight top specificity heads
        flat_spec = specificity.flatten()
        top_idx = np.argpartition(flat_spec, -top_n)[-top_n:]
        for idx in top_idx:
            li, hi = divmod(idx, n_heads)
            axes[2].add_patch(plt.Rectangle(
                (hi - 0.5, li - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=1.5,
            ))

        fig.suptitle(
            f"Head Ablation — {pair.id} ({pair.category})\n"
            f"Alignment heads = those with high specificity (right panel)",
            fontsize=11,
        )
        plt.tight_layout()
        out = out_dir / f"{pair.id}_ablation_heatmap.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  → Saved {out}")

        # Top alignment heads
        top_heads = [
            (int(li), int(hi), float(specificity[li, hi]),
             float(delta_s[li, hi]), float(delta_c[li, hi]))
            for li, hi in (divmod(i, n_heads) for i in top_idx)
        ]
        top_heads.sort(key=lambda x: -x[2])
        print(f"  Top {top_n} alignment heads (layer, head, specificity, Δstereo, Δcontrast):")
        for li, hi, sp, ds, dc in top_heads:
            print(f"    L{li:2d} H{hi:2d}: specificity={sp:+.3f}  Δstereo={ds:+.3f}  Δcontrast={dc:+.3f}")

        # ── Multi-head ablation: does removing top-N recover correct answer? ──
        head_list = [(li, hi) for li, hi, *_ in top_heads]

        in_dev = get_decoder_handles(model).embed.weight.device
        inputs_s = tokenizer(pair.stereotyped_prompt, return_tensors="pt").to(in_dev)
        with torch.no_grad():
            base_ld = get_logit_diff(model(**inputs_s).logits, yes_id, no_id)
        base_ans = "yes" if base_ld > 0 else "no"

        print(f"\n  Multi-head ablation recovery (stereotyped prompt):")
        print(f"  Baseline: answer='{base_ans}', logit_diff={base_ld:+.3f}")
        for k in [1, 3, 5, top_n]:
            if k > len(head_list):
                continue
            new_ld = multi_head_ablation(model, tokenizer, pair.stereotyped_prompt,
                                          yes_id, no_id, head_list[:k])
            new_ans = "yes" if new_ld > 0 else "no"
            print(f"  Ablate top-{k}: answer='{new_ans}', logit_diff={new_ld:+.3f}  "
                  f"{'✓ RECOVERED' if new_ans == 'yes' else '✗'}")

        # Save results
        all_results[pair.id] = {
            "top_alignment_heads": [
                {"layer": li, "head": hi, "specificity": sp,
                 "delta_stereo": ds, "delta_contrast": dc}
                for li, hi, sp, ds, dc in top_heads
            ],
            "base_answer": base_ans,
            "base_logit_diff": base_ld,
        }

    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}/ablation_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example", nargs="+")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    pairs = build_prompt_pairs(tokenizer)
    run_head_ablation(model, tokenizer, pairs, example_ids=args.example, top_n=args.top_n)


if __name__ == "__main__":
    main()
