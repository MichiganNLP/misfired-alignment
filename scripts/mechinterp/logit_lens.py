"""
Logit lens analysis: track yes/no logit difference across layers.

The logit lens applies the final LayerNorm + unembedding to the residual stream
after each transformer layer. This shows how the model's "current best guess"
evolves as information flows through the network.

Hypothesis: if there are two competing circuits, we expect to see:
  - Stereotyped prompt: yes-logit starts positive (evidence circuit fires early),
    then drops at some intermediate layer (alignment circuit overrides).
  - Contrast prompt: yes-logit stays positive throughout.

The layer where divergence first appears identifies the alignment circuit's
approximate depth.

Usage:
  python scripts/mechinterp/logit_lens.py
  python scripts/mechinterp/logit_lens.py --example hiv_gay_straight
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import (
    build_prompt_pairs, get_yes_no_ids, load_model,
    get_logit_diff, get_answer, get_decoder_handles,
)


def compute_logit_lens(model, tokenizer, prompt: str, yes_id, no_id) -> list[float]:
    """
    Return yes_logit - no_logit at the last token position for each layer,
    using the final LayerNorm + unembedding as the projection head.
    """
    residuals: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # out[0] is the hidden state after this decoder layer
            h = out[0] if isinstance(out, tuple) else out
            residuals[idx] = h.detach()
        return hook_fn

    H = get_decoder_handles(model)
    hooks = [
        layer.register_forward_hook(make_hook(i))
        for i, layer in enumerate(H.layers)
    ]

    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    final_ln = H.norm
    lm_head = H.lm_head
    diffs = []

    with torch.no_grad():
        for i in range(H.n_layers):
            h = residuals[i]           # [1, seq_len, d_model]
            normed = final_ln(h.to(final_ln.weight.device))
            logits = lm_head(normed.to(lm_head.weight.device))   # [1, seq_len, vocab]
            diff = get_logit_diff(logits, yes_id, no_id)
            diffs.append(diff)

    return diffs


def run_logit_lens(model, tokenizer, pairs, example_ids=None):
    yes_id, no_id = get_yes_no_ids(tokenizer)
    print(f"  yes token ids: {yes_id}")
    print(f"  no  token ids: {no_id}")

    figs_dir = utils.RESULTS_DIR / "logit_lens"
    figs_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        if example_ids and pair.id not in example_ids:
            continue

        print(f"\n[{pair.id}] Computing logit lens ...")

        # Verify the model actually shows the failure we expect
        in_dev = get_decoder_handles(model).embed.weight.device
        inputs_s = tokenizer(pair.stereotyped_prompt, return_tensors="pt").to(in_dev)
        inputs_c = tokenizer(pair.contrast_prompt, return_tensors="pt").to(in_dev)
        with torch.no_grad():
            out_s = model(**inputs_s)
            out_c = model(**inputs_c)
        ans_s = get_answer(out_s.logits, yes_id, no_id)
        ans_c = get_answer(out_c.logits, yes_id, no_id)
        ld_s = get_logit_diff(out_s.logits, yes_id, no_id)
        ld_c = get_logit_diff(out_c.logits, yes_id, no_id)
        print(f"  Stereotyped ({pair.stereotyped_group}): {ans_s}  (logit diff {ld_s:+.2f})")
        print(f"  Contrast    ({pair.contrast_group}):   {ans_c}  (logit diff {ld_c:+.2f})")

        diffs_s = compute_logit_lens(model, tokenizer, pair.stereotyped_prompt, yes_id, no_id)
        diffs_c = compute_logit_lens(model, tokenizer, pair.contrast_prompt, yes_id, no_id)

        layers = list(range(len(diffs_s)))

        # Find first divergence layer (where the two lines cross zero in different directions)
        divergence_layer = None
        for i in range(len(layers) - 1):
            if (diffs_s[i] > 0) != (diffs_c[i] > 0):
                divergence_layer = i
                break
        if divergence_layer is None:
            # No crossing — find layer of maximum gap
            gaps = [abs(diffs_c[i] - diffs_s[i]) for i in layers]
            divergence_layer = gaps.index(max(gaps))

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(layers, diffs_s, color="tomato", linewidth=1.8,
                label=f"Stereotyped ({pair.stereotyped_group}): final answer = '{ans_s}'")
        ax.plot(layers, diffs_c, color="steelblue", linewidth=1.8,
                label=f"Contrast ({pair.contrast_group}): final answer = '{ans_c}'")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(divergence_layer, color="purple", linewidth=1.2, linestyle=":",
                   alpha=0.7, label=f"Max divergence ≈ layer {divergence_layer}")
        ax.fill_between(layers, diffs_s, diffs_c, alpha=0.08, color="purple",
                        label="Divergence region")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Logit(yes) − Logit(no)", fontsize=12)
        ax.set_title(
            f"Logit Lens — {pair.id}\n"
            f"Stereotyped group: '{pair.stereotyped_group}' | Contrast: '{pair.contrast_group}'",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        out = figs_dir / f"{pair.id}_logit_lens.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()

        # Per-pair JSON for cross-pair aggregation
        with open(figs_dir / f"{pair.id}_logit_lens.json", "w") as f:
            json.dump({
                "pair_id": pair.id,
                "category": pair.category,
                "stereotyped_group": pair.stereotyped_group,
                "contrast_group": pair.contrast_group,
                "final_answer_stereo": ans_s,
                "final_answer_contrast": ans_c,
                "final_logit_diff_stereo": ld_s,
                "final_logit_diff_contrast": ld_c,
                "per_layer_logit_diff_stereo": diffs_s,
                "per_layer_logit_diff_contrast": diffs_c,
                "divergence_layer": divergence_layer,
            }, f, indent=2)

        print(f"  → Saved {out}")
        print(f"  Max divergence at layer {divergence_layer}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--example", nargs="+", help="Example IDs to analyze (default: all)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.device)
    pairs = build_prompt_pairs(tokenizer)
    run_logit_lens(model, tokenizer, pairs, example_ids=args.example)


if __name__ == "__main__":
    main()
