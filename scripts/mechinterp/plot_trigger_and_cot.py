"""
Two extra layer-profile figures for the mechinterp section:

  (1) Trigger vs no-trigger on Llama-3.1-8B (Instruct + Base, 2x2 panels).
      Same 60 pair_ids, only the prompt prefix differs ("It is not okay
      to assume…" present vs absent). Demonstrates that the alignment
      circuit fires intrinsically on stereotyped content but is amplified
      ~12x by the explicit trigger sentence.

  (2) Direct vs CoT-simple on Qwen3-8B (Instruct only, 1x2 panels).
      Same model, same pair set; only the chat template differs
      (enable_thinking=False vs True). The simple version probes the
      model BEFORE it generates a reasoning trace.

Reuses the panel renderer + aggregator from plot_cross_family.
Writes per-panel PDFs for individual adjustment, plus combined PNGs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_cross_family import (
    DATA, MECH, load_role_map, aggregate_one, plot_one_panel
)

OUT_DIR  = MECH / "aggregate"
PER_DIR  = OUT_DIR / "per_model_layer_profile"
PER_DIR.mkdir(parents=True, exist_ok=True)


# ── Configuration: trigger vs no-trigger on Llama ────────────────────────────

TRIGGER_NOTRIGGER = [
    # (label, model-or-dir-key, pairs_file)
    # The "model" here is used both as the title source and as the
    # MECH/<safe(model)> directory name; for the no-trigger variants we
    # use the *_notrigger output dirs which contain the same pair_ids
    # but use prompt_pairs_bbq.json prompts.
    [
        ("Llama-3.1-8B-Instruct (trigger)",
         "meta-llama/Llama-3.1-8B-Instruct",
         "mechinterp_pairs.json"),
        ("Llama-3.1-8B-Instruct (no trigger)",
         "meta-llama/Llama-3.1-8B-Instruct_notrigger",
         "mechinterp_pairs_notrigger.json"),
    ],
    [
        ("Llama-3.1-8B base (trigger)",
         "meta-llama/Llama-3.1-8B",
         "mechinterp_pairs.json"),
        ("Llama-3.1-8B base (no trigger)",
         "meta-llama/Llama-3.1-8B_notrigger",
         "mechinterp_pairs_notrigger.json"),
    ],
]


# ── Configuration: direct vs CoT-simple on Qwen3-8B ──────────────────────────

DIRECT_COT = [
    ("Qwen3-8B Instruct (direct)",
     "Qwen/Qwen3-8B",
     "mechinterp_pairs_Qwen_Qwen3-8B.json"),
    ("Qwen3-8B Instruct (CoT, enable_thinking=True)",
     "Qwen/Qwen3-8B_cot",
     "mechinterp_pairs_Qwen_Qwen3-8B.json"),
]


def render_grid(rows, out_combined: Path, suptitle: str,
                 per_model_pdf: bool = True):
    """Render a list-of-rows configuration. Each row is a list of
    (label, model-key, pairs_file). Y-axis is shared per row."""
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)

    # Pass 1: per-row y-limits + cache aggregations
    cache = []
    for row in rows:
        ymin, ymax = 0.0, 0.0
        row_aggs = []
        for label, key, pairs_file in row:
            rm = load_role_map(DATA / pairs_file)
            agg = aggregate_one(key, rm)
            row_aggs.append((label, key, agg))
            if agg:
                for role in ("failure", "control"):
                    if role in agg:
                        ymax = max(ymax, float(np.nanmax(agg[role]["gap"] + agg[role]["sem"])))
                        ymin = min(ymin, float(np.nanmin(agg[role]["gap"] - agg[role]["sem"])))
        cache.append((ymin, ymax, row_aggs))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6.5 * n_cols, 3.6 * n_rows),
                              sharex=False)
    if n_rows == 1: axes = np.array([axes])
    if n_cols == 1: axes = axes.reshape(n_rows, 1)

    for r, (ymin, ymax, row_aggs) in enumerate(cache):
        for c, (label, key, agg) in enumerate(row_aggs):
            plot_one_panel(axes[r, c], agg, key, label,
                           ymin, ymax, show_legend=(r == 0 and c == 0))
            # Use the explicit label (e.g. "trigger" / "no trigger") as title
            axes[r, c].set_title(label, fontsize=11)

    fig.suptitle(suptitle, fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_combined, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote combined figure: {out_combined}")

    # Per-panel PDFs
    if per_model_pdf:
        for r, (ymin, ymax, row_aggs) in enumerate(cache):
            for label, key, agg in row_aggs:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                plot_one_panel(ax, agg, key, label, ymin, ymax,
                               show_legend=True)
                ax.set_title(label, fontsize=11)
                plt.tight_layout()
                stem = key.replace("/", "_")
                out = PER_DIR / f"{stem}_layer_profile.pdf"
                plt.savefig(out, dpi=160, bbox_inches="tight")
                plt.close()
                print(f"  per-panel: {out}")


def main():
    render_grid(
        TRIGGER_NOTRIGGER,
        out_combined=OUT_DIR / "trigger_vs_notrigger_layer_profile.png",
        suptitle="Trigger vs no-trigger on Llama-3.1-8B "
                 "(same 60 pair_ids, only prompt prefix differs)",
    )
    print()
    render_grid(
        [DIRECT_COT],
        out_combined=OUT_DIR / "direct_vs_cot_layer_profile.png",
        suptitle="Direct vs CoT-simple on Qwen3-8B Instruct "
                 "(same model, enable_thinking toggled in chat template)",
    )


if __name__ == "__main__":
    main()
