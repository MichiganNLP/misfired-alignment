"""
Plot the two asymmetric off-diagonal cells per model:
  - Stereo Wrong + Contrast Correct  (MAR — overalignment failure)
  - Stereo Correct + Contrast Wrong  (reverse — model disfavours contrast group)

Outputs:
  results/figures/asymmetry/        <- per-model PNGs
  results/figures/asymmetry_grid.png  <- all models in one figure
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures" / "asymmetry"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DISPLAY = {
    "openai_gpt-5.4-20260305": "GPT-5.4",
    "openai_gpt-5.4-mini-20260317": "GPT-5.4-mini",
    "openai_gpt-5.4-nano-20260317": "GPT-5.4-nano",
    "anthropic_claude-4.7-opus-20260416": "Claude-4.7-Opus",
    "anthropic_claude-4.6-sonnet-20260217": "Claude-4.6-Sonnet",
    "google_gemini-3.1-pro-preview-20260219": "Gemini-3.1-Pro",
    "google_gemini-3.1-flash-lite-preview-20260303": "Gemini-3.1-Flash-Lite",
    "google_gemma-3-27b-it": "Gemma-3-27B",
    "deepseek-chat": "DeepSeek-Chat",
    "deepseek-reasoner": "DeepSeek-Reasoner",
    "x-ai_grok-4.20-20260309": "Grok-4",
    "Qwen_Qwen3.5-27B": "Qwen3.5-27B",
    "Qwen_Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen_Qwen3.5-4B": "Qwen3.5-4B",
    "Qwen_Qwen3-32B": "Qwen3-32B",
    "Qwen_Qwen3-14B": "Qwen3-14B",
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "Qwen_Qwen3-4B": "Qwen3-4B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "mistralai_Mistral-7B-Instruct-v0.3": "Mistral-7B",
}

COLOR_MAR     = "#EECDCD"
COLOR_REVERSE = "#F8E6D0"
plt.rcParams['font.family'] = 'serif'


def model_key(path):
    return re.sub(r"_bbq_results$", "", path.stem)


def load_entries(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def asymmetry_counts(entries):
    """Return (mar_count, reverse_count, total)."""
    mar = reverse = 0
    for e in entries:
        sc = e["responses"]["stereotyped"]["correct"]
        cc = e["responses"]["contrast"]["correct"]
        if not sc and cc:
            mar += 1
        elif sc and not cc:
            reverse += 1
    return mar, reverse, len(entries)


def draw_pair(ax, mar, reverse, total, title):
    """
    Full 2×2 confusion matrix layout (rows=stereo, cols=contrast).
    Diagonal cells (both-correct, both-wrong) are whited out.
    Only off-diagonal cells are coloured.

    Layout:
                  Contrast ✓       Contrast ✗
    Stereo ✓   [  white/blank  ]  [  Reverse  ]
    Stereo ✗   [     MAR       ]  [  white/blank  ]
    """
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")

    # (row, col) -> (count, color, top_label)  for off-diagonal cells
    # row 0 = Stereo Correct (top), row 1 = Stereo Wrong (bottom)
    # col 0 = Contrast Correct (left), col 1 = Contrast Wrong (right)
    cells = {
        (0, 0): None,                                        # both correct — white
        (0, 1): (reverse, COLOR_REVERSE, "Stereo$\\checkmark$ Contrast$\\times$\n(Bias)"),
        (1, 0): (mar,     COLOR_MAR,     "Stereo$\\times$ Contrast$\\checkmark$\n(MAR)"),
        (1, 1): None,                                        # both wrong   — white
    }

    for (row, col), cell in cells.items():
        x, y = col, 1 - row   # bottom-left corner of cell
        if cell is None:
            ax.add_patch(plt.Rectangle(
                (x, y), 1, 1,
                facecolor="white", edgecolor="#cccccc", linewidth=0,
            ))
        else:
            count, color, label = cell
            pct = count / total * 100 if total > 0 else 0.0
            ax.add_patch(plt.Rectangle(
                (x, y), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0,
            ))
            
            if "MAR" in label:
                ax.text(x + 0.5, y + 0.72, label,
                        ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")
                ax.text(x + 0.5, y + 0.30, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=13, color="black", fontweight="bold")
            else:
                ax.text(x + 0.5, y + 0.72, label,
                        ha="center", va="center",
                        fontsize=9, color="black")
                ax.text(x + 0.5, y + 0.30, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=13, color="black")

    ax.set_xticks([])
    # ax.set_xticklabels(["Contrast\nCorrect", "Contrast\nWrong"], fontsize=8)
    ax.set_yticks([])
    # ax.set_yticklabels(["Stereo\nWrong", "Stereo\nCorrect"], fontsize=8)
    ax.tick_params(length=0)
    # ax.set_title(title, fontsize=9, pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_model(entries, display_name, save_path):
    mar, reverse, total = asymmetry_counts(entries)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_pair(ax, mar, reverse, total, display_name)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_grid(model_data, save_path, ncols=4):
    n = len(model_data)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.7 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (display_name, entries) in enumerate(model_data):
        mar, reverse, total = asymmetry_counts(entries)
        draw_pair(axes_flat[idx], mar, reverse, total,
                  f"{display_name}  (n={total})")

    for idx in range(len(model_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Asymmetric Prediction Errors per Model\n"
        "Red = MAR (stereo wrong, contrast correct) | Blue = Reverse (stereo correct, contrast wrong)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout(h_pad=1.5, w_pad=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def main():
    result_files = sorted(
        p for p in RESULTS_DIR.glob("*_bbq_results.jsonl")
        if "trigger" not in p.name
        and "cot" not in p.name
        and "mar_examples" not in p.name
    )

    all_data = {}
    for path in result_files:
        key = model_key(path)
        entries = load_entries(path)
        if len(entries) < 1000:
            print(f"  Skipping {key} ({len(entries)} entries — incomplete)")
            continue
        display = MODEL_DISPLAY.get(key, key)
        all_data[key] = (display, entries)
        print(f"Loaded {display}: {len(entries)} entries")

    print(f"\nGenerating per-model figures ({len(all_data)} models)...")
    for key, (display, entries) in sorted(all_data.items()):
        plot_model(entries, display, FIG_DIR / f"asym_{key}.pdf")

    print("\nGenerating combined grid figure...")
    grid_order = sorted(
        all_data.items(),
        key=lambda kv: -asymmetry_counts(kv[1][1])[0] / asymmetry_counts(kv[1][1])[2],
    )
    grid_data = [(disp, ent) for _, (disp, ent) in grid_order]
    plot_grid(grid_data, RESULTS_DIR / "figures" / "asymmetry_grid.png", ncols=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
