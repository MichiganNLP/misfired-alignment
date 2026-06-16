"""
Plot pair-level confusion matrices for each model.

For each pair, the model's stereotyped answer and contrast answer are each
either correct or incorrect.  This gives four outcome cells:

                    Contrast Correct  |  Contrast Wrong
  Stereo Correct |   Both Correct    |  Stereo-only OK
  Stereo Wrong   |  *** MAR ***      |   Both Wrong

Outputs:
  results/figures/confusion_matrices/      <- per-model PNGs
  results/figures/confusion_matrix_grid.png  <- selected models in one figure
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures" / "confusion_matrices"
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

# Representative subset for the combined grid
GRID_MODELS = [
    "openai_gpt-5.4-20260305",
    "anthropic_claude-4.7-opus-20260416",
    "anthropic_claude-4.6-sonnet-20260217",
    "google_gemma-3-27b-it",
    "deepseek-reasoner",
    "x-ai_grok-4.20-20260309",
    "Qwen_Qwen3-14B",
    "Qwen_Qwen2.5-7B-Instruct",
    "meta-llama_Llama-3.1-8B-Instruct",
    "meta-llama_Llama-3.2-3B-Instruct",
    "mistralai_Mistral-7B-Instruct-v0.3",
    "Qwen_Qwen3.5-27B",
]

# Cell colours
# (row=stereo, col=contrast): (facecolor, is_mar)
CELL_COLORS = {
    (0, 0): "#4393c3",   # both correct  — blue
    (0, 1): "#92c5de",   # stereo-only   — light blue
    (1, 0): "#d6604d",   # MAR           — red
    (1, 1): "#f4a582",   # both wrong    — light red/salmon
}
CELL_LABELS = {
    (0, 0): "Both\nCorrect",
    (0, 1): "Stereo\nOnly",
    (1, 0): "MAR\n★",
    (1, 1): "Both\nWrong",
}


def model_key(path: Path) -> str:
    return re.sub(r"_bbq_results$", "", path.stem)


def load_entries(path: Path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def pair_counts(entries):
    """
    Return a 2×2 array:
        [stereo_correct][contrast_correct]
    where 0=correct, 1=wrong  (so MAR is at [1][0]).
    """
    counts = np.zeros((2, 2), dtype=int)
    for e in entries:
        sc = int(not e["responses"]["stereotyped"]["correct"])  # 0 correct, 1 wrong
        cc = int(not e["responses"]["contrast"]["correct"])
        counts[sc, cc] += 1
    return counts


def draw_cm(ax, counts, title):
    total = counts.sum()
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")

    for sr in range(2):
        for cc in range(2):
            val = counts[sr, cc]
            pct = val / total * 100 if total > 0 else 0.0
            color = CELL_COLORS[(sr, cc)]
            is_mar = (sr == 1 and cc == 0)

            ax.add_patch(plt.Rectangle(
                (cc, 1 - sr), 1, 1,
                facecolor=color,
                edgecolor="white", linewidth=1.5,
            ))
            # cell type label (small, top)
            ax.text(cc + 0.5, 1.5 - sr + 0.28,
                    CELL_LABELS[(sr, cc)],
                    ha="center", va="center",
                    fontsize=7, color="white",
                    fontweight="bold" if is_mar else "normal",
                    alpha=0.9)
            # count + percentage (large, centre)
            ax.text(cc + 0.5, 1.5 - sr - 0.10,
                    f"{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    # axis ticks / labels
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Contrast\nCorrect", "Contrast\nWrong"], fontsize=8)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Stereo\nWrong", "Stereo\nCorrect"], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_model(entries, display_name: str, save_path: Path):
    counts = pair_counts(entries)
    mar_pct = counts[1, 0] / counts.sum() * 100 if counts.sum() > 0 else 0.0

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    draw_cm(ax, counts, f"{display_name}  (MAR = {mar_pct:.1f}%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path.name}")


def plot_grid(model_data: list[tuple[str, list]], save_path: Path, ncols=4):
    n = len(model_data)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.6 * nrows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (display_name, entries) in enumerate(model_data):
        counts = pair_counts(entries)
        mar_pct = counts[1, 0] / counts.sum() * 100 if counts.sum() > 0 else 0.0
        draw_cm(axes_flat[idx], counts,
                f"{display_name}\nMAR = {mar_pct:.1f}%")

    # hide unused axes
    for idx in range(len(model_data), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(h_pad=2.0, w_pad=1.0)
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
        save_path = FIG_DIR / f"cm_{key}.png"
        plot_model(entries, display, save_path)

    print("\nGenerating combined grid figure...")
    grid_data = [all_data[k] for k in GRID_MODELS if k in all_data]
    if grid_data:
        plot_grid(grid_data, RESULTS_DIR / "figures" / "confusion_matrix_grid.png", ncols=4)

    print("\nDone.")


if __name__ == "__main__":
    main()
