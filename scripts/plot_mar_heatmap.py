"""
Per-(model, BBQ category) MAR heatmap.

Reproduces (and visualizes) the per-category significance table in
paper/neurips_2026.tex (Table 4) as a heatmap, suitable for inclusion in
the main body of the paper.

  rows = models (sorted by overall MAR, descending)
  cols = 8 BBQ categories
  cell value = MAR % on the bbq direct condition for that (model, category)
  cell color = sequential heatmap (low = white, high = dark red)

Outputs:
  results/figures/mar_heatmap_bbq.png
  results/figures/mar_heatmap_bbq_trigger.png
  results/figures/mar_heatmap_combined.png   (bbq + bbq_trigger side by side)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).parent.parent
RESULTS = PROJ / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Display names — same as plot_confusion_matrices.py
MODEL_DISPLAY = {
    "openai_gpt-5.4-20260305":                       "GPT-5.4",
    "openai_gpt-5.4-mini-20260317":                  "GPT-5.4-mini",
    "openai_gpt-5.4-nano-20260317":                  "GPT-5.4-nano",
    "gpt-5.5":                                       "GPT-5.5",
    "anthropic_claude-4.7-opus-20260416":            "Claude-4.7-Opus",
    "anthropic_claude-4.6-sonnet-20260217":          "Claude-4.6-Sonnet",
    "google_gemini-3.1-pro-preview-20260219":        "Gemini-3.1-Pro",
    "google_gemini-3.1-flash-lite-preview-20260303": "Gemini-3.1-Flash-Lite",
    "google_gemma-3-27b-it":                         "Gemma-3-27B",
    "deepseek-chat":                                 "DeepSeek-V3-chat",
    "deepseek-reasoner":                             "DeepSeek-R1",
    "x-ai_grok-4.20-20260309":                       "Grok-4.20",
    "Qwen_Qwen3.5-27B":                              "Qwen3.5-27B",
    "Qwen_Qwen3.5-9B":                               "Qwen3.5-9B",
    "Qwen_Qwen3.5-4B":                               "Qwen3.5-4B",
    "Qwen_Qwen3-32B":                                "Qwen3-32B",
    "Qwen_Qwen3-14B":                                "Qwen3-14B",
    "Qwen_Qwen3-8B":                                 "Qwen3-8B",
    "Qwen_Qwen3-4B":                                 "Qwen3-4B",
    "Qwen_Qwen2.5-72B-Instruct":                     "Qwen2.5-72B",
    "Qwen_Qwen2.5-7B-Instruct":                      "Qwen2.5-7B",
    "meta-llama_Llama-3.1-70B-Instruct":             "Llama-3.1-70B",
    "meta-llama_Llama-3.1-8B-Instruct":              "Llama-3.1-8B",
    "meta-llama_Llama-3.2-3B-Instruct":              "Llama-3.2-3B",
    "mistralai_Mistral-7B-Instruct-v0.3":            "Mistral-7B",
}

# Display order for categories — short labels matching the paper table
CATEGORIES = [
    ("Disability_status",   "Disab."),
    ("Physical_appearance", "Phys."),
    ("Gender_identity",     "Gender"),
    ("SES",                 "SES"),
    ("Religion",            "Relig."),
    ("Race_ethnicity",      "Race"),
    ("Sexual_orientation",  "Sexual"),
    ("Age",                 "Age"),
]


def load_mar_per_category(
    json_path: Path,
    fallback_contrast: Path | None = None,
) -> tuple[dict[str, float], float]:
    """Return ({category: mar%}, overall_mar%) from one results json.

    If the file lacks the `contrast` half (a stereo-only trigger run), pull
    the contrast outcomes from `fallback_contrast` (typically the matching
    bbq direct results — same prompts apart from the stereo-prefix)."""
    with open(json_path) as f:
        d = json.load(f)
    rs = d["results"]

    # Map pair_id -> contrast_correct from a fallback file if needed
    fb = None
    if fallback_contrast is not None and fallback_contrast.exists():
        with open(fallback_contrast) as f:
            fb_data = json.load(f)
        fb = {r["id"]: r["responses"]["contrast"]["correct"]
              for r in fb_data["results"]
              if "contrast" in r["responses"]}

    by_cat = defaultdict(lambda: {"n": 0, "mar": 0})
    n_total = mar_total = 0
    for r in rs:
        if "stereotyped" not in r["responses"]:
            continue
        s_correct = r["responses"]["stereotyped"]["correct"]
        if "contrast" in r["responses"]:
            c_correct = r["responses"]["contrast"]["correct"]
        elif fb is not None and r["id"] in fb:
            c_correct = fb[r["id"]]
        else:
            continue
        cat = r["category"]
        by_cat[cat]["n"] += 1
        is_mar = (not s_correct) and c_correct
        by_cat[cat]["mar"] += int(is_mar)
        n_total += 1
        mar_total += int(is_mar)
    cat_mar = {c: 100 * st["mar"] / max(st["n"], 1) for c, st in by_cat.items()}
    overall_mar = 100 * mar_total / max(n_total, 1)
    return cat_mar, overall_mar


def collect_data(tag: str, results_dir: Path) -> tuple[list[str], np.ndarray, list[float]]:
    """Walk results_dir for *_<tag>_results.json. Return (display_names,
    matrix [n_models x n_cats], overall_mar_per_model). Sorted by overall MAR
    descending."""
    pattern = f"*_{tag}_results.json"
    rows = []
    for jf in sorted(results_dir.glob(pattern)):
        if "_cot" in jf.stem or "_trigger_cot" in jf.stem:
            continue
        if tag == "bbq" and "_trigger" in jf.stem:
            continue
        # Strip suffix to get model key
        key = jf.stem.replace(f"_{tag}_results", "")
        # Some files might be edge cases
        display = MODEL_DISPLAY.get(key)
        if display is None:
            continue
        # If this is a trigger run that's stereo-only, join with the matched
        # bbq direct file for contrast outcomes (prompts are identical).
        fallback = None
        if tag == "bbq_trigger":
            bbq_match = results_dir / f"{key}_bbq_results.json"
            if bbq_match.exists():
                fallback = bbq_match
        try:
            cat_mar, overall = load_mar_per_category(jf, fallback_contrast=fallback)
        except Exception as e:
            print(f"  skip {jf.name}: {e}"); continue
        rows.append((display, overall, cat_mar))
    rows.sort(key=lambda x: -x[1])

    n = len(rows); cats = [c for c, _ in CATEGORIES]
    M = np.full((n, len(cats)), np.nan)
    names, overalls = [], []
    for i, (display, overall, cat_mar) in enumerate(rows):
        names.append(display); overalls.append(overall)
        for j, c in enumerate(cats):
            if c in cat_mar:
                M[i, j] = cat_mar[c]
    return names, M, overalls


def plot_heatmap(names, M, overalls, title, outpath, vmax=None,
                  show_overall_col: bool = True):
    cats_short = [s for _, s in CATEGORIES]
    cols = (cats_short + ["Overall"]) if show_overall_col else cats_short
    if show_overall_col:
        Mfull = np.column_stack([M, np.array(overalls)])
    else:
        Mfull = M

    if vmax is None:
        vmax = float(np.nanmax(Mfull)) * 1.05
    n_rows = len(names); n_cols = len(cols)

    fig_h = max(5.5, 0.32 * n_rows + 1.5)
    fig_w = 1.4 * n_cols + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.YlOrRd
    im = ax.imshow(Mfull, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

    # Cell annotations — MAR percentage
    for i in range(n_rows):
        for j in range(n_cols):
            v = Mfull[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=9, color="gray")
                continue
            # Pick text color based on cell darkness for readability
            color = "white" if v / vmax > 0.55 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=9, color=color)

    # Visual separator before "Overall" column
    if show_overall_col:
        ax.axvline(n_cols - 1.5, color="black", lw=1.0)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("BBQ Category", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Faint grid
    ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", lw=0.5)
    ax.tick_params(which="minor", length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MAR (%)", fontsize=10)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {outpath}")


def plot_combined(bbq_data, trigger_data, outpath, vmax=None):
    """Side-by-side heatmaps: bbq direct (left), bbq_trigger (right).
    Models aligned by row across the two — sorted by bbq_trigger overall MAR
    so the worst-performing models appear at the top of both panels."""
    # Build a unified row order from intersection, sorted by trigger overall
    bbq_names, bbq_M, bbq_over = bbq_data
    tr_names, tr_M, tr_over = trigger_data
    bbq_idx = {n: i for i, n in enumerate(bbq_names)}
    tr_idx  = {n: i for i, n in enumerate(tr_names)}
    common = [n for n in tr_names if n in bbq_idx]  # use trigger order

    cats_short = [s for _, s in CATEGORIES]
    bbq_M_aligned     = np.array([bbq_M[bbq_idx[n]] for n in common])
    tr_M_aligned      = np.array([tr_M[tr_idx[n]] for n in common])
    bbq_overall       = [bbq_over[bbq_idx[n]] for n in common]
    tr_overall        = [tr_over[tr_idx[n]] for n in common]

    cols  = cats_short + ["Overall"]
    bbq_full = np.column_stack([bbq_M_aligned, np.array(bbq_overall)])
    tr_full  = np.column_stack([tr_M_aligned,  np.array(tr_overall)])
    if vmax is None:
        vmax = float(np.nanmax(np.concatenate([bbq_full, tr_full]))) * 1.02
    n_rows = len(common); n_cols = len(cols)

    fig_h = max(6.0, 0.34 * n_rows + 1.5)
    fig_w = 2 * (1.25 * n_cols) + 3.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                              gridspec_kw={"wspace": 0.28})

    cmap = plt.cm.YlOrRd
    for ax, M, label in zip(axes, [bbq_full, tr_full],
                              ["bbq (no trigger)", "bbq_trigger"]):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        for i in range(n_rows):
            for j in range(n_cols):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "–", ha="center", va="center",
                            fontsize=9, color="gray"); continue
                color = "white" if v / vmax > 0.55 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.axvline(n_cols - 1.5, color="black", lw=1.0)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(common, fontsize=10)
        ax.set_xlabel("BBQ Category", fontsize=11)
        ax.set_title(f"MAR (%) — {label}", fontsize=12)
        ax.set_xticks(np.arange(n_cols) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_rows) - 0.5, minor=True)
        ax.grid(which="minor", color="lightgray", lw=0.5)
        ax.tick_params(which="minor", length=0)

    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02, label="MAR (%)")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(RESULTS),
                    help="Directory containing *_bbq_results.json files")
    ap.add_argument("--vmax-bbq",     type=float, default=None,
                    help="Color scale max for bbq panel (auto if omitted)")
    ap.add_argument("--vmax-trigger", type=float, default=None,
                    help="Color scale max for bbq_trigger panel (auto if omitted)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    print("Collecting bbq direct …")
    bbq_data     = collect_data("bbq",         results_dir)
    print(f"  {len(bbq_data[0])} models")

    print("Collecting bbq_trigger …")
    trigger_data = collect_data("bbq_trigger", results_dir)
    print(f"  {len(trigger_data[0])} models")

    plot_heatmap(*bbq_data,     "MAR (%) by category — bbq direct",
                 FIG_DIR / "mar_heatmap_bbq.png",
                 vmax=args.vmax_bbq)
    plot_heatmap(*trigger_data, "MAR (%) by category — bbq_trigger",
                 FIG_DIR / "mar_heatmap_bbq_trigger.png",
                 vmax=args.vmax_trigger)
    plot_combined(bbq_data, trigger_data,
                  FIG_DIR / "mar_heatmap_combined.png")


if __name__ == "__main__":
    main()
