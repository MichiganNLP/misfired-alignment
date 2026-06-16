"""
MAR_cond: base (PT) vs instruct (IT), 3 model families × 2 prompt conditions.

Two-panel grouped bar chart:
  - Left:  direct condition  (no priming prefix)
  - Right: trigger condition ("It is not okay to assume...")

Each panel has 3 model families on the x-axis, 2 bars per family (base, IT).
Family colors match the dumbbell figure; hatching distinguishes base vs IT.

Source data: results/yiming_results/y2_base_model/*.json (base) and
             results/<model>_<tag>_results.json                (IT).
Output:      paper/images/base_vs_it_mar_cond.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_paper_figures as pf  # FAMILY_COLOR

PROJ_DIR  = Path(__file__).parent.parent
RES       = PROJ_DIR / "results"
RES_BASE  = RES / "yiming_results" / "y2_base_model"
OUT_PATH  = PROJ_DIR / "paper" / "images" / "base_vs_it_mar_cond.pdf"

PAIRS = [
    ("Llama-3.1-8B",  "Meta",  "meta-llama_Llama-3.1-8B",  "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen3-8B",      "Qwen",  "Qwen_Qwen3-8B-Base",        "Qwen_Qwen3-8B"),
    ("Qwen3.5-9B",    "Qwen",  "Qwen_Qwen3.5-9B-Base",      "Qwen_Qwen3.5-9B"),
]
TAGS = [("Direct", "bbq"), ("Trigger", "bbq_trigger")]

BASE_HATCH = "///"
IT_HATCH   = ""    # solid (no hatch) for IT, slashes for base

plt.rcParams["font.family"]       = "serif"
plt.rcParams["font.size"]         = 9
plt.rcParams["axes.linewidth"]    = 0.6
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


def mar_cond(records):
    nk = nc = 0
    for r in records:
        if "stereotyped" not in r["responses"] or "contrast" not in r["responses"]:
            continue
        s = r["responses"]["stereotyped"]["correct"]
        c = r["responses"]["contrast"]["correct"]
        if c:
            nc += 1
            if not s: nk += 1
    return (100.0 * nk / nc) if nc else float("nan")


def collect():
    """Return {(family, cond): {variant: mar_cond}}."""
    data = {}
    for disp, fam, base_key, it_key in PAIRS:
        for cond_name, tag in TAGS:
            bp = RES_BASE / f"{base_key}_{tag}_results.json"
            ip = RES      / f"{it_key}_{tag}_results.json"
            if not bp.exists() or not ip.exists(): continue
            b = mar_cond(json.load(open(bp))["results"])
            i = mar_cond(json.load(open(ip))["results"])
            data[(disp, cond_name)] = {"base": b, "instruct": i}
    return data


def main():
    data = collect()
    fams = [(d, f) for d, f, _, _ in PAIRS]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

    for ax, (cond_name, _) in zip(axes, TAGS):
        x = np.arange(len(fams))
        width = 0.36
        for j, (disp, fam) in enumerate(fams):
            d = data.get((disp, cond_name))
            if d is None: continue
            col = pf.FAMILY_COLOR[fam]
            # Base bar (left, hatched)
            ax.bar(x[j] - width / 2, d["base"], width=width,
                   facecolor=col, edgecolor="white", hatch=BASE_HATCH,
                   linewidth=0.6, zorder=2)
            # IT bar (right, solid)
            ax.bar(x[j] + width / 2, d["instruct"], width=width,
                   facecolor=col, edgecolor="white", hatch=IT_HATCH,
                   linewidth=0.6, zorder=2)
            # Numeric labels
            ax.text(x[j] - width / 2, d["base"] + 0.6,
                    f"{d['base']:.1f}", ha="center", va="bottom",
                    fontsize=7, color=col)
            ax.text(x[j] + width / 2, d["instruct"] + 0.6,
                    f"{d['instruct']:.1f}", ha="center", va="bottom",
                    fontsize=7, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([d for d, _ in fams], fontsize=8.5)
        for tick, (_, fam) in zip(ax.get_xticklabels(), fams):
            tick.set_color(pf.FAMILY_COLOR[fam])
        ax.set_ylabel(r"MAR$_{\mathrm{cond}}$ (%)" if ax is axes[0] else "")
        ax.set_title(cond_name, fontsize=10)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4, zorder=0)
        # Same y-limits per panel based on its data
        max_in_panel = max(
            max(d["base"], d["instruct"])
            for k, d in data.items() if k[1] == cond_name
        )
        ax.set_ylim(0, max_in_panel * 1.18)

    # Single shared legend at top
    legend_face = "#aaaaaa"
    base_h = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                            edgecolor="white", hatch=BASE_HATCH, label="Base (PT)")
    it_h   = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                            edgecolor="white", hatch=IT_HATCH, label="Instruct (IT)")
    fig.legend(handles=[base_h, it_h],
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               frameon=False, fontsize=9, ncol=2,
               handlelength=1.6, handletextpad=0.5, columnspacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {OUT_PATH}")

    # Print compact summary
    print(f"\n{'Family':<14}  {'Condition':<8}  {'base':>9}  {'IT':>9}  {'IT-base':>9}")
    print("-" * 56)
    for disp, fam, _, _ in PAIRS:
        for cond_name, _ in TAGS:
            d = data.get((disp, cond_name))
            if d is None: continue
            print(f"{disp:<14}  {cond_name:<8}  "
                  f"{d['base']:>8.2f}%  {d['instruct']:>8.2f}%  "
                  f"{d['instruct']-d['base']:>+8.2f}")


if __name__ == "__main__":
    main()
