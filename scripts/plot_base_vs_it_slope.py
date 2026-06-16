"""
Base (PT) vs Instruct (IT) MAR_cond on the direct (no-trigger) condition.

Mirrors the CoT slope plot style: horizontal grouped bars, hatch patterns
to distinguish base vs IT, family-coded bar colours, numeric annotations
to the right of each bar, sorted by Δ (IT − base) descending so the
alignment-installs-the-failure effect (largest positive) appears on top.

Output: paper/images/base_vs_it_slope_mar_cond.{pdf,png}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_paper_figures as pf  # FAMILY_COLOR

PROJ_DIR = Path(__file__).parent.parent
RES      = PROJ_DIR / "results"
RES_BASE = RES / "yiming_results" / "y2_base_model"
OUT_PATH = PROJ_DIR / "paper" / "images" / "base_vs_it_slope_mar_cond.pdf"

# (display, family, base_key, it_key)
PAIRS = [
    ("Llama-3.1-8B", "Meta", "meta-llama_Llama-3.1-8B",  "meta-llama_Llama-3.1-8B-Instruct"),
    ("Qwen3-8B",     "Qwen", "Qwen_Qwen3-8B-Base",        "Qwen_Qwen3-8B"),
    ("Qwen3.5-9B",   "Qwen", "Qwen_Qwen3.5-9B-Base",      "Qwen_Qwen3.5-9B"),
]
TAG = "bbq"   # direct condition only

BASE_HATCH = "///"
IT_HATCH   = "..."

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


def main():
    rows = []
    for disp, fam, bk, ik in PAIRS:
        bp = RES_BASE / f"{bk}_{TAG}_results.json"
        ip = RES      / f"{ik}_{TAG}_results.json"
        if not bp.exists() or not ip.exists():
            continue
        b = mar_cond(json.load(open(bp))["results"])
        i = mar_cond(json.load(open(ip))["results"])
        rows.append((disp, fam, b, i))

    # Largest IT-base Δ on top (alignment introduces the most)
    rows.sort(key=lambda r: r[3] - r[2], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(4.8, 0.55 * n + 1.0))

    y = np.arange(n)
    bar_h = 0.38
    for i, (disp, fam, base, it) in enumerate(rows):
        col = pf.FAMILY_COLOR[fam]
        ax.barh(y[i] - bar_h / 2, base, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=BASE_HATCH, zorder=2)
        ax.barh(y[i] + bar_h / 2, it, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=IT_HATCH, zorder=2)
        ax.text(base + 0.15, y[i] - bar_h / 2, f"{base:.1f}",
                va="center", ha="left", fontsize=7, color=col)
        ax.text(it + 0.15, y[i] + bar_h / 2, f"{it:.1f}",
                va="center", ha="left", fontsize=7, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels([d for d, _, _, _ in rows], fontsize=8.5)
    for tick, (_, fam, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(pf.FAMILY_COLOR[fam])
    ax.invert_yaxis()
    ax.set_xlabel(r"MAR$_{\mathrm{cond}}$ (%)")
    ax.set_xlim(0, max(max(b, i) for _, _, b, i in rows) * 1.45)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    legend_face = "#bbbbbb"
    base_h = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                            edgecolor="white", hatch=BASE_HATCH, label="Base (PT)")
    it_h   = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                            edgecolor="white", hatch=IT_HATCH,   label="Instruct (IT)")
    ax.legend(handles=[base_h, it_h],
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, fontsize=7.5, ncol=2,
              handlelength=1.6, handletextpad=0.4, columnspacing=1.4)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {OUT_PATH}")
    print()
    print(f"{'Model':<14}  {'base':>8}  {'IT':>8}  {'IT-base':>9}")
    for d, _, b, i in rows:
        print(f"{d:<14}  {b:>7.2f}%  {i:>7.2f}%  {i-b:>+8.2f}")


if __name__ == "__main__":
    main()
