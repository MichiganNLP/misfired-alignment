"""
Render the CoT slope figure using conditional MAR as the y-axis metric.
Mirrors `fig_cot` from `plot_paper_figures.py` (hatch patterns for Direct vs
CoT, family-coded colours), but with MAR_cond instead of unconditional MAR.

Source: results/mar_cond_cot.csv (computed by an earlier analysis run).
Output: paper/images/cot_slope_mar_cond.{pdf,png}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import plot_paper_figures as pf  # FAMILY_COLOR, family_of

PROJ_DIR = Path(__file__).parent.parent
CSV_PATH = PROJ_DIR / "results" / "mar_cond_cot.csv"
OUT_PATH = PROJ_DIR / "paper" / "images" / "cot_slope_mar_cond.pdf"

DIRECT_HATCH = "///"
COT_HATCH    = "..."

plt.rcParams["font.family"]       = "serif"
plt.rcParams["font.size"]         = 9
plt.rcParams["axes.linewidth"]    = 0.6
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append((
                r["model"],
                float(r["direct_mar_cond"]),
                float(r["cot_mar_cond"]),
            ))
    # CoT-worsens (positive Δ) at top, CoT-improves at bottom
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(4.8, 0.45 * n + 0.9))

    y = np.arange(n)
    bar_h = 0.38
    for i, (disp, direct, cot) in enumerate(rows):
        col = pf.FAMILY_COLOR[pf.family_of(disp)]
        ax.barh(y[i] - bar_h / 2, direct, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=DIRECT_HATCH, zorder=2)
        ax.barh(y[i] + bar_h / 2, cot, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=COT_HATCH, zorder=2)
        ax.text(direct + 0.25, y[i] - bar_h / 2, f"{direct:.1f}",
                va="center", ha="left", fontsize=7, color=col)
        ax.text(cot + 0.25, y[i] + bar_h / 2, f"{cot:.1f}",
                va="center", ha="left", fontsize=7, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels([d for d, _, _ in rows], fontsize=8.5)
    for tick, (disp, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(pf.FAMILY_COLOR[pf.family_of(disp)])
    ax.invert_yaxis()
    ax.set_xlabel(r"MAR$_{\mathrm{cond}}$ (%)")
    ax.set_xlim(0, max(max(d, c) for _, d, c in rows) * 1.45)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    legend_face = "#bbbbbb"
    direct_h = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                              edgecolor="white", hatch=DIRECT_HATCH, label="Direct")
    cot_h    = plt.Rectangle((0, 0), 1, 1, facecolor=legend_face,
                              edgecolor="white", hatch=COT_HATCH, label="CoT")
    ax.legend(handles=[direct_h, cot_h],
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, fontsize=7.5, ncol=2,
              handlelength=1.6, handletextpad=0.4, columnspacing=1.4)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.savefig(OUT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {OUT_PATH}  ({n} models)")
    print(f"\n{'Model':<18}  {'direct MAR_cond':>15}  {'cot MAR_cond':>13}  {'Δ':>+6}")
    for d, dr, ct in rows:
        print(f"{d:<18}  {dr:>14.2f}%  {ct:>12.2f}%  {ct-dr:>+6.2f}")


if __name__ == "__main__":
    main()
