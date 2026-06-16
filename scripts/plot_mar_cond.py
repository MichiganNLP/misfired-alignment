"""
Conditional-MAR variants of the dumbbell figure and the overall results
table. Conditional MAR is defined as

    MAR_cond = #(stereo wrong AND contrast right) / #(contrast right)

i.e. the fraction of *answerable* prompts on which the model exhibits
the suppression behaviour. This denominator removes pairs where the
model could not answer the contrast prompt correctly --- so the metric
is unaffected by the model's general QA competence and isolates the
bias-induced asymmetry.

Outputs (all new files, no existing file is modified):
    paper/images/mar_dumbbell_mar_cond.pdf
    paper/images/mar_dumbbell_mar_cond.png
    paper/overall_table_mar_cond.tex

Run from project root:
    python scripts/plot_mar_cond.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse canonical model display names + family colors from the main figures.
sys.path.insert(0, str(Path(__file__).parent))
import plot_paper_figures as pf  # noqa: E402

PROJ_DIR    = Path(__file__).parent.parent
RESULTS_DIR = PROJ_DIR / "results"
OUT_FIG_DIR = PROJ_DIR / "paper" / "images"
OUT_TEX     = PROJ_DIR / "paper" / "overall_table_mar_cond.tex"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> list[dict]:
    return json.load(open(path)).get("results", [])


def mar_pair(results: list[dict],
             contrast_fallback: dict[str, bool] | None = None
             ) -> tuple[float, float, int, int]:
    """Return (mar, mar_cond, n_total, n_contrast_right)."""
    n_total = n_mar = n_contrast_r = 0
    for r in results:
        if "stereotyped" not in r["responses"]:
            continue
        s_correct = r["responses"]["stereotyped"]["correct"]
        if "contrast" in r["responses"]:
            c_correct = r["responses"]["contrast"]["correct"]
        elif contrast_fallback is not None and r["id"] in contrast_fallback:
            c_correct = contrast_fallback[r["id"]]
        else:
            continue
        n_total += 1
        if c_correct:
            n_contrast_r += 1
        if (not s_correct) and c_correct:
            n_mar += 1
    if n_total == 0:
        return float("nan"), float("nan"), 0, 0
    mar = 100.0 * n_mar / n_total
    mar_cond = (100.0 * n_mar / n_contrast_r) if n_contrast_r else float("nan")
    return mar, mar_cond, n_total, n_contrast_r


def collect() -> dict[str, dict]:
    """Return {disp: {base_mar, base_mar_cond, trigger_mar, trigger_mar_cond,
                      n, n_contrast_r, contrast_acc}}."""
    # Pre-build per-model bbq contrast lookup as fallback for stereo-only triggers.
    bbq_contrast: dict[str, dict[str, bool]] = {}
    for path in sorted(RESULTS_DIR.glob("*_bbq_results.json")):
        m = re.match(r"^(.+?)_bbq_results\.json$", path.name)
        if not m:
            continue
        key = m.group(1)
        if key not in pf.MODEL_DISPLAY:
            continue
        try:
            rs = load_results(path)
        except Exception:
            continue
        bbq_contrast[key] = {
            r["id"]: r["responses"]["contrast"]["correct"]
            for r in rs if "contrast" in r["responses"]
        }

    stats: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        m = re.match(r"^(.+?)_(bbq|bbq_trigger)_results\.json$", path.name)
        if not m:
            continue
        key, tag = m.group(1), m.group(2)
        if key not in pf.MODEL_DISPLAY:
            continue
        disp = pf.MODEL_DISPLAY[key]
        results = load_results(path)
        if len(results) < 1500:
            continue
        fb = bbq_contrast.get(key) if tag == "bbq_trigger" else None
        mar, mar_cond, n_total, n_contrast_r = mar_pair(results, contrast_fallback=fb)
        bucket = stats.setdefault(disp, {})
        if tag == "bbq":
            bucket["base_mar"]      = mar
            bucket["base_mar_cond"] = mar_cond
            bucket["n"]             = n_total
            bucket["n_contrast_r"]  = n_contrast_r
            bucket["contrast_acc"]  = 100.0 * n_contrast_r / n_total if n_total else float("nan")
        elif tag == "bbq_trigger":
            bucket["trigger_mar"]      = mar
            bucket["trigger_mar_cond"] = mar_cond
    return stats


# ── Dumbbell on conditional MAR ──────────────────────────────────────────────
def fig_dumbbell_cond(stats: dict, out_path: Path):
    rows = [
        (disp, s["base_mar_cond"], s["trigger_mar_cond"])
        for disp, s in stats.items()
        if "base_mar_cond" in s and "trigger_mar_cond" in s
    ]
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.8, 0.3 * n + 0.6))

    y = np.arange(n)
    for i, (disp, base, trig) in enumerate(rows):
        col = pf.FAMILY_COLOR[pf.family_of(disp)]
        ax.plot([base, trig], [y[i], y[i]], color=col, alpha=0.45, lw=1.4,
                zorder=1, solid_capstyle="round")
        ax.scatter([base], [y[i]], s=30, facecolors="white",
                   edgecolors=col, linewidths=1.3, zorder=3)
        ax.scatter([trig], [y[i]], s=70, color=col, zorder=3,
                   edgecolors="white", linewidths=0.8)
        delta = trig - base
        sign = "+" if delta >= 0 else "−"
        ax.text(trig + 2.0, y[i], f"{sign}{abs(delta):.1f}",
                va="center", ha="left", fontsize=7, color=col, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels([d for d, _, _ in rows], fontsize=8.5, rotation=0)
    for tick, (disp, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(pf.FAMILY_COLOR[pf.family_of(disp)])

    ax.invert_yaxis()
    ax.set_xlabel("MAR (%)")
    # Trigger conditional MAR can exceed 75% for the most impacted open-weight
    # models; widen the x-grid range and pad the right edge so the +Δ
    # annotations don't clip.
    xmax_data = max(r[2] for r in rows)
    step = 5 if xmax_data < 45 else 10
    ax.set_xlim(left=0, right=xmax_data + step + 4)
    ax.set_xticks(np.arange(0, int(xmax_data) + step + 1, step))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    base_handle = plt.scatter([], [], s=30, facecolors="white",
                              edgecolors="#555", linewidths=1.3, label="Base")
    trig_handle = plt.scatter([], [], s=70, color="#555", label="Primed",
                              edgecolors="white", linewidths=0.8)
    ax.legend(handles=[base_handle, trig_handle],
              loc="lower right", frameon=False, fontsize=8, handletextpad=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} models)")


# ── Overall results table ────────────────────────────────────────────────────
def write_table(stats: dict, out_path: Path):
    rows = [
        (disp, s.get("base_mar"), s.get("base_mar_cond"),
         s.get("trigger_mar"), s.get("trigger_mar_cond"),
         s.get("contrast_acc"))
        for disp, s in stats.items()
        if "base_mar_cond" in s
    ]
    # Sort by base conditional MAR descending (so the table reads top-to-bottom
    # by bias-induced asymmetry, not by raw failure rate).
    rows.sort(key=lambda r: -(r[2] if r[2] is not None else -1))

    lines = [
        "% Auto-generated by scripts/plot_mar_cond.py — do not edit by hand.",
        "% Conditional MAR = #(stereo wrong AND contrast right) / #(contrast right).",
        "",
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Unconditional vs.\ conditional MAR per model. "
        r"Conditional MAR (denoted $\mathrm{MAR}_{\mathrm{cond}}$) divides by "
        r"the number of pairs on which the model answers the matched contrast "
        r"prompt correctly, isolating bias-induced asymmetry from general QA "
        r"competence. Models are sorted by base $\mathrm{MAR}_{\mathrm{cond}}$ "
        r"descending. The two metrics agree closely on frontier models with "
        r"high contrast accuracy and diverge on smaller open-weight models. "
        r"Trigger columns reflect the priming-prefix variant.}",
        r"\label{tab:overall_mar_cond}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{\textbf{Base}} & \multicolumn{2}{c}{\textbf{Trigger}}"
        r" & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"\textbf{Model} & "
        r"\textbf{MAR} & $\mathbf{MAR_{cond}}$ & "
        r"\textbf{MAR} & $\mathbf{MAR_{cond}}$ & "
        r"\textbf{Contr.\ Acc.} \\",
        r"\midrule",
    ]
    def fmt(v):
        return "---" if v is None else f"{v:.2f}"

    for disp, b, b_c, t, t_c, ca in rows:
        lines.append(
            f"{disp.replace('_', '-')} & "
            f"{fmt(b)} & \\textbf{{{fmt(b_c)}}} & "
            f"{fmt(t)} & {fmt(t_c)} & "
            f"{fmt(ca)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}  ({len(rows)} models)")


def main():
    stats = collect()
    fig_dumbbell_cond(stats, OUT_FIG_DIR / "mar_dumbbell_mar_cond.pdf")
    write_table(stats, OUT_TEX)


if __name__ == "__main__":
    main()
