"""
ICL ablation plot: MAR vs number of demonstrations for Claude-4.6 and
DeepSeek-V3, computed on a 2,022-pair held-out evaluation set (10 demo
pair_ids withheld).

Outputs:
    paper/images/icl_ablation.pdf
    paper/images/icl_ablation.png

Family colors match the dumbbell figure:
    Anthropic = #c8704a   (warm orange)
    DeepSeek  = #4d6bfe   (blue)

Run from project root:
    python scripts/plot_icl_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJ_DIR    = Path(__file__).parent.parent
RESULTS_DIR = PROJ_DIR / "results"
DEMO_FILE   = PROJ_DIR / "data" / "icl_demos.json"
OUT_DIR     = PROJ_DIR / "paper" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Match the dumbbell color scheme.
FAMILY_COLOR = {
    "Anthropic": "#c8704a",
    "DeepSeek":  "#4d6bfe",
    "OpenAI":    "#10a37f",
}

MODELS = [
    ("Claude-4.6-Sonnet", "Anthropic", "anthropic_claude-4.6-sonnet-20260217"),
    ("DeepSeek-V3-chat",  "DeepSeek",  "deepseek-chat"),
    ("GPT-5.4",           "OpenAI",    "openai_gpt-5.4-20260305"),
]
SHOTS = [0, 1, 3, 5]

plt.rcParams["font.family"]       = "serif"
plt.rcParams["font.size"]         = 9
plt.rcParams["axes.linewidth"]    = 0.6
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


def load_records(stem: str) -> list[dict]:
    final = RESULTS_DIR / f"{stem}_results.json"
    if final.exists():
        return json.load(open(final))["results"]
    with open(RESULTS_DIR / f"{stem}_results.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def mar_cond(records: list[dict]) -> float:
    """Conditional MAR: #(stereo wrong AND contrast right) / #(contrast right).

    Isolates suppression behaviour from general task competence."""
    n_mar = n_contrast_right = 0
    for r in records:
        if "stereotyped" not in r["responses"] or "contrast" not in r["responses"]:
            continue
        s = r["responses"]["stereotyped"]["correct"]
        c = r["responses"]["contrast"]["correct"]
        if c:
            n_contrast_right += 1
            if not s:
                n_mar += 1
    return 100.0 * n_mar / n_contrast_right if n_contrast_right else float("nan")


def collect() -> dict[str, list[float]]:
    held = {d["pair_id"] for d in json.load(open(DEMO_FILE))["demos"]}
    out: dict[str, list[float]] = {}
    for disp, _, stem in MODELS:
        ys = []
        for n in SHOTS:
            if n == 0:
                rs = load_records(f"{stem}_bbq")
                rs = [r for r in rs if r["id"] not in held]
            else:
                rs = load_records(f"{stem}_bbq_icl{n}")
            ys.append(mar_cond(rs))
        out[disp] = ys
    return out


def main():
    data = collect()

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for disp, fam, _stem in MODELS:
        col = FAMILY_COLOR[fam]
        ys  = data[disp]
        ax.plot(SHOTS, ys,
                color=col, lw=1.6, marker="o", markersize=5,
                markerfacecolor=col, markeredgecolor="white",
                markeredgewidth=0.8, label=disp, zorder=3)
        # Baseline reference line per model
        ax.axhline(ys[0], color=col, ls=(0, (1.5, 2)), lw=0.7,
                   alpha=0.45, zorder=1)
        # Annotate every point with its MAR value above the marker
        for x, y in zip(SHOTS, ys):
            ax.annotate(f"{y:.1f}", xy=(x, y),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5, color=col, zorder=4)

    ax.set_xticks(SHOTS)
    ax.set_xticklabels([str(s) for s in SHOTS])
    ax.set_xlabel("# Demonstrations")
    ax.set_ylabel(r"MAR$_{\mathrm{cond}}$ (%)")
    ax.set_xlim(-1.2, max(SHOTS) + 1.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    # Single-row legend above the plot, so it doesn't overlap the data lines.
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08),
                    ncol=len(MODELS), frameon=False, fontsize=8,
                    handlelength=1.4, handletextpad=0.5,
                    columnspacing=1.4, borderaxespad=0.0)
    for txt, (_, fam, _stem) in zip(leg.get_texts(), MODELS):
        txt.set_color(FAMILY_COLOR[fam])

    fig.tight_layout()
    fig.savefig(OUT_DIR / "icl_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "icl_ablation.png", bbox_inches="tight", dpi=220)
    print(f"  → {OUT_DIR / 'icl_ablation.pdf'}")
    print(f"  → {OUT_DIR / 'icl_ablation.png'}")

    # Also print the table to stdout for sanity
    print("\nMAR_cond (%) by shot count, on the held-out 2,022-pair set:")
    print(f"  {'shots':>5}  " + "  ".join(f"{disp:>18}" for disp, _, _ in MODELS))
    for i, n in enumerate(SHOTS):
        row = f"  {n:>5}  "
        for disp, _, _ in MODELS:
            row += f"  {data[disp][i]:>18.2f}"
        print(row)


if __name__ == "__main__":
    main()
