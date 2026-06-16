"""
Cross-family layer-profile figure for §C of the paper.

3 rows × 2 cols:
  rows = Llama-3.1-8B, Mistral-7B-v0.3, Qwen3-8B
  cols = Instruct, Base

Each panel: per-layer mean of (contrast logit-diff − stereotyped logit-diff)
on failure pairs (solid purple) and control pairs (dashed gray), with SEM
shading. Y-axis is shared per row (within family) for a fair Instruct-vs-base
comparison; the peak F/C layer is annotated.

Output: results/mechinterp/aggregate/cross_family_layer_profile.png
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).parent.parent.parent
MECH = PROJ / "results" / "mechinterp"
DATA = PROJ / "data"
OUT  = MECH / "aggregate" / "cross_family_layer_profile.png"

ROWS = [
    ("Llama-3.1-8B",
     "meta-llama/Llama-3.1-8B-Instruct",
     "meta-llama/Llama-3.1-8B",
     "mechinterp_pairs.json"),
    ("Mistral-7B-v0.3",
     "mistralai/Mistral-7B-Instruct-v0.3",
     "mistralai/Mistral-7B-v0.3",
     "mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json"),
    ("Qwen3-8B",
     "Qwen/Qwen3-8B",
     "Qwen/Qwen3-8B-Base",
     "mechinterp_pairs_Qwen_Qwen3-8B.json"),
    ("Qwen3.5-9B",
     "Qwen/Qwen3.5-9B",
     "Qwen/Qwen3.5-9B-Base",
     "mechinterp_pairs_Qwen_Qwen3.5-9B.json"),
]


def safe(name): return name.replace("/", "_").replace(":", "_")


def load_role_map(pairs_file: Path) -> dict[str, str]:
    d = json.load(open(pairs_file))
    return {p["id"]: p["role"] for p in d["pairs"]}


def aggregate_one(model: str, role_map: dict[str, str]) -> dict | None:
    d = MECH / safe(model) / "logit_lens"
    if not d.exists():
        return None
    by_role = defaultdict(lambda: {"stereo": [], "contrast": []})
    for jf in sorted(d.glob("*_logit_lens.json")):
        rec = json.load(open(jf))
        role = role_map.get(rec["pair_id"], "unknown")
        if role in ("failure", "control"):
            by_role[role]["stereo"].append(rec["per_layer_logit_diff_stereo"])
            by_role[role]["contrast"].append(rec["per_layer_logit_diff_contrast"])

    if not by_role:
        return None

    out = {}
    for role, d2 in by_role.items():
        if not d2["stereo"]:
            continue
        nL = max(len(x) for x in d2["stereo"])
        pad = lambda lst: np.array(
            [x + [np.nan] * (nL - len(x)) for x in lst], dtype=float)
        s = pad(d2["stereo"]); c = pad(d2["contrast"])
        gap = np.nanmean(c, axis=0) - np.nanmean(s, axis=0)
        n_per_layer = np.maximum(np.sum(~np.isnan(s), axis=0), 1)
        # SEM of contrast-stereo gap (rough): combine SEMs additively
        sem_s = np.nanstd(s, axis=0, ddof=1) / np.sqrt(n_per_layer)
        sem_c = np.nanstd(c, axis=0, ddof=1) / np.sqrt(n_per_layer)
        sem_gap = np.sqrt(sem_s ** 2 + sem_c ** 2)
        out[role] = {
            "n":   s.shape[0],
            "gap": gap,
            "sem": sem_gap,
        }
    return out


def plot_one_panel(ax, agg, model: str, label: str,
                   ymin: float, ymax: float, show_legend: bool = True):
    """Render the per-layer profile for one model on a given axes."""
    if agg is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([])
        return

    peak_layer = None
    fc_ratio = None
    if "failure" in agg:
        f_gap = agg["failure"]["gap"]
        peak_layer = int(np.nanargmax(f_gap))
        f_at_peak = float(f_gap[peak_layer])
        c_at_peak = float(agg["control"]["gap"][peak_layer]) if "control" in agg else 0
        # fc_ratio = f_at_peak / max(abs(c_at_peak), 1e-9)
        fc_ratio = f_at_peak - max(abs(c_at_peak), 1e-9)

    x = np.arange(len(next(iter(agg.values()))["gap"]))
    for role, color, ls, lw in [
        ("failure", "purple",  "-",  2.2),
        ("control", "dimgray", "--", 1.6),
    ]:
        if role not in agg: continue
        g, s = agg[role]["gap"], agg[role]["sem"]
        ax.plot(x, g, color=color, ls=ls, lw=lw,
                label=f"{role.capitalize()} (n={agg[role]['n']})")
        ax.fill_between(x, g - s, g + s, alpha=0.15, color=color)

    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    if peak_layer is not None:
        ax.axvline(peak_layer, color="purple", lw=1.0, ls=":", alpha=0.6)
        ax.annotate(f"L{peak_layer}, F-C={fc_ratio:.1f}",
                    xy=(peak_layer, agg["failure"]["gap"][peak_layer]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=10, color="purple",
                    bbox=dict(facecolor="white", edgecolor="purple",
                               boxstyle="round,pad=0.3", alpha=0.85))
    ax.set_ylim(ymin * 1.05 if ymin < 0 else -0.1 * ymax, ymax * 1.1)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Δ logit-diff (contrast − stereo)", fontsize=11)
    short_model = model.split("/")[-1]
    # ax.set_title(f"{label}\n({short_model})", fontsize=11)
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=9)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: per-row y-limits (Instruct + Base of each family share scale) ──
    row_ylims = []
    aggs = []  # [(family, label, model, agg, pairs_file)]
    for family, instruct, base, pairs_file in ROWS:
        rm = load_role_map(DATA / pairs_file)
        ymin, ymax = 0.0, 0.0
        family_aggs = []
        for label, model in (("Instruct", instruct), ("Base", base)):
            agg = aggregate_one(model, rm)
            family_aggs.append((label, model, agg))
            if agg:
                for role in ("failure", "control"):
                    if role in agg:
                        ymax = max(ymax, float(np.nanmax(agg[role]["gap"] + agg[role]["sem"])))
                        ymin = min(ymin, float(np.nanmin(agg[role]["gap"] - agg[role]["sem"])))
        row_ylims.append((ymin, ymax))
        aggs.append((family, family_aggs, pairs_file))

    # ── Combined N×2 figure (paper headline) ────────────────────────────────
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.4 * n_rows), sharex=False)
    if n_rows == 1:
        axes = np.array([axes])
    for row, (family, family_aggs, _) in enumerate(aggs):
        ymin, ymax = row_ylims[row]
        for col, (label, model, agg) in enumerate(family_aggs):
            plot_one_panel(axes[row, col], agg, model, f"{family} — {label}",
                           ymin, ymax, show_legend=(row == 0 and col == 0))
    fig.suptitle("Cross-family layer profile: where does the model split groups?",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Wrote combined figure: {OUT}")

    # ── Per-model standalone figures (for individual adjustment) ──────────────
    per_model_dir = OUT.parent / "per_model_layer_profile"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    for row, (family, family_aggs, _) in enumerate(aggs):
        ymin, ymax = row_ylims[row]
        for label, model, agg in family_aggs:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            plot_one_panel(ax, agg, model, f"{family} — {label}",
                           ymin, ymax, show_legend=True)
            plt.tight_layout()
            short = model.replace("/", "_")
            out = per_model_dir / f"{short}_layer_profile.pdf"
            plt.savefig(out, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"  per-model: {out}")

    print(f"\nPer-model figures: {per_model_dir}")


if __name__ == "__main__":
    main()
