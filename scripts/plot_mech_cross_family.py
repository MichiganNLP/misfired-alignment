"""
Cross-family mech-interp layer profile figure.

Three side-by-side panels — one per model family (Llama-3.1-8B, Mistral-7B,
Gemma-3-27B) — each showing the mean per-layer (contrast logit-diff −
stereo logit-diff) gap, separately for failure pairs (red) and control
pairs (blue), on the IT variant. Vertical line marks the peak-gap layer.
The base (PT) curve is shown as a faint dotted overlay to make the
alignment-causation contrast visible at a glance.

Outputs:
    paper/images/mechanistic/cross_family_layer_profile.pdf
    paper/images/mechanistic/cross_family_layer_profile.png
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).parent.parent
RES  = PROJ / "results" / "mechinterp"
OUT  = PROJ / "paper" / "images" / "mechanistic" / "cross_family_layer_profile.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FAMILIES = [
    {"display": "Llama-3.1-8B",   "n_layers": 32,
     "it_dir":  "meta-llama_Llama-3.1-8B-Instruct",
     "pt_dir":  "meta-llama_Llama-3.1-8B",
     "role_file": "data/mechinterp_pairs.json"},
    {"display": "Mistral-7B",     "n_layers": 32,
     "it_dir":  "mistralai_Mistral-7B-Instruct-v0.3",
     "pt_dir":  "mistralai_Mistral-7B-v0.3",
     "role_file": "data/mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json"},
    {"display": "Gemma-3-27B",    "n_layers": 62,
     "it_dir":  "google_gemma-3-27b-it",
     "pt_dir":  "google_gemma-3-27b-pt",
     "role_file": "data/mechinterp_pairs_gemma3.json"},
]

C_FAIL = "#c0392b"   # red — failure pairs
C_CTRL = "#2980b9"   # blue — control pairs

plt.rcParams["font.family"]       = "serif"
plt.rcParams["font.size"]         = 9
plt.rcParams["axes.linewidth"]    = 0.6
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


def load_role_map(p):
    d = json.load(open(p))
    return {x["id"]: x.get("role", "unknown") for x in d.get("pairs", [])}


def per_layer_gap_curves(model_dir: str, role_map: dict[str, str]) -> dict[str, list[float]]:
    """Mean per-layer (contrast - stereo) curves keyed by role."""
    by_role = defaultdict(list)
    for jf in sorted((RES / model_dir / "logit_lens").glob("*_logit_lens.json")):
        try: d = json.load(open(jf))
        except Exception: continue
        pid  = jf.stem.replace("_logit_lens", "")
        role = role_map.get(pid, "unknown")
        if role not in ("failure", "control"): continue
        s = d.get("per_layer_logit_diff_stereo")
        c = d.get("per_layer_logit_diff_contrast")
        if not s or not c: continue
        by_role[role].append([ci - si for si, ci in zip(s, c)])
    means = {}
    for role, curves in by_role.items():
        n_layers = max(len(g) for g in curves)
        means[role] = [
            sum(g[L] for g in curves if len(g) > L) / sum(1 for g in curves if len(g) > L)
            for L in range(n_layers)
        ]
    return means


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0), sharey=False)

    for ax, fam in zip(axes, FAMILIES):
        rmap = load_role_map(fam["role_file"])
        it_curves = per_layer_gap_curves(fam["it_dir"], rmap)
        pt_curves = per_layer_gap_curves(fam["pt_dir"], rmap)
        n_layers = fam["n_layers"]
        x = np.arange(n_layers)

        # IT curves (solid, prominent)
        if "failure" in it_curves:
            y = it_curves["failure"]
            ax.plot(x[:len(y)], y, color=C_FAIL, lw=2.0, label="Failure (IT)", zorder=4)
            peak_l = int(np.argmax(y)); peak_v = y[peak_l]
            ax.axvline(peak_l, color=C_FAIL, ls=":", lw=0.8, alpha=0.8, zorder=2)
            ax.annotate(f"L{peak_l}\n{peak_v:+.1f}", xy=(peak_l, peak_v),
                        xytext=(6, -2), textcoords="offset points",
                        fontsize=11.5, color=C_FAIL, va="top",
                        path_effects=None)
        if "control" in it_curves:
            y = it_curves["control"]
            ax.plot(x[:len(y)], y, color=C_CTRL, lw=2.0, label="Control (IT)", zorder=3)

        # PT curves (faint dotted overlay)
        if "failure" in pt_curves:
            y = pt_curves["failure"]
            ax.plot(x[:len(y)], y, color=C_FAIL, lw=1.0, ls=(0, (1.5, 1.5)),
                    alpha=0.45, label="Failure (PT)", zorder=1)
        if "control" in pt_curves:
            y = pt_curves["control"]
            ax.plot(x[:len(y)], y, color=C_CTRL, lw=1.0, ls=(0, (1.5, 1.5)),
                    alpha=0.45, label="Control (PT)", zorder=1)

        ax.axhline(0, color="#888", lw=0.5, zorder=0)
        ax.set_title(fam["display"], fontsize=12)
        ax.set_xlabel(f"Layer (out of {n_layers})", fontsize=12.5)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.grid(axis="y", lw=0.4, alpha=0.4, zorder=0)

    axes[0].set_ylabel(r"Contrast $-$ Stereo logit-diff")

    # One shared legend in the upper-right of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", bbox_to_anchor=(0.5, 1.04),
               frameon=False, ncol=4, fontsize=10.5,
               handlelength=2.2, handletextpad=0.5, columnspacing=1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT, bbox_inches="tight")
    plt.savefig(OUT.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
