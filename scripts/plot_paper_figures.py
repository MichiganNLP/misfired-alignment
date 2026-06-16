"""
Generate paper figures for the Results section.

Outputs (paper/images/):
  - mar_dumbbell.pdf       Figure 1: base vs trigger MAR per model (replaces tab:overall + tab:trigger)
  - cot_slope.pdf          Figure 2: direct vs CoT MAR for 5 models (replaces tab:cot)
  - per_category_heatmap.pdf  Figure 3: 23 models x 8 categories MAR heatmap

Run from project root:
    python scripts/plot_paper_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

PROJ_DIR    = Path(__file__).parent.parent
RESULTS_DIR = PROJ_DIR / "results"
SIG_PATH    = RESULTS_DIR / "significance.json"
OUT_DIR     = PROJ_DIR / "paper" / "images"
ICON_DIR    = OUT_DIR / "icons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-family icon files. SVG is preferred (vector-perfect); PNG also supported.
# Optional — missing files are silently skipped (figure falls back to text only).
# If a name has no extension, .svg is tried first, then .png.
ICON_FILE = {
    "OpenAI":    "openai",
    "Anthropic": "anthropic",
    "Google":    "google",
    "DeepSeek":  "deepseek",
    "xAI":       "xai",
    "Qwen":      "qwen",
    "Meta":      "meta",
    "Mistral":   "mistral",
}
ICON_TARGET_PX  = 6     # rendered icon height, in pixels (regardless of
                          # source PNG resolution). Tweak smaller for tighter,
                          # larger for more prominent icons.
ICON_X_FRACTION = -0.04  # how far left of the y-axis to place the icon
                          # (axes-fraction units; more negative = further out)
ICON_OVERSAMPLE = 40      # icon is resized to ICON_TARGET_PX * ICON_OVERSAMPLE
                          # in pixels (then displayed at the target size); the
                          # extra resolution stays crisp when readers zoom into
                          # the PDF. Bump higher for sharper, lower for smaller files.
                          # 30–40 is the sweet spot for SVG icons.
ICON_INTERPOLATION = "hanning"  # downsampling filter for OffsetImage. "hanning"
                          # is sharper than "lanczos" without ringing; try
                          # "nearest" for pixel-perfect (no smoothing) or
                          # "antialiased" for matplotlib's auto-pick.

# Per-row x position of the icon (in axes-fraction units). Edit one entry
# per row to nudge icons left/right individually. Missing keys fall back to
# ICON_X_DEFAULT. More negative = further left of the y-axis. The icon's
# right edge is anchored at this x value.
ICON_X_DEFAULT = -0.06
ICON_X_PER_ROW: dict[str, float] = {
    "GPT-5.4":               -0.4,
    "GPT-5.4-mini":          -0.5,
    "GPT-5.4-nano":          -0.5,
    "GPT-5.5":               -0.4,
    "Claude-4.7-Opus":       -0.6,
    "Claude-4.6-Sonnet":     -0.6,
    "Gemini-3.1-Pro":        -0.6,
    "Gemini-3.1-Flash-Lite": -0.75,
    "Gemma-3-27B":           -0.5,
    "DeepSeek-V3-chat":      -0.7,
    "DeepSeek-R1":           -0.55,
    "Grok-4.20":             -0.45,
    "Qwen3.5-27B":           -0.65,
    "Qwen3.5-9B":            -0.65,
    "Qwen3.5-4B":            -0.65,
    "Qwen3-32B":             -0.55,
    "Qwen3-14B":             -0.55,
    "Qwen3-8B":              -0.55,
    "Qwen3-4B":              -0.45,
    "Qwen2.5-72B":           -0.65,
    "Qwen2.5-7B":            -0.45,
    "Llama-3.1-70B":         -0.55,
    "Llama-3.1-8B":          -0.55,
    "Llama-3.2-3B":          -0.55,
    "Mistral-7B":            -0.45,
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ── Display names ────────────────────────────────────────────────────────────
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

# Map display name -> family (for coloring)
def family_of(disp: str) -> str:
    if disp.startswith("GPT"):           return "OpenAI"
    if disp.startswith("Claude"):        return "Anthropic"
    if disp.startswith("Gemini") or disp.startswith("Gemma"): return "Google"
    if disp.startswith("DeepSeek"):      return "DeepSeek"
    if disp.startswith("Grok"):          return "xAI"
    if disp.startswith("Qwen"):          return "Qwen"
    if disp.startswith("Llama"):         return "Meta"
    if disp.startswith("Mistral"):       return "Mistral"
    return "Other"

FAMILY_COLOR = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#c8704a",
    "Google":    "#4285f4",
    "DeepSeek":  "#4d6bfe",
    "xAI":       "#000000",
    "Qwen":      "#a45ee5",
    "Meta":      "#0668e1",
    "Mistral":   "#ff7000",
}

CATEGORIES = [
    "Disability_status", "Physical_appearance", "Gender_identity",
    "SES", "Religion", "Race_ethnicity", "Sexual_orientation", "Age",
]
CATEGORY_LABEL = {
    "Disability_status":   "Disab.",
    "Physical_appearance": "Phys.",
    "Gender_identity":     "Gender",
    "SES":                 "SES",
    "Religion":            "Relig.",
    "Race_ethnicity":      "Race",
    "Sexual_orientation":  "Sexual",
    "Age":                 "Age",
}


# ── Data loading ─────────────────────────────────────────────────────────────
def model_key(path: Path, tag: str) -> str:
    suffix = f"_{tag}_results"
    return re.sub(rf"{suffix}$", "", path.stem)


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d.get("results", [])


def compute_mar(results: list[dict],
                 contrast_fallback: dict[str, bool] | None = None) -> float:
    """MAR = #(stereo wrong AND contrast right) / N. If `contrast_fallback`
    is provided (typically from the matched bbq file), pairs missing
    `contrast` in `results` use the fallback's contrast outcome instead."""
    if not results:
        return float("nan")
    n_mar = n_used = 0
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
        n_used += 1
        if (not s_correct) and c_correct:
            n_mar += 1
    if n_used == 0:
        return float("nan")
    return 100.0 * n_mar / n_used


def compute_mar_by_category(results: list[dict],
                             contrast_fallback: dict[str, bool] | None = None) -> dict[str, float]:
    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    return {c: compute_mar(rs, contrast_fallback) for c, rs in by_cat.items()}


def load_significance() -> dict:
    """Load the significance JSON and reshape into convenient lookups."""
    if not SIG_PATH.exists():
        print(f"  warning: {SIG_PATH} not found — figures will skip significance overlays.")
        return {"overall": {}, "trigger": {}, "cot": {}, "per_cat": {}}
    raw = json.loads(SIG_PATH.read_text())
    overall  = {r["model"]: r for r in raw["overall"]}
    trigger  = {r["model"]: r for r in raw["trigger"]}
    cot      = {r["model"]: r for r in raw["cot"]}
    per_cat  = {(r["model"], r["category"]): r for r in raw["per_category"]}
    return {"overall": overall, "trigger": trigger, "cot": cot, "per_cat": per_cat}


def sig_stars(p_value: float, sig_fdr: bool) -> str:
    """Asterisk markup for significance (only stars if FDR-significant)."""
    if not sig_fdr:
        return ""
    if p_value < 1e-3: return "***"
    if p_value < 1e-2: return "**"
    return "*"


def collect_model_stats() -> dict[str, dict[str, float]]:
    """
    Returns:
      {display_name: {"base_mar": float, "trigger_mar": float,
                      "direct_mar": float, "cot_mar": float,
                      "by_cat": {category: mar}}}
    """
    stats: dict[str, dict] = {}
    # Pre-build a per-model bbq contrast lookup for use as a fallback
    # when a trigger run is stereo-only.
    bbq_contrast: dict[str, dict[str, bool]] = {}
    for path in sorted(RESULTS_DIR.glob("*_bbq_results.json")):
        m = re.match(r"^(.+?)_bbq_results\.json$", path.name)
        if not m: continue
        key = m.group(1)
        if key not in MODEL_DISPLAY: continue
        try:
            rs = load_results(path)
        except Exception:
            continue
        bbq_contrast[key] = {
            r["id"]: r["responses"]["contrast"]["correct"]
            for r in rs if "contrast" in r["responses"]
        }

    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        m = re.match(r"^(.+?)_(bbq|bbq_trigger|bbq_cot|bbq_trigger_cot)_results\.json$", path.name)
        if not m:
            continue
        key, tag = m.group(1), m.group(2)
        if key not in MODEL_DISPLAY:
            continue
        disp = MODEL_DISPLAY[key]

        results = load_results(path)
        if len(results) < 1500:           # skip incomplete runs
            continue

        # For trigger runs, fall back to bbq contrast outcomes (stereo-only
        # trigger runs don't have contrast in their own JSON).
        fb = bbq_contrast.get(key) if tag in ("bbq_trigger", "bbq_trigger_cot") else None

        bucket = stats.setdefault(disp, {})
        mar = compute_mar(results, contrast_fallback=fb)
        if tag == "bbq":
            bucket["base_mar"]  = mar
            bucket["direct_mar"] = mar
            bucket["by_cat"]    = compute_mar_by_category(results)
        elif tag == "bbq_trigger":
            bucket["trigger_mar"] = mar
        elif tag == "bbq_cot":
            bucket["cot_mar"] = mar
    return stats


# ── Figure 1: dumbbell (base vs trigger MAR) ──────────────────────────────────
def fig_dumbbell(stats: dict, sig: dict, out_path: Path):
    rows = [
        (disp, s["base_mar"], s["trigger_mar"])
        for disp, s in stats.items()
        if "base_mar" in s and "trigger_mar" in s
    ]
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)  # sort by Δ (trigger − base) desc

    n = len(rows)
    fig, ax = plt.subplots(figsize=(5.8, 0.22 * n + 0.6))

    y = np.arange(n)
    for i, (disp, base, trig) in enumerate(rows):
        col = FAMILY_COLOR[family_of(disp)]
        # connecting line = absolute change
        ax.plot([base, trig], [y[i], y[i]], color=col, alpha=0.45, lw=1.4, zorder=1,
                solid_capstyle="round")
        # base = small open dot
        ax.scatter([base], [y[i]], s=30, facecolors="white",
                   edgecolors=col, linewidths=1.3, zorder=3)
        # trigger = filled larger dot
        ax.scatter([trig], [y[i]], s=70, color=col, zorder=3, edgecolors="white",
                   linewidths=0.8)
        # Δ annotation (percentage-point difference), right of trigger
        delta = trig - base
        sign  = "+" if delta >= 0 else "−"
        ax.text(trig + 0.8, y[i], f"{sign}{abs(delta):.1f}",
                va="center", ha="left", fontsize=7, color=col, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels([d for d, _, _ in rows], fontsize=8.5, rotation=0)
    for tick, (disp, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(FAMILY_COLOR[family_of(disp)])

    ax.invert_yaxis()
    ax.set_xlabel("MAR (%)")
    ax.set_xlim(left=0)
    ax.set_xticks(np.arange(0, 45, 5))
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    base_handle = plt.scatter([], [], s=30, facecolors="white",
                              edgecolors="#555", linewidths=1.3, label="Base")
    trig_handle = plt.scatter([], [], s=70, color="#555", label="Priming",
                              edgecolors="white", linewidths=0.8)
    ax.legend(handles=[base_handle, trig_handle],
              loc="lower right", frameon=False, fontsize=8, handletextpad=0.4)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} models)")


# ── Figure 2: CoT grouped horizontal bars ────────────────────────────────────
def fig_cot(stats: dict, sig: dict, out_path: Path):
    rows = [
        (disp, s["direct_mar"], s["cot_mar"])
        for disp, s in stats.items()
        if "direct_mar" in s and "cot_mar" in s
    ]
    # CoT-worsens at top, CoT-improves at bottom (largest positive Δ first)
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(4.8, 0.45 * n + 0.9))

    # Hatch patterns distinguish Direct vs CoT, while bar color stays
    # family-coded (matches the dumbbell figure's family scheme).
    DIRECT_HATCH = "///"     # forward slashes
    COT_HATCH    = "..."     # asterisks (try "\\\\\\", "xxx", "...", "ooo" for variants)

    y = np.arange(n)
    bar_h = 0.38
    for i, (disp, direct, cot) in enumerate(rows):
        col = FAMILY_COLOR[family_of(disp)]
        ax.barh(y[i] - bar_h / 2, direct, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=DIRECT_HATCH, zorder=2)
        ax.barh(y[i] + bar_h / 2, cot, height=bar_h,
                facecolor=col, edgecolor="white", linewidth=0.6,
                hatch=COT_HATCH, zorder=2)

        # Numeric annotations: text in family color
        d_end = direct + 0.15
        c_end = cot    + 0.15
        ax.text(d_end, y[i] - bar_h / 2, f"{direct:.1f}",
                va="center", ha="left", fontsize=7, color=col)
        ax.text(c_end, y[i] + bar_h / 2, f"{cot:.1f}",
                va="center", ha="left", fontsize=7, color=col)

    ax.set_yticks(y)
    ax.set_yticklabels([d for d, _, _ in rows], fontsize=8.5)
    # Color y-tick labels by family — same scheme as the dumbbell figure
    for tick, (disp, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(FAMILY_COLOR[family_of(disp)])
    ax.invert_yaxis()
    ax.set_xlabel("MAR (%)")
    ax.set_xlim(0, max(max(d, c) for _, d, c in rows) * 1.6)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Legend: neutral gray rectangles with the two hatches, so the legend
    # encodes the *pattern* (the variable that actually distinguishes Direct
    # from CoT in the figure), independent of family color.
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
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} models)")


# ── Icon placement helper ────────────────────────────────────────────────────
_ICON_CACHE: dict[str, np.ndarray] = {}

def _load_icon(family: str):
    """Load icon for a family, with caching. Returns None if missing.

    Supports two source formats. SVG is rasterized at the target buffer
    height directly via cairosvg — ideal because the rasterization is
    vector-perfect at any resolution. PNG/JPEG sources are resized via
    Pillow's LANCZOS filter. Either way the buffer ends up at
    ICON_TARGET_PX * ICON_OVERSAMPLE pixels in height so the embedded
    raster stays crisp when the PDF is zoomed in.
    """
    fname = ICON_FILE.get(family)
    if fname is None:
        return None
    if family in _ICON_CACHE:
        return _ICON_CACHE[family]
    # If the user gave a name without extension, prefer .svg, fall back to .png
    candidates = ([ICON_DIR / fname] if Path(fname).suffix
                  else [ICON_DIR / f"{fname}.svg",
                        ICON_DIR / f"{fname}.png",
                        ICON_DIR / f"{fname}.jpg"])
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        _ICON_CACHE[family] = None
        return None

    target_h = ICON_TARGET_PX * ICON_OVERSAMPLE
    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            # Rasterize the SVG vector source at exactly the target height.
            import io
            import cairosvg
            from PIL import Image as PILImage
            buf = io.BytesIO()
            cairosvg.svg2png(url=str(path), write_to=buf,
                              output_height=target_h)
            buf.seek(0)
            pil = PILImage.open(buf)
            if pil.mode != "RGBA":
                pil = pil.convert("RGBA")
            arr = np.asarray(pil).astype(np.float32) / 255.0
        else:
            # Raster source — high-quality downsample via Pillow LANCZOS.
            from PIL import Image as PILImage
            pil = PILImage.open(path)
            if pil.mode != "RGBA":
                pil = pil.convert("RGBA")
            new_w = max(1, int(round(pil.width * target_h / pil.height)))
            pil = pil.resize((new_w, target_h), PILImage.LANCZOS)
            arr = np.asarray(pil).astype(np.float32) / 255.0
    except Exception as e:
        print(f"  warning: failed to load {path}: {e}")
        _ICON_CACHE[family] = None
        return None
    _ICON_CACHE[family] = arr
    return arr


def _add_family_icons(ax, families_per_row, model_per_row,
                       target_px=ICON_TARGET_PX):
    """Place a family icon to the LEFT of each y-axis tick label.

    Each icon's x position is read from `ICON_X_PER_ROW[model_display_name]`
    (axes-fraction units). Missing entries fall back to `ICON_X_DEFAULT`.
    Edit those constants near the top of the file to nudge each row's icon
    position individually.

    `model_per_row[i]` is the display name for row i; `families_per_row[i]`
    is the family used to look up the icon file.
    """
    transform = ax.get_yaxis_transform()  # x: axes frac, y: data
    for i, (fam, model) in enumerate(zip(families_per_row, model_per_row)):
        img = _load_icon(fam)
        if img is None:
            continue
        x_frac = ICON_X_PER_ROW.get(model, ICON_X_DEFAULT)
        zoom = 1.0 / ICON_OVERSAMPLE
        oi = OffsetImage(img, zoom=zoom,
                          interpolation=ICON_INTERPOLATION, resample=True)
        ab = AnnotationBbox(
            oi,
            (x_frac, i),
            xycoords=transform,
            frameon=False,
            box_alignment=(1, 0.5),  # right-aligned: icon's right edge sits at x_frac
            pad=0,
            annotation_clip=False,
        )
        ax.add_artist(ab)


# ── Figure 3: per-category heatmap ────────────────────────────────────────────
def fig_heatmap(stats: dict, sig: dict, out_path: Path):
    rows = [
        (disp, s["base_mar"], s["by_cat"])
        for disp, s in stats.items()
        if "base_mar" in s and "by_cat" in s
    ]
    rows.sort(key=lambda r: r[1], reverse=True)        # sort by overall MAR desc

    matrix = np.array([[r[2].get(c, np.nan) for c in CATEGORIES] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 0.22 * len(rows) + 1.0))

    vmax = np.nanpercentile(matrix, 97)
    im = ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORIES],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} ({r[1]:.1f})" for r in rows], fontsize=8)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(FAMILY_COLOR[family_of(row[0])])

    # Family icons to the left of the y-axis (silently skipped if missing)
    _add_family_icons(ax,
                       families_per_row=[family_of(r[0]) for r in rows],
                       model_per_row=[r[0] for r in rows])

    # Cell annotations: raw MAR values (no significance overlay).
    for i in range(len(rows)):
        for j in range(len(CATEGORIES)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MAR (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("BBQ category")

    plt.tight_layout()
    # dpi=300 makes the embedded raster (icons) crisp at PDF zoom levels
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_path}  ({len(rows)} models × {len(CATEGORIES)} cats)")


# ── Figure 4: combined heatmap + dumbbell (shared model y-axis) ───────────────
def fig_combined(stats: dict, out_path: Path):
    """One figure: heatmap on the left, dumbbell on the right, sharing the
    model labels on the y-axis. Models sorted by overall (base) MAR descending
    — the natural order for the heatmap; the dumbbell still reads the trigger
    Δ via its dot positions and inline annotations."""
    rows = [
        (disp, s["base_mar"], s.get("trigger_mar"), s["by_cat"])
        for disp, s in stats.items()
        if "base_mar" in s and "by_cat" in s
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    n = len(rows)
    matrix = np.array([[r[3].get(c, np.nan) for c in CATEGORIES] for r in rows])
    fams = [family_of(r[0]) for r in rows]

    # ── Layout: heatmap (left) + dumbbell (right), shared y-axis ──
    fig = plt.figure(figsize=(8.0, 0.22 * n + 1.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.05)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1], sharey=ax_h)

    # ── Left: heatmap ──
    vmax = np.nanpercentile(matrix, 97)
    im = ax_h.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
    for i in range(n):
        for j in range(len(CATEGORIES)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > vmax * 0.55 else "black"
            ax_h.text(j, i, f"{v:.1f}", ha="center", va="center",
                      fontsize=6.2, color=color)
    ax_h.set_xticks(range(len(CATEGORIES)))
    ax_h.set_xticklabels([CATEGORY_LABEL[c] for c in CATEGORIES],
                          rotation=35, ha="right", fontsize=8)
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels([f"{r[0]} ({r[1]:.1f})" for r in rows], fontsize=8)
    for tick, fam in zip(ax_h.get_yticklabels(), fams):
        tick.set_color(FAMILY_COLOR[fam])
    ax_h.set_xlabel("BBQ category", fontsize=9)
    # No colorbar — cells already annotate raw MAR. Keep spines clean.
    ax_h.set_xlim(-0.5, len(CATEGORIES) - 0.5)

    # ── Right: dumbbell ──
    has_trigger = [(i, r) for i, r in enumerate(rows) if r[2] is not None]
    for i, (disp, base, trig, _) in enumerate(rows):
        col = FAMILY_COLOR[fams[i]]
        if trig is None:
            ax_d.scatter([base], [i], s=30, facecolors="white",
                         edgecolors=col, linewidths=1.3, zorder=3)
            continue
        ax_d.plot([base, trig], [i, i], color=col, alpha=0.45, lw=1.4,
                  zorder=1, solid_capstyle="round")
        ax_d.scatter([base], [i], s=30, facecolors="white",
                     edgecolors=col, linewidths=1.3, zorder=3)
        ax_d.scatter([trig], [i], s=70, color=col, zorder=3,
                     edgecolors="white", linewidths=0.8)
        delta = trig - base
        sign = "+" if delta >= 0 else "−"
        ax_d.text(trig + 0.8, i, f"{sign}{abs(delta):.1f}",
                  va="center", ha="left", fontsize=6.5, color=col, alpha=0.9)

    ax_d.set_xlim(left=0)
    xmax_data = max(
        (r[2] for r in rows if r[2] is not None), default=max(r[1] for r in rows)
    )
    ax_d.set_xlim(0, xmax_data * 1.1 + 4)
    step = 5 if xmax_data < 35 else 10
    ax_d.set_xticks(np.arange(0, xmax_data * 1.1 + 4, step))
    ax_d.set_xlabel("MAR (%)", fontsize=9)
    ax_d.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7, zorder=0)
    ax_d.set_axisbelow(True)
    plt.setp(ax_d.get_yticklabels(), visible=False)
    ax_d.tick_params(axis="y", which="both", left=False)
    ax_d.spines["left"].set_visible(False)

    # Dumbbell legend — placed in lower-right of the dumbbell panel
    base_handle = plt.scatter([], [], s=30, facecolors="white",
                              edgecolors="#555", linewidths=1.3, label="Base")
    trig_handle = plt.scatter([], [], s=70, color="#555",
                              edgecolors="white", linewidths=0.8, label="Trigger")
    ax_d.legend(handles=[base_handle, trig_handle],
                loc="lower right", frameon=False, fontsize=7,
                handletextpad=0.3, borderpad=0.3)

    # Invert y so highest-MAR model is at the top in BOTH panels
    ax_h.invert_yaxis()

    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}  ({n} models, combined heatmap + dumbbell)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading results from {RESULTS_DIR} ...")
    stats = collect_model_stats()
    print(f"  {len(stats)} models matched")

    print(f"Loading significance from {SIG_PATH} ...")
    sig = load_significance()

    fig_dumbbell(stats, sig, OUT_DIR / "mar_dumbbell.pdf")
    fig_cot     (stats, sig, OUT_DIR / "cot_slope.pdf")
    fig_heatmap (stats, sig, OUT_DIR / "per_category_heatmap.pdf")
    fig_combined(stats,      OUT_DIR / "mar_heatmap_dumbbell.pdf")
    print("done.")


if __name__ == "__main__":
    main()
