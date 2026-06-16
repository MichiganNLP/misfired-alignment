"""
Render the per-(model, BBQ category) MAR heatmap as a LaTeX table with
colored cells, instead of as a matplotlib figure. Produces a vector-quality
table where:

  * model names render with the paper's native LaTeX font
  * each cell is shaded via `\\cellcolor[HTML]{...}` based on its MAR value
  * an optional family icon is embedded with `\\includegraphics{...}`
  * the user controls layout entirely from the LaTeX side (column widths,
    row spacing, font sizes), no matplotlib idiosyncrasies

Output:  paper/per_category_heatmap_table.tex

To use in the paper preamble:
    \\usepackage[table]{xcolor}     % or just \\usepackage{xcolor}
    \\usepackage{colortbl}
    \\usepackage{graphicx}
    \\usepackage{booktabs}

Then \\input{per_category_heatmap_table} inside any \\begin{table*} or
\\begin{figure*} environment in the main body.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

PROJ_DIR    = Path(__file__).parent.parent
RESULTS_DIR = PROJ_DIR / "results"
OUT_TEX     = PROJ_DIR / "paper" / "per_category_heatmap_table.tex"
ICON_REL    = "images/icons"   # path relative to the .tex's includegraphics root

# ── Display config (mirrors plot_paper_figures.py) ────────────────────────────
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


def family_of(disp: str) -> str:
    if disp.startswith("GPT"):           return "openai"
    if disp.startswith("Claude"):        return "anthropic"
    if disp.startswith("Gemini") or disp.startswith("Gemma"): return "google"
    if disp.startswith("DeepSeek"):      return "deepseek"
    if disp.startswith("Grok"):          return "xai"
    if disp.startswith("Qwen"):          return "qwen"
    if disp.startswith("Llama"):         return "meta"
    if disp.startswith("Mistral"):       return "mistral"
    return ""


# ── Data loading ──────────────────────────────────────────────────────────────
def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("results", [])


def compute_mar_by_cat(results: list[dict],
                        contrast_fallback: dict[str, bool] | None = None
                        ) -> tuple[dict[str, float], float]:
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "mar": 0})
    n_total = mar_total = 0
    for r in results:
        if "stereotyped" not in r["responses"]:
            continue
        s = r["responses"]["stereotyped"]["correct"]
        if "contrast" in r["responses"]:
            c = r["responses"]["contrast"]["correct"]
        elif contrast_fallback is not None and r["id"] in contrast_fallback:
            c = contrast_fallback[r["id"]]
        else:
            continue
        cat = r["category"]
        is_mar = (not s) and c
        by_cat[cat]["n"] += 1
        by_cat[cat]["mar"] += int(is_mar)
        n_total += 1
        mar_total += int(is_mar)
    cat_mar = {c: 100 * st["mar"] / max(st["n"], 1) for c, st in by_cat.items()}
    overall = 100 * mar_total / max(n_total, 1)
    return cat_mar, overall


def collect() -> list[tuple[str, float, dict[str, float]]]:
    """Return [(display_name, overall_mar, {category: mar})], sorted by overall desc."""
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        path = RESULTS_DIR / f"{key}_bbq_results.json"
        if not path.exists():
            continue
        results = load_results(path)
        if len(results) < 100:
            continue
        cat_mar, overall = compute_mar_by_cat(results)
        rows.append((disp, overall, cat_mar))
    rows.sort(key=lambda r: -r[1])
    return rows


# ── LaTeX rendering ──────────────────────────────────────────────────────────

def mar_to_hex(mar: float, vmax: float, cmap_name: str = "Reds") -> str:
    """Map MAR (%) to a hex color via the given matplotlib colormap, gamma-toned
    so low values aren't pure white (helps cells stay visible)."""
    if np.isnan(mar):
        return "FFFFFF"
    norm = mcolors.Normalize(vmin=0, vmax=vmax, clip=True)
    rgba = cm.get_cmap(cmap_name)(norm(mar))
    r, g, b = (int(round(255 * c)) for c in rgba[:3])
    return f"{r:02X}{g:02X}{b:02X}"


def text_color_for_bg(hex_color: str) -> str:
    """Return 'white' if the bg color is dark enough that black text would be
    hard to read, else 'black'. Uses perceptual luminance."""
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 128 else "black"


def render(rows, out_path: Path,
            cmap: str = "Reds",
            vmax_pct: float | None = None,
            icon_height: str = "1.6ex"):
    """Write a single LaTeX table fragment to out_path."""
    matrix = np.array([[r[2].get(c, np.nan) for c, _ in CATEGORIES] for r in rows])
    if vmax_pct is None:
        vmax_pct = float(np.nanpercentile(matrix, 97))

    cat_keys, cat_labels = zip(*CATEGORIES)
    n_cols = 1 + len(CATEGORIES) + 1   # model | 8 cats | overall
    col_spec = "@{}l@{\\hspace{4pt}}" + "r" * len(CATEGORIES) + "@{\\hspace{6pt}}r@{}"

    lines = []
    lines.append(
        "% Auto-generated by scripts/render_mar_heatmap_table.py — do not edit by hand."
    )
    lines.append(
        "% Required preamble: \\usepackage[table]{xcolor}, \\usepackage{colortbl},"
        " \\usepackage{graphicx}, \\usepackage{booktabs}."
    )
    lines.append("")
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = (
        "\\textbf{Model} & "
        + " & ".join(f"\\textbf{{{lbl}}}" for lbl in cat_labels)
        + " & \\textbf{Overall} \\\\"
    )
    lines.append(header)
    lines.append("\\midrule")

    for disp, overall, cat_mar in rows:
        fam = family_of(disp)
        # Icon next to the model name (graceful no-op if file missing —
        # LaTeX will just print a warning at compile time).
        if fam:
            icon = f"\\raisebox{{-0.2ex}}{{\\includegraphics[height={icon_height}]{{{ICON_REL}/{fam}}}}}\\,"
        else:
            icon = ""
        cells = [f"{icon}{disp}"]
        for cat in cat_keys:
            v = cat_mar.get(cat)
            if v is None or np.isnan(v):
                cells.append("--")
                continue
            bg   = mar_to_hex(v, vmax_pct, cmap)
            fg   = text_color_for_bg(bg)
            text = f"{v:.1f}"
            if fg == "white":
                cells.append(f"\\cellcolor[HTML]{{{bg}}}{{\\color{{white}}{text}}}")
            else:
                cells.append(f"\\cellcolor[HTML]{{{bg}}}{text}")
        # Overall column
        bg   = mar_to_hex(overall, vmax_pct, cmap)
        fg   = text_color_for_bg(bg)
        text = f"{overall:.1f}"
        if fg == "white":
            cells.append(f"\\cellcolor[HTML]{{{bg}}}{{\\color{{white}}\\textbf{{{text}}}}}")
        else:
            cells.append(f"\\cellcolor[HTML]{{{bg}}}\\textbf{{{text}}}")
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"  {len(rows)} models, vmax = {vmax_pct:.1f}% (97th percentile of cell MARs)")


def main():
    rows = collect()
    render(rows, OUT_TEX)


if __name__ == "__main__":
    main()
