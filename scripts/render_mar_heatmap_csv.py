"""
Per-(model, BBQ category) MAR table as a plain CSV.

Columns: Model, Disab., Phys., Gender, SES, Relig., Race, Sexual, Age, Overall

Output: paper/per_category_heatmap.csv
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJ_DIR    = Path(__file__).parent.parent
RESULTS_DIR = PROJ_DIR / "results"
OUT_CSV     = PROJ_DIR / "paper" / "per_category_heatmap.csv"

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


def mar_per_cat(results: list[dict]) -> tuple[dict[str, float], float]:
    by_cat = defaultdict(lambda: {"n": 0, "mar": 0})
    n_total = mar_total = 0
    for r in results:
        if "stereotyped" not in r["responses"] or "contrast" not in r["responses"]:
            continue
        s = r["responses"]["stereotyped"]["correct"]
        c = r["responses"]["contrast"]["correct"]
        by_cat[r["category"]]["n"] += 1
        is_mar = (not s) and c
        by_cat[r["category"]]["mar"] += int(is_mar)
        n_total += 1
        mar_total += int(is_mar)
    cat_mar = {c: 100 * st["mar"] / max(st["n"], 1) for c, st in by_cat.items()}
    return cat_mar, 100 * mar_total / max(n_total, 1)


def main():
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        path = RESULTS_DIR / f"{key}_bbq_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            results = json.load(f).get("results", [])
        if len(results) < 100:
            continue
        cat_mar, overall = mar_per_cat(results)
        rows.append((disp, overall, cat_mar))
    rows.sort(key=lambda r: -r[1])

    cat_keys = [c for c, _ in CATEGORIES]
    cat_labels = [s for _, s in CATEGORIES]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + cat_labels + ["Overall"])
        for disp, overall, cat_mar in rows:
            row = [disp]
            for c in cat_keys:
                v = cat_mar.get(c)
                row.append(f"{v:.1f}" if v is not None else "")
            row.append(f"{overall:.1f}")
            w.writerow(row)
    print(f"Wrote {OUT_CSV}  ({len(rows)} models)")


if __name__ == "__main__":
    main()
