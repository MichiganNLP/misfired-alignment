"""
Regenerate paper/significance.tex from current eval results.

Tests (matching the existing TeX schema):
  Test 1 (overall MAR asymmetry)   — one-sided McNemar exact-binomial, c > b
  Test 2 (trigger effect)          — one-sided paired McNemar on MAR-failure indicator
  Test 3 (CoT effect)              — two-sided paired McNemar on MAR-failure indicator
  Test 4 (Test 1 per category)     — one-sided McNemar within each (model, category)

For each test family we apply Benjamini–Hochberg FDR correction at q = 0.05.
95% CIs are pair-level percentile bootstraps (B = 10,000).

Usage:
    python scripts/compute_significance.py

Writes:
    paper/significance.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
from math import comb

import numpy as np

PROJ    = Path(__file__).parent.parent
RESULTS = PROJ / "results"
OUT_TEX = PROJ / "paper" / "significance.tex"

BOOTSTRAP_B = 10_000
ALPHA       = 0.05
SEED        = 42

# Display order / names (mirrors plot_paper_figures.py)
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


# ── IO helpers ───────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f).get("results", [])


def index_by_id(results: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in results}


def get_correct(rec: dict, condition: str) -> bool | None:
    cond = rec.get("responses", {}).get(condition)
    return cond.get("correct") if cond is not None else None


# ── Statistical primitives ───────────────────────────────────────────────────

def binom_sf_one_sided(c: int, n: int, p: float = 0.5) -> float:
    """Right-tail sf: P(X >= c) where X ~ Binomial(n, p). Exact summation."""
    if c <= 0:
        return 1.0
    if c > n:
        return 0.0
    # Sum P(X = k) for k = c..n, computed exactly via comb
    s = 0.0
    for k in range(c, n + 1):
        s += comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    # numerical safety
    return min(max(s, 0.0), 1.0)


def mcnemar_one_sided(b: int, c: int) -> float:
    """One-sided McNemar exact binomial, alternative c > b.
    Treat the b+c discordant pairs as Binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return binom_sf_one_sided(c, n, 0.5)


def mcnemar_two_sided(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar = 2 * one-sided, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    p_one = binom_sf_one_sided(max(b, c), n, 0.5)
    return min(2 * p_one, 1.0)


def bh_fdr(pvals: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Benjamini–Hochberg FDR. Returns (q-values, sig-mask)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = p[order] * n / ranks
    # Enforce monotonicity from the right
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q = np.zeros(n)
    q[order] = np.minimum(q_sorted, 1.0)
    sig = q <= alpha
    return q.tolist(), sig.tolist()


def stars_for_p(p: float, q: float, sig_fdr: bool) -> str:
    """Star annotation: stars only if FDR-significant; thresholds use raw p."""
    if not sig_fdr:
        return ""
    if p < 1e-3: return "$^{***}$"
    if p < 1e-2: return "$^{**}$"
    if p < 5e-2: return "$^{*}$"
    return ""


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    if p < 1e-3:
        return f"${p:.1e}$".replace("e-0", "e-")
    return f"{p:.3f}"


def bootstrap_mar_ci(b_arr: np.ndarray, c_arr: np.ndarray,
                      n_pairs: int, B: int = BOOTSTRAP_B,
                      seed: int = SEED) -> tuple[float, float]:
    """Bootstrap percentile CI for MAR rate (= mean of c_arr)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pairs, size=(B, n_pairs))
    boot = c_arr[idx].mean(axis=1) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_paired_delta_ci(z_a: np.ndarray, z_b: np.ndarray,
                               B: int = BOOTSTRAP_B,
                               seed: int = SEED) -> tuple[float, float]:
    """Bootstrap percentile CI for Δ = mean(z_b) - mean(z_a) on paired data."""
    n = len(z_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    delta = (z_b[idx].mean(axis=1) - z_a[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return float(lo), float(hi)


# ── Per-test computation ─────────────────────────────────────────────────────

def mar_indicator(rec: dict, contrast_fallback: dict[str, bool] | None = None) -> int | None:
    """MAR indicator for one pair = 1 iff stereo wrong AND contrast right."""
    s = get_correct(rec, "stereotyped")
    if s is None:
        return None
    c = get_correct(rec, "contrast")
    if c is None and contrast_fallback is not None:
        c = contrast_fallback.get(rec["id"])
    if c is None:
        return None
    return int((not s) and c)


def reverse_indicator(rec: dict) -> int | None:
    """Reverse indicator (stereo right AND contrast wrong) — for Test 1."""
    s = get_correct(rec, "stereotyped")
    c = get_correct(rec, "contrast")
    if s is None or c is None:
        return None
    return int(s and (not c))


def collect_per_pair(json_path: Path,
                      contrast_fallback: dict[str, bool] | None = None) -> dict[str, np.ndarray]:
    """Return arrays of per-pair indicators (MAR, reverse) and pair_ids."""
    rs = load_results(json_path)
    ids, mar, rev = [], [], []
    for r in rs:
        k = mar_indicator(r, contrast_fallback)
        if k is None:
            continue
        ids.append(r["id"]); mar.append(k)
        rv = reverse_indicator(r)
        rev.append(rv if rv is not None else 0)
    return {"ids": np.array(ids), "mar": np.array(mar, dtype=int),
            "rev": np.array(rev, dtype=int)}


def collect_per_pair_per_cat(json_path: Path,
                              contrast_fallback: dict[str, bool] | None = None
                              ) -> dict[str, dict]:
    """Same as collect_per_pair but split by category."""
    rs = load_results(json_path)
    by_cat: dict[str, dict] = defaultdict(lambda: {"ids": [], "mar": [], "rev": []})
    for r in rs:
        k = mar_indicator(r, contrast_fallback)
        if k is None:
            continue
        cat = r["category"]
        by_cat[cat]["ids"].append(r["id"])
        by_cat[cat]["mar"].append(k)
        rv = reverse_indicator(r)
        by_cat[cat]["rev"].append(rv if rv is not None else 0)
    return {c: {"ids": np.array(v["ids"]),
                "mar": np.array(v["mar"], dtype=int),
                "rev": np.array(v["rev"], dtype=int)}
            for c, v in by_cat.items()}


def build_contrast_fallback(key: str) -> dict[str, bool] | None:
    """For trigger / trigger_cot, look up bbq's contrast outcomes by pair_id."""
    bbq = RESULTS / f"{key}_bbq_results.json"
    if not bbq.exists():
        return None
    fb = {}
    for r in load_results(bbq):
        c = get_correct(r, "contrast")
        if c is not None:
            fb[r["id"]] = c
    return fb


# ── Test 1: overall MAR asymmetry ────────────────────────────────────────────

def run_test1() -> list[dict]:
    """One-sided McNemar c > b on overall MAR per model."""
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        path = RESULTS / f"{key}_bbq_results.json"
        if not path.exists():
            continue
        per = collect_per_pair(path)
        if len(per["mar"]) < 100:
            continue
        n = len(per["mar"])
        c = int(per["mar"].sum())
        b = int(per["rev"].sum())
        mar_pct = 100 * c / n
        ci_lo, ci_hi = bootstrap_mar_ci(per["rev"], per["mar"], n)
        p = mcnemar_one_sided(b, c)
        rows.append({
            "key": key, "model": disp,
            "n": n, "mar": mar_pct, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "b": b, "c": c, "p": p,
        })
    pvals = [r["p"] for r in rows]
    qs, sigs = bh_fdr(pvals)
    for r, q, s in zip(rows, qs, sigs):
        r["q"] = q; r["sig_fdr"] = s
    return rows


# ── Test 2: per-model trigger effect (paired McNemar on MAR indicator) ───────

def run_test2() -> list[dict]:
    """Paired one-sided McNemar: trigger amplifies MAR over base."""
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        bbq = RESULTS / f"{key}_bbq_results.json"
        trg = RESULTS / f"{key}_bbq_trigger_results.json"
        if not (bbq.exists() and trg.exists()):
            continue
        # Trigger may be stereo-only — fall back to bbq contrast outcomes
        fb = build_contrast_fallback(key)
        bbq_per = collect_per_pair(bbq)
        trg_per = collect_per_pair(trg, contrast_fallback=fb)
        # Align by pair_id
        bbq_idx = {pid: i for i, pid in enumerate(bbq_per["ids"])}
        common = [pid for pid in trg_per["ids"] if pid in bbq_idx]
        if len(common) < 100:
            continue
        z_base = np.array([bbq_per["mar"][bbq_idx[pid]] for pid in common], dtype=int)
        z_trig = np.array([trg_per["mar"][list(trg_per["ids"]).index(pid)]
                            for pid in common], dtype=int)
        # Paired McNemar on MAR indicator: discordances are b/c
        b = int(((z_base == 1) & (z_trig == 0)).sum())  # MAR removed by trigger
        c = int(((z_base == 0) & (z_trig == 1)).sum())  # MAR introduced by trigger
        mar_base = 100 * z_base.mean()
        mar_trig = 100 * z_trig.mean()
        delta    = mar_trig - mar_base
        ci_lo, ci_hi = bootstrap_paired_delta_ci(z_base, z_trig)
        p = mcnemar_one_sided(b, c)
        rows.append({
            "key": key, "model": disp,
            "n": len(common),
            "mar_base": mar_base, "mar_trigger": mar_trig,
            "delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "b": b, "c": c, "p": p,
        })
    pvals = [r["p"] for r in rows]
    qs, sigs = bh_fdr(pvals)
    for r, q, s in zip(rows, qs, sigs):
        r["q"] = q; r["sig_fdr"] = s
    return rows


# ── Test 3: per-model CoT effect (paired two-sided McNemar) ──────────────────

def run_test3() -> list[dict]:
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        bbq = RESULTS / f"{key}_bbq_results.json"
        cot = RESULTS / f"{key}_bbq_cot_results.json"
        if not (bbq.exists() and cot.exists()):
            continue
        bbq_per = collect_per_pair(bbq)
        cot_per = collect_per_pair(cot)
        bbq_idx = {pid: i for i, pid in enumerate(bbq_per["ids"])}
        common = [pid for pid in cot_per["ids"] if pid in bbq_idx]
        if len(common) < 100:
            continue
        z_dir = np.array([bbq_per["mar"][bbq_idx[pid]] for pid in common], dtype=int)
        cot_idx = {pid: i for i, pid in enumerate(cot_per["ids"])}
        z_cot = np.array([cot_per["mar"][cot_idx[pid]] for pid in common], dtype=int)
        b = int(((z_dir == 1) & (z_cot == 0)).sum())
        c = int(((z_dir == 0) & (z_cot == 1)).sum())
        mar_dir = 100 * z_dir.mean()
        mar_cot = 100 * z_cot.mean()
        delta   = mar_cot - mar_dir
        ci_lo, ci_hi = bootstrap_paired_delta_ci(z_dir, z_cot)
        p = mcnemar_two_sided(b, c)
        rows.append({
            "key": key, "model": disp,
            "n": len(common),
            "mar_direct": mar_dir, "mar_cot": mar_cot,
            "delta": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "b": b, "c": c, "p": p,
        })
    pvals = [r["p"] for r in rows]
    qs, sigs = bh_fdr(pvals)
    for r, q, s in zip(rows, qs, sigs):
        r["q"] = q; r["sig_fdr"] = s
    return rows


# ── Test 4: Test 1 per category ──────────────────────────────────────────────

def run_test4() -> list[dict]:
    """Per (model, category) one-sided McNemar c > b. BH-FDR across all cells."""
    rows = []
    for key, disp in MODEL_DISPLAY.items():
        path = RESULTS / f"{key}_bbq_results.json"
        if not path.exists():
            continue
        per_cat = collect_per_pair_per_cat(path)
        for cat_key, _ in CATEGORIES:
            arr = per_cat.get(cat_key)
            if arr is None or len(arr["mar"]) == 0:
                continue
            n = len(arr["mar"])
            c = int(arr["mar"].sum())
            b = int(arr["rev"].sum())
            mar_pct = 100 * c / n
            p = mcnemar_one_sided(b, c)
            rows.append({
                "key": key, "model": disp, "category": cat_key,
                "n": n, "mar": mar_pct, "b": b, "c": c, "p": p,
            })
    pvals = [r["p"] for r in rows]
    qs, sigs = bh_fdr(pvals)
    for r, q, s in zip(rows, qs, sigs):
        r["q"] = q; r["sig_fdr"] = s
    return rows


# ── TeX rendering ────────────────────────────────────────────────────────────

def fmt_table1(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: -r["mar"])
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Test 1 (per-model overall MAR asymmetry). One-sided McNemar's "
        r"exact-binomial test with directional alternative "
        r"$P(\textrm{stereo wrong, contrast right}) > P(\textrm{stereo right, contrast wrong})$, "
        r"matching the paper's pre-specified overalignment hypothesis. 95\% CI from pair-level "
        r"percentile bootstrap ($B=10{,}000$). $q$-values via Benjamini--Hochberg at FDR $0.05$ "
        r"within this test family. $n$ here is "
        + f"${rows[0]['n']:,}$." + "}")
    lines.append(r"\label{tab:sig_overall}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{MAR (\%)} & \textbf{95\% CI} & $\bm{p}$ & $\bm{q}$ (BH) \\")
    lines.append(r"\midrule")
    for r in rows:
        stars = stars_for_p(r["p"], r["q"], r["sig_fdr"])
        lines.append(
            f"{r['model']}  & {r['mar']:.2f}{stars} & "
            f"[{r['ci_lo']:.2f},\\,{r['ci_hi']:.2f}] & "
            f"{fmt_p(r['p'])} & {fmt_p(r['q'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def fmt_table2(rows: list[dict]) -> str:
    # Sort by Δ descending (most amplification first)
    rows = sorted(rows, key=lambda r: -r["delta"])
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Test 2 (per-model trigger effect). One-sided paired McNemar's exact-binomial "
        r"test on the MAR-failure indicator $\mathbf{1}\{\text{stereo wrong} \wedge "
        r"\text{contrast right}\}$ with directional alternative that the trigger introduces more "
        r"new MAR failures than it removes (matching the paper's `amplifies' hypothesis). "
        r"$\Delta$ is the percentage-point change in MAR (trigger $-$ base); 95\% CI is paired "
        r"pair-level percentile bootstrap. BH FDR within this test family.}")
    lines.append(r"\label{tab:sig_trigger}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & $\bm{n}$ & \textbf{Base} & \textbf{Trigger} & "
                 r"$\bm{\Delta}$ \textbf{(95\% CI)} & $\bm{p}$ & $\bm{q}$ (BH) \\")
    lines.append(r"\midrule")
    for r in rows:
        stars = stars_for_p(r["p"], r["q"], r["sig_fdr"])
        sign  = "+" if r["delta"] >= 0 else "$-$"
        lines.append(
            f"{r['model']}  & {r['n']:,} & {r['mar_base']:.2f} & {r['mar_trigger']:.2f} & "
            f"{sign}{abs(r['delta']):.2f}{stars} "
            f"[{r['ci_lo']:+.2f},\\,{r['ci_hi']:+.2f}] & "
            f"{fmt_p(r['p'])} & {fmt_p(r['q'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def fmt_table3(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: -r["delta"])
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Test 3 (per-model CoT effect). Two-sided paired McNemar's exact-binomial test "
        r"on the MAR-failure indicator comparing direct and CoT prompting on the same pairs "
        r"(two-sided because the direction varies across model classes — open-weight models "
        r"worsen under CoT while frontier API models improve). $\Delta$ is the percentage-point "
        r"change (CoT $-$ direct); 95\% CI is paired bootstrap. BH FDR within this test family.}")
    lines.append(r"\label{tab:sig_cot}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & $\bm{n}$ & \textbf{Direct} & \textbf{CoT} & "
                 r"$\bm{\Delta}$ \textbf{(95\% CI)} & $\bm{p}$ & $\bm{q}$ (BH) \\")
    lines.append(r"\midrule")
    for r in rows:
        stars = stars_for_p(r["p"], r["q"], r["sig_fdr"])
        sign = "+" if r["delta"] >= 0 else "$-$"
        lines.append(
            f"{r['model']}  & {r['n']:,} & {r['mar_direct']:.2f} & {r['mar_cot']:.2f} & "
            f"{sign}{abs(r['delta']):.2f}{stars} "
            f"[{r['ci_lo']:+.2f},\\,{r['ci_hi']:+.2f}] & "
            f"{fmt_p(r['p'])} & {fmt_p(r['q'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def fmt_table4(rows: list[dict]) -> str:
    # Group by model, ordered by overall MAR (descending). Compute per-model overall.
    overall = defaultdict(float); overall_n = defaultdict(int)
    cell = {}
    for r in rows:
        cell[(r["model"], r["category"])] = r
        overall[r["model"]] += r["c"]; overall_n[r["model"]] += r["n"]
    overall_mar = {m: 100 * overall[m] / overall_n[m] for m in overall}
    models_sorted = sorted(overall_mar, key=overall_mar.get, reverse=True)
    cats = [c for c, _ in CATEGORIES]
    cat_labels = [s for _, s in CATEGORIES]

    n_cells = len(rows)
    n_models = len(models_sorted)
    n_cats   = len(cats)

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Test 1 per category. MAR (\%) for each (model, BBQ category) cell with stars "
        r"marking one-sided McNemar's exact-binomial significance (alternative $c > b$, the "
        r"overalignment direction) after Benjamini--Hochberg FDR correction at $q=0.05$ across "
        r"all "
        + f"${n_models} \\times {n_cats} = {n_cells}$ tests. $^{{*}}\\,p<0.05$, "
        r"$^{**}\,p<0.01$, $^{***}\,p<0.001$ (raw $p$, FDR-significant). Cells without stars "
        r"do not survive FDR correction.}")
    lines.append(r"\label{tab:sig_per_category}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    cols_spec = "l" + "r" * len(cats)
    lines.append(r"\begin{tabular}{" + cols_spec + "}")
    lines.append(r"\toprule")
    header = r"\textbf{Model} & " + " & ".join(rf"\textbf{{{lbl}}}" for lbl in cat_labels) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for m in models_sorted:
        row_parts = [m]
        for cat in cats:
            r = cell.get((m, cat))
            if r is None:
                row_parts.append(r"--")
            else:
                stars = stars_for_p(r["p"], r["q"], r["sig_fdr"])
                row_parts.append(f"{r['mar']:.1f}{stars}")
        lines.append(" & ".join(row_parts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Test 1 (overall) ...")
    t1 = run_test1();  print(f"  {len(t1)} models")
    print("Test 2 (trigger) ...")
    t2 = run_test2();  print(f"  {len(t2)} models")
    print("Test 3 (CoT) ...")
    t3 = run_test3();  print(f"  {len(t3)} models")
    print("Test 4 (per-category) ...")
    t4 = run_test4();  print(f"  {len(t4)} cells")

    parts = [
        "% Auto-generated by scripts/compute_significance.py — do not edit by hand.",
        "",
        fmt_table1(t1),
        "",
        fmt_table2(t2),
        "",
        fmt_table3(t3),
        "",
        fmt_table4(t4),
        "",
    ]
    OUT_TEX.write_text("\n".join(parts))
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
