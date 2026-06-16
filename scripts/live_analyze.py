"""
Live analysis of fairness-logic evaluation results.

Reads both completed *_results.json files and in-progress *_results.jsonl files.
Designed to be called repeatedly by a watcher as new results land.

Usage:
    python scripts/live_analyze.py                    # all files in results/
    python scripts/live_analyze.py --no-failures      # skip per-pair failure dump
    python scripts/live_analyze.py --plot             # save figures too
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_records(path: Path) -> list:
    """Load records from either a .json or .jsonl result file."""
    if path.suffix == ".jsonl":
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    else:
        with open(path) as f:
            data = json.load(f)
        return data.get("results", [])


def load_meta(path: Path) -> dict:
    """Return metadata (model, provider, cot, tag) inferred from filename."""
    stem = path.stem.replace("_results", "")
    # Reverse the safe_name transform: last segment after known tags
    known_tags = ["bbq_trigger_cot", "bbq_trigger", "bbq_cot", "bbq"]
    tag = ""
    model_part = stem
    for t in known_tags:
        if stem.endswith(f"_{t}"):
            tag = t
            model_part = stem[: -(len(t) + 1)]
            break
    model_name = model_part.replace("_", "/", model_part.count("_") - model_part.replace("/","").count("_"))
    # Try reading actual metadata from .json if available
    json_path = path.with_suffix(".json") if path.suffix == ".jsonl" else path
    meta = {"model": model_part, "tag": tag, "cot": "cot" in tag, "trigger": "trigger" in tag}
    if json_path.exists():
        with open(json_path) as f:
            try:
                d = json.load(f)
                meta["model"] = d.get("model", model_part)
                meta["provider"] = d.get("provider", "?")
                meta["native_thinking"] = d.get("native_thinking", False)
            except Exception:
                pass
    return meta


def compute_stats(records: list) -> dict:
    n = len(records)
    if n == 0:
        return {}
    s_ok = sum(r["responses"]["stereotyped"]["correct"] for r in records)
    c_ok = sum(r["responses"]["contrast"]["correct"] for r in records
               if "contrast" in r["responses"])
    c_n  = sum(1 for r in records if "contrast" in r["responses"])
    kf   = sum(1 for r in records
               if not r["responses"]["stereotyped"]["correct"]
               and "contrast" in r["responses"]
               and r["responses"]["contrast"]["correct"])
    unk_s = sum(1 for r in records if r["responses"]["stereotyped"]["parsed_answer"] == "unknown")
    unk_c = sum(1 for r in records if "contrast" in r["responses"]
                and r["responses"]["contrast"]["parsed_answer"] == "unknown")

    cats = defaultdict(lambda: {"n": 0, "s": 0, "c": 0, "c_n": 0, "kf": 0})
    for r in records:
        cat = r.get("category", "unknown")
        cats[cat]["n"] += 1
        cats[cat]["s"] += int(r["responses"]["stereotyped"]["correct"])
        if "contrast" in r["responses"]:
            cats[cat]["c"] += int(r["responses"]["contrast"]["correct"])
            cats[cat]["c_n"] += 1
            cats[cat]["kf"] += int(
                not r["responses"]["stereotyped"]["correct"]
                and r["responses"]["contrast"]["correct"]
            )

    return {
        "n": n,
        "stereo_acc": s_ok / n,
        "contrast_acc": c_ok / c_n if c_n else None,
        "discrepancy": (c_ok / c_n - s_ok / n) if c_n else None,
        "kf_rate": kf / n,
        "unknown_s": unk_s,
        "unknown_c": unk_c,
        "cats": dict(cats),
    }


def _pct(v):
    return f"{v:.1%}" if v is not None else "  n/a "


# ── Main report ────────────────────────────────────────────────────────────────

def print_summary_table(runs: dict):
    """One row per (model × tag), sorted by model then tag."""
    cols = ["model", "tag", "n", "stereo", "contrast", "Δ", "key_fail", "unk"]
    fmt  = f"  {{:<42}} {{:<20}} {{:>6}} {{:>8}} {{:>9}} {{:>7}} {{:>9}} {{:>5}}"
    print(fmt.format(*cols))
    print("  " + "-" * 112)

    for key in sorted(runs):
        model, tag = key
        st = runs[key]["stats"]
        if not st:
            continue
        label = model.split("/")[-1][:42]
        contrast_str = _pct(st["contrast_acc"]) if st["contrast_acc"] is not None else "  —   "
        disc_str     = (f"{st['discrepancy']:+.1%}" if st["discrepancy"] is not None else "   —  ")
        unk = f"{st['unknown_s']}/{st['unknown_c']}"
        print(fmt.format(label, tag, st["n"], _pct(st["stereo_acc"]),
                         contrast_str, disc_str, _pct(st["kf_rate"]), unk))


def print_trigger_analysis(runs: dict):
    """Compare bbq stereo accuracy vs bbq_trigger stereo accuracy per model."""
    models = set(m for m, t in runs if t in ("bbq", "bbq_trigger"))
    rows = []
    for m in sorted(models):
        base  = runs.get((m, "bbq"),         {}).get("stats", {})
        trig  = runs.get((m, "bbq_trigger"), {}).get("stats", {})
        if not base or not trig:
            continue
        delta = trig["stereo_acc"] - base["stereo_acc"]
        rows.append((m, base["stereo_acc"], trig["stereo_acc"], delta,
                     base["n"], trig["n"]))

    if not rows:
        print("  (no paired bbq / bbq_trigger runs yet)")
        return

    print(f"  {'Model':<40} {'bbq stereo':>12} {'trigger stereo':>15} {'Δ (trigger-base)':>18}  pairs")
    print("  " + "-" * 96)
    for m, b, t, d, nb, nt in sorted(rows, key=lambda x: x[3]):
        label = m.split("/")[-1][:40]
        print(f"  {label:<40} {b:>12.1%} {t:>15.1%} {d:>+18.1%}  {nb}/{nt}")


def print_cot_analysis(runs: dict):
    """Compare direct vs CoT Misfired Alignment Rates per model."""
    models = set(m for m, t in runs if t in ("bbq", "bbq_cot"))
    rows = []
    for m in sorted(models):
        direct = runs.get((m, "bbq"),     {}).get("stats", {})
        cot    = runs.get((m, "bbq_cot"), {}).get("stats", {})
        if not direct or not cot:
            continue
        rows.append((m, direct["kf_rate"], cot["kf_rate"],
                     cot["kf_rate"] - direct["kf_rate"],
                     direct["n"], cot["n"]))

    if not rows:
        print("  (no paired bbq / bbq_cot runs yet)")
        return

    print(f"  {'Model':<40} {'direct KF':>10} {'CoT KF':>8} {'Δ CoT-direct':>14}  pairs")
    print("  " + "-" * 80)
    for m, d, c, delta, nd, nc in sorted(rows, key=lambda x: x[3]):
        label = m.split("/")[-1][:40]
        print(f"  {label:<40} {d:>10.1%} {c:>8.1%} {delta:>+14.1%}  {nd}/{nc}")


def print_category_heatmap(runs: dict):
    """Aggregate Misfired Alignment Rate by category across all bbq (direct) runs."""
    cat_agg = defaultdict(lambda: {"kf": 0, "n": 0, "s": 0})
    for (model, tag), info in runs.items():
        if tag != "bbq":
            continue
        for cat, cv in info["stats"].get("cats", {}).items():
            cat_agg[cat]["kf"] += cv["kf"]
            cat_agg[cat]["n"]  += cv["n"]
            cat_agg[cat]["s"]  += cv["s"]

    if not cat_agg:
        return

    print(f"  {'Category':<35} {'KF rate':>8}  {'Stereo acc':>11}  pairs")
    print("  " + "-" * 65)
    for cat, v in sorted(cat_agg.items(), key=lambda x: -x[1]["kf"] / max(x[1]["n"], 1)):
        n = v["n"]
        print(f"  {cat:<35} {v['kf']/n:>8.1%}  {v['s']/n:>11.1%}  {n}")


def print_failure_examples(runs: dict, max_per_run: int = 3):
    for (model, tag), info in sorted(runs.items()):
        if tag not in ("bbq", "bbq_cot"):
            continue
        failures = [
            r for r in info["records"]
            if "contrast" in r["responses"]
            and not r["responses"]["stereotyped"]["correct"]
            and r["responses"]["contrast"]["correct"]
        ]
        if not failures:
            continue
        label = f"{model.split('/')[-1]} [{tag}]"
        print(f"\n  --- {label} ({len(failures)} total failures, showing {min(max_per_run, len(failures))}) ---")
        for r in failures[:max_per_run]:
            s = r["responses"]["stereotyped"]
            c = r["responses"]["contrast"]
            print(f"  [{r['category']}] {r['id']}")
            print(f"    Stereo  ({s['group']}): got '{s['parsed_answer']}' want '{s['expected']}'")
            print(f"    Contrast({c['group']}): got '{c['parsed_answer']}' want '{c['expected']}'")
            print(f"    Prompt: {s['prompt'][:110]}...")


def plot_results(runs: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not available — skipping")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Summary bar: stereo vs contrast vs kf by model (bbq direct only) ---
    bbq_runs = {m: v["stats"] for (m, t), v in runs.items()
                if t == "bbq" and v["stats"] and v["stats"]["n"] >= 50}
    if bbq_runs:
        models = sorted(bbq_runs)
        labels = [m.split("/")[-1] for m in models]
        s_acc  = [bbq_runs[m]["stereo_acc"]   for m in models]
        c_acc  = [bbq_runs[m]["contrast_acc"] or 0 for m in models]
        kf     = [bbq_runs[m]["kf_rate"]       for m in models]

        x = np.arange(len(models))
        w = 0.28
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
        ax.bar(x - w, s_acc, w, label="Stereo acc",   color="#e07070")
        ax.bar(x,     c_acc, w, label="Contrast acc",  color="#6090d0")
        ax.bar(x + w, kf,   w, label="Key fail rate", color="#e09030")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rate")
        ax.set_title("BBQ (direct): stereo acc / contrast acc / Misfired Alignment Rate")
        ax.legend()
        ax.axhline(1.0, color="gray", lw=0.5, ls="--")
        plt.tight_layout()
        out = FIGURES_DIR / "summary_bbq_direct.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] saved {out}")

    # --- Trigger delta bar ---
    trig_rows = []
    for m in set(m for m, t in runs if t == "bbq"):
        base = runs.get((m, "bbq"),         {}).get("stats", {})
        trig = runs.get((m, "bbq_trigger"), {}).get("stats", {})
        if base and trig and base["n"] >= 50 and trig["n"] >= 50:
            trig_rows.append((m.split("/")[-1], trig["stereo_acc"] - base["stereo_acc"]))
    if trig_rows:
        trig_rows.sort(key=lambda x: x[1])
        labels, deltas = zip(*trig_rows)
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.0), 4))
        colors = ["#c0392b" if d < 0 else "#27ae60" for d in deltas]
        ax.barh(labels, deltas, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Stereo accuracy: trigger − base  (negative = trigger hurts)")
        ax.set_title("Effect of 'It's not okay to assume…' trigger on stereotyped accuracy")
        plt.tight_layout()
        out = FIGURES_DIR / "trigger_delta.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] saved {out}")

    # --- Category heatmap (bbq direct, all models combined) ---
    cat_agg = defaultdict(lambda: {"kf": 0, "n": 0})
    for (m, t), v in runs.items():
        if t != "bbq" or not v["stats"]:
            continue
        for cat, cv in v["stats"]["cats"].items():
            cat_agg[cat]["kf"] += cv["kf"]
            cat_agg[cat]["n"]  += cv["n"]
    if cat_agg:
        cats   = sorted(cat_agg, key=lambda c: -cat_agg[c]["kf"] / max(cat_agg[c]["n"], 1))
        values = [cat_agg[c]["kf"] / cat_agg[c]["n"] for c in cats]
        fig, ax = plt.subplots(figsize=(7, max(4, len(cats) * 0.5)))
        ax.barh(cats, values, color="#c0392b")
        ax.set_xlim(0, max(values) * 1.25 + 0.01)
        ax.set_xlabel("Misfired Alignment Rate (stereotyped wrong, contrast right)")
        ax.set_title("Category-level Misfired Alignment Rate (all models, bbq direct)")
        for i, v in enumerate(values):
            ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=9)
        plt.tight_layout()
        out = FIGURES_DIR / "category_kf_heatmap.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] saved {out}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--no-failures", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--min-pairs", type=int, default=10,
                        help="Minimum completed pairs to include a run in the report")
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    # Collect all result files — prefer .json (complete) over .jsonl (partial)
    seen: dict[str, Path] = {}
    for p in sorted(rdir.glob("*_results.json")):
        seen[p.stem] = p
    for p in sorted(rdir.glob("*_results.jsonl")):
        key = p.stem.replace(".jsonl", "")
        if key not in seen:   # only use JSONL if no completed JSON yet
            seen[key] = p

    if not seen:
        print("No result files found yet.")
        sys.exit(0)

    # Load runs
    runs: dict = {}
    for stem, path in seen.items():
        records = load_records(path)
        if len(records) < args.min_pairs:
            continue
        meta = load_meta(path)
        stats = compute_stats(records)
        key = (meta["model"], meta["tag"])
        runs[key] = {"meta": meta, "records": records, "stats": stats,
                     "path": path, "complete": path.suffix == ".json"}

    if not runs:
        print(f"No runs with ≥{args.min_pairs} pairs yet.")
        sys.exit(0)

    total_complete = sum(1 for v in runs.values() if v["complete"])
    total_partial  = sum(1 for v in runs.values() if not v["complete"])

    from datetime import datetime
    print(f"\n{'#'*70}")
    print(f"# fairness-logic — live analysis  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# {total_complete} complete runs, {total_partial} in-progress")
    print(f"{'#'*70}")

    print("\n── OVERVIEW (all runs) ──")
    print_summary_table(runs)

    print("\n── TRIGGER EFFECT (bbq vs bbq_trigger, stereo accuracy) ──")
    print_trigger_analysis(runs)

    print("\n── COT EFFECT (bbq direct vs bbq_cot, Misfired Alignment Rate) ──")
    print_cot_analysis(runs)

    print("\n── CATEGORY BREAKDOWN (bbq direct, all models aggregated) ──")
    print_category_heatmap(runs)

    if not args.no_failures:
        print("\n── SAMPLE KEY FAILURE CASES ──")
        print_failure_examples(runs)

    if args.plot:
        print("\n── PLOTS ──")
        plot_results(runs)

    print(f"\n[done — {sum(v['stats']['n'] for v in runs.values())} total pairs across {len(runs)} runs]")


if __name__ == "__main__":
    main()
