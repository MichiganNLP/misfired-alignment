"""
Analyze human annotation results from data/annotation_results/*.json

Computes:
  - Per-annotator accuracy (stereotyped vs contrast) and MAR
  - Aggregated human accuracy and MAR (micro-average across annotators)
  - Inter-annotator agreement (Cohen's kappa for each pair of annotators)
  - Per-category MAR

Usage:
  python scripts/annotation/analyze_annotations.py
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

PROJ_DIR     = Path(__file__).parent.parent.parent
RESULTS_DIR  = PROJ_DIR / "data" / "annotation_results"


def load_results():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"No annotation files found in {RESULTS_DIR}")
        return []
    annotators = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        annotators.append(data)
        print(f"  Loaded: {data['annotator']}  ({data['n_items']} items, {f.name})")
    return annotators


def per_annotator_stats(data: dict) -> dict:
    results = data["results"]
    st = [r for r in results if r["condition"] == "stereotyped"]
    ct = [r for r in results if r["condition"] == "contrast"]

    st_acc = sum(r["correct"] for r in st) / len(st) if st else 0
    ct_acc = sum(r["correct"] for r in ct) / len(ct) if ct else 0

    # MAR: pairs where stereo=wrong AND contrast=right.
    # Pair items share the same pair_id.
    pair_answers = defaultdict(dict)
    for r in results:
        pair_answers[r["pair_id"]][r["condition"]] = r["correct"]

    kf = sum(
        1 for p in pair_answers.values()
        if "stereotyped" in p and "contrast" in p
        and not p["stereotyped"] and p["contrast"]
    )
    n_pairs = len(pair_answers)

    return {
        "annotator":    data["annotator"],
        "n_items":      data["n_items"],
        "n_pairs":      n_pairs,
        "stereo_acc":   st_acc * 100,
        "contrast_acc": ct_acc * 100,
        "delta":        (ct_acc - st_acc) * 100,
        "mar":          kf / n_pairs * 100 if n_pairs else 0,
    }


def cohen_kappa(a1_answers: list, a2_answers: list) -> float:
    assert len(a1_answers) == len(a2_answers)
    n = len(a1_answers)
    agree = sum(x == y for x, y in zip(a1_answers, a2_answers))
    po = agree / n

    labels = set(a1_answers) | set(a2_answers)
    pe = sum(
        (a1_answers.count(l) / n) * (a2_answers.count(l) / n)
        for l in labels
    )
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def inter_annotator_agreement(all_data: list) -> None:
    if len(all_data) < 2:
        print("  (Need at least 2 annotators for IAA)")
        return

    # Align by task_id
    by_task = {}
    for data in all_data:
        name = data["annotator"]
        for r in data["results"]:
            tid = r["task_id"]
            by_task.setdefault(tid, {})[name] = r["answer"]

    annotator_names = [d["annotator"] for d in all_data]
    for i in range(len(annotator_names)):
        for j in range(i + 1, len(annotator_names)):
            a1, a2 = annotator_names[i], annotator_names[j]
            shared = [
                (by_task[t][a1], by_task[t][a2])
                for t in by_task
                if a1 in by_task[t] and a2 in by_task[t]
            ]
            if not shared:
                continue
            l1, l2 = zip(*shared)
            kappa = cohen_kappa(list(l1), list(l2))
            agree_pct = sum(x == y for x, y in shared) / len(shared) * 100
            print(f"  {a1} vs {a2}: kappa={kappa:.3f}, agree={agree_pct:.1f}% (n={len(shared)})")


def per_category_stats(all_data: list) -> None:
    agg = defaultdict(lambda: {"n": 0, "stereo": 0, "contrast": 0, "kf": 0, "pairs": defaultdict(dict)})

    for data in all_data:
        for r in data["results"]:
            cat = r["category"]
            agg[cat]["n"] += 1
            if r["condition"] == "stereotyped":
                agg[cat]["stereo"] += r["correct"]
            else:
                agg[cat]["contrast"] += r["correct"]
            agg[cat]["pairs"][r["pair_id"]][r["condition"]] = r["correct"]

    for cat in sorted(agg, key=lambda c: -sum(
        1 for p in agg[c]["pairs"].values()
        if "stereotyped" in p and "contrast" in p and not p["stereotyped"] and p["contrast"]
    ) / max(len(agg[c]["pairs"]), 1)):
        v    = agg[cat]
        n_st = sum(1 for r_cond in v["pairs"].values() if "stereotyped" in r_cond)
        n_ct = sum(1 for r_cond in v["pairs"].values() if "contrast"    in r_cond)
        kf   = sum(
            1 for p in v["pairs"].values()
            if "stereotyped" in p and "contrast" in p and not p["stereotyped"] and p["contrast"]
        )
        n_pairs = len(v["pairs"])
        st_acc  = v["stereo"]   / n_st  * 100 if n_st  else 0
        ct_acc  = v["contrast"] / n_ct  * 100 if n_ct  else 0
        mar     = kf / n_pairs  * 100   if n_pairs else 0
        print(f"  {cat:25s}  stereo_acc={st_acc:5.1f}  contrast_acc={ct_acc:5.1f}  MAR={mar:5.1f}  (n_pairs={n_pairs})")


def main():
    print("Loading annotation files...")
    all_data = load_results()
    if not all_data:
        return

    print()
    print("=== Per-Annotator Results ===")
    agg_st, agg_ct, agg_mar = [], [], []
    for data in all_data:
        s = per_annotator_stats(data)
        agg_st.append(s["stereo_acc"])
        agg_ct.append(s["contrast_acc"])
        agg_mar.append(s["mar"])
        print(f"  {s['annotator']:20s}  stereo={s['stereo_acc']:5.1f}%  "
              f"contrast={s['contrast_acc']:5.1f}%  Δ={s['delta']:+5.1f}%  MAR={s['mar']:5.1f}%")

    if len(all_data) > 1:
        print()
        print(f"  {'AVERAGE':20s}  stereo={sum(agg_st)/len(agg_st):5.1f}%  "
              f"contrast={sum(agg_ct)/len(agg_ct):5.1f}%  "
              f"Δ={sum(agg_ct)/len(agg_ct)-sum(agg_st)/len(agg_st):+5.1f}%  "
              f"MAR={sum(agg_mar)/len(agg_mar):5.1f}%")

    print()
    print("=== Inter-Annotator Agreement ===")
    inter_annotator_agreement(all_data)

    print()
    print("=== Per-Category MAR (pooled across annotators) ===")
    per_category_stats(all_data)


if __name__ == "__main__":
    main()
