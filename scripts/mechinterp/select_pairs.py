"""
Build a data-driven failure/control pair set for mechinterp from an evaluation
results JSON. Replaces the hardcoded ANALYSIS_EXAMPLES list.

Definitions (per pair, w.r.t. a chosen reference model's eval results):
  - failure pair: stereotyped prompt was answered WRONG and contrast was RIGHT
                  (the categorical alignment-induced failure)
  - control pair: BOTH stereotyped and contrast were answered RIGHT
                  (no suppression event — used as baseline for layer-localization)

The output JSON is a list of dicts compatible with utils.build_prompt_pairs():
  {id, category, role, stereotyped_group, contrast_group,
   stereotyped_user, contrast_user}

Stratification:
  - Stratified sample by category proportional to category MAR mass
    (so categories with more failures contribute more pairs).
  - Each pair's role ("failure" or "control") is preserved in the output.

Usage:
  python scripts/mechinterp/select_pairs.py
  python scripts/mechinterp/select_pairs.py --tag bbq_trigger --n-fail 30 --n-ctrl 30
  python scripts/mechinterp/select_pairs.py --reference meta-llama/Llama-3.1-8B-Instruct --tag bbq
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

PROJ_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJ_DIR / "data"
RESULTS_DIR = PROJ_DIR / "results"


def safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _strip_trigger_tail(text: str) -> str:
    """The pairs file stores prompts with the 'Answer yes or no.' tail.
    Mechinterp uses its own system prompt, so we keep the user message verbatim."""
    return text


def stratified_sample(
    items_by_cat: dict[str, list],
    n: int,
    seed: int,
) -> list:
    """Sample `n` items, stratified by category proportional to category size,
    falling back to as-uniform-as-possible if any category is short."""
    rng = random.Random(seed)
    cats = sorted(items_by_cat)
    if not cats:
        return []
    total = sum(len(items_by_cat[c]) for c in cats)
    if total == 0:
        return []

    # Proportional quotas (rounded), then redistribute leftovers.
    quotas = {c: int(round(n * len(items_by_cat[c]) / total)) for c in cats}
    while sum(quotas.values()) > n:
        # Trim from the largest quota first
        c = max(cats, key=lambda x: quotas[x])
        quotas[c] -= 1
    while sum(quotas.values()) < n:
        # Add to the category with largest remaining slack
        c = max(cats, key=lambda x: len(items_by_cat[x]) - quotas[x])
        quotas[c] += 1

    # Cap each quota at category size
    sampled = []
    for c in cats:
        q = min(quotas[c], len(items_by_cat[c]))
        sampled.extend(rng.sample(items_by_cat[c], q))

    # If we under-sampled due to capping, draw extras from any non-empty leftover
    deficit = n - len(sampled)
    if deficit > 0:
        leftover = []
        used_ids = {s["id"] for s in sampled}
        for c in cats:
            for item in items_by_cat[c]:
                if item["id"] not in used_ids:
                    leftover.append(item)
        deficit = min(deficit, len(leftover))
        sampled.extend(rng.sample(leftover, deficit))

    rng.shuffle(sampled)
    return sampled


def build_pair_record(eval_record: dict, pair_record: dict, role: str) -> dict:
    """Convert (eval result row, prompt-pair row) into the mechinterp dict."""
    return {
        "id": pair_record["id"],
        "category": pair_record["category"],
        "role": role,
        "stereotyped_group": pair_record["stereotyped_group"],
        "contrast_group": pair_record["contrast_group"],
        "stereotyped_user": _strip_trigger_tail(pair_record["prompts"]["stereotyped"]["text"]),
        "contrast_user":    _strip_trigger_tail(pair_record["prompts"]["contrast"]["text"]),
        "expected_stereotyped": pair_record["prompts"]["stereotyped"]["expected"],
        "expected_contrast":    pair_record["prompts"]["contrast"]["expected"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Model whose eval results define failure/control labels")
    ap.add_argument("--tag", default="bbq_trigger", choices=["bbq", "bbq_trigger"],
                    help="Eval condition to source from (with-trigger has more failures)")
    ap.add_argument("--n-fail", type=int, default=30)
    ap.add_argument("--n-ctrl", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DATA_DIR / "mechinterp_pairs.json"))
    args = ap.parse_args()

    # Locate eval results
    eval_path = RESULTS_DIR / f"{safe(args.reference)}_{args.tag}_results.json"
    if not eval_path.exists():
        raise SystemExit(f"Eval results not found: {eval_path}")
    with open(eval_path) as f:
        eval_data = json.load(f)
    eval_results = eval_data["results"]

    pairs_file = DATA_DIR / f"prompt_pairs_{args.tag}.json"
    with open(pairs_file) as f:
        pairs_raw = json.load(f)
    pair_by_id = {p["id"]: p for p in pairs_raw}

    # Bucket eval rows by failure/control × category
    failures_by_cat: dict[str, list] = defaultdict(list)
    controls_by_cat: dict[str, list] = defaultdict(list)
    for r in eval_results:
        s_ok = r["responses"]["stereotyped"]["correct"]
        c_ok = r["responses"]["contrast"]["correct"]
        cat = r["category"]
        pair = pair_by_id.get(r["id"])
        if pair is None:
            continue
        if (not s_ok) and c_ok:
            failures_by_cat[cat].append(build_pair_record(r, pair, "failure"))
        elif s_ok and c_ok:
            controls_by_cat[cat].append(build_pair_record(r, pair, "control"))

    print(f"Reference: {args.reference}  ({args.tag})")
    print(f"Eval source: {eval_path.name}")
    print(f"Per-category counts:")
    print(f"  {'category':<22}  {'fail':>6}  {'ctrl':>6}")
    for c in sorted(set(list(failures_by_cat) + list(controls_by_cat))):
        print(f"  {c:<22}  {len(failures_by_cat[c]):>6}  {len(controls_by_cat[c]):>6}")
    total_fail = sum(len(v) for v in failures_by_cat.values())
    total_ctrl = sum(len(v) for v in controls_by_cat.values())
    print(f"  {'TOTAL':<22}  {total_fail:>6}  {total_ctrl:>6}")

    fail_sample = stratified_sample(failures_by_cat, args.n_fail, args.seed)
    ctrl_sample = stratified_sample(controls_by_cat, args.n_ctrl, args.seed + 1)

    selected = fail_sample + ctrl_sample
    print(f"\nSelected: {len(fail_sample)} failures + {len(ctrl_sample)} controls = {len(selected)}")

    out = {
        "reference_model": args.reference,
        "reference_tag": args.tag,
        "n_fail_requested": args.n_fail,
        "n_ctrl_requested": args.n_ctrl,
        "seed": args.seed,
        "pairs": selected,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
