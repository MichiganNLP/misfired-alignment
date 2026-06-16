"""
Pick a held-out, stratified, balanced set of 10 ICL demos from
data/prompt_pairs_bbq.json. The demos are used as few-shot context for
the ICL ablation; the 10 demo pair_ids are excluded from the evaluation
set so there's no test-time contamination.

Output: data/icl_demos.json. Schema:
  {
    "seed": 42,
    "n_demos": 10,
    "categories": ["Disability_status", ...],
    "demos": [
      {
        "pair_id":   "...",
        "category":  "...",
        "condition": "stereotyped" | "contrast",
        "text":      "<full prompt body up to and including the question>",
        "answer":    "yes"
      }, ...
    ]
  }

Design (per the experiment spec):
  - 1 demo from each of the 8 BBQ categories (8 demos)
  - 2 extra demos from the two largest categories
    (Gender_identity, Age — chosen by population in prompt_pairs_bbq.json)
  - For each demo, deterministically pick stereo or contrast so the
    overall balance is 5/5
  - All demo answers are "yes" (correct under both stereo and contrast
    versions of bbq pairs)
  - Final demo order is shuffled (seed=42) so categories aren't grouped
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

PROJ      = Path(__file__).parent.parent
DATA      = PROJ / "data"
PAIRS_JSON = DATA / "prompt_pairs_bbq.json"
OUT_JSON   = DATA / "icl_demos.json"

SEED      = 42
N_PER_CAT = 1            # base: 1 demo per BBQ category
EXTRAS    = 2            # add 2 extra demos from the largest categories


def main():
    rng = random.Random(SEED)

    pairs = json.load(open(PAIRS_JSON))
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)

    # Sort categories by population descending
    cats_by_size = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    print("Pair counts per category:")
    for c in cats_by_size:
        print(f"  {c:<22}  {len(by_cat[c])}")

    # Allocate quotas: 1 each, plus EXTRAS to the two largest
    quotas = {c: N_PER_CAT for c in by_cat}
    for c in cats_by_size[:EXTRAS]:
        quotas[c] += 1
    total = sum(quotas.values())
    print(f"\nDemo quotas (total {total}):")
    for c in cats_by_size:
        print(f"  {c:<22}  {quotas[c]}")

    # Pick demos per category, alternating stereo/contrast to hit the
    # global 5/5 balance across all 10 demos.
    chosen: list[dict] = []
    cond_alternator = ["stereotyped", "contrast"]
    cond_idx = 0

    # Process categories in sorted order for determinism
    for cat in sorted(by_cat):
        n = quotas[cat]
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        picked = pool[:n]
        for pair in picked:
            cond = cond_alternator[cond_idx % 2]
            cond_idx += 1
            chosen.append({
                "pair_id":   pair["id"],
                "category":  pair["category"],
                "condition": cond,
                "text":      pair["prompts"][cond]["text"],
                "answer":    "yes",
            })

    # Verify balance
    n_stereo = sum(1 for d in chosen if d["condition"] == "stereotyped")
    n_contrast = sum(1 for d in chosen if d["condition"] == "contrast")
    print(f"\nFinal balance: stereotyped={n_stereo}, contrast={n_contrast}")
    assert n_stereo + n_contrast == total

    # Shuffle the demo order so categories aren't blocked
    rng.shuffle(chosen)

    # Verify pair_ids are unique
    ids = [d["pair_id"] for d in chosen]
    assert len(set(ids)) == len(ids), "Duplicate pair_id in demos"

    out = {
        "seed":       SEED,
        "n_demos":    total,
        "categories": list(by_cat.keys()),
        "demos":      chosen,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print("Selected demo pair_ids (held out from eval):")
    for d in chosen:
        print(f"  {d['pair_id']:<25}  {d['category']:<22}  {d['condition']}")


if __name__ == "__main__":
    main()
