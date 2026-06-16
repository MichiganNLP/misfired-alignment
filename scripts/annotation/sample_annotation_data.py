"""
Sample pairs from prompt_pairs_bbq.json and build the annotation task.

For each pair we create 4 items:
  (1) stereotyped prompt  → expected = "yes"   (real)
  (2) contrast prompt     → expected = "yes"   (real)
  (3) stereo context  + contrast question → expected = "no"  (foil)
  (4) contrast context + stereo question  → expected = "no"  (foil)

This gives a 50/50 yes/no split, preventing annotators from learning
the "always yes" shortcut while keeping both conditions per pair so
that pair-level MAR is still computable from real items (1) and (2).

Output: data/annotation_task.json  (list of N_PAIRS * 4 items)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

PROJ_DIR   = Path(__file__).parent.parent.parent
PAIRS_FILE = PROJ_DIR / "data" / "prompt_pairs_bbq.json"
OUT_FILE   = PROJ_DIR / "data" / "annotation_task.json"

N_PAIRS = 100   # × 4 items each = 400 total, 50 % yes / 50 % no
SEED    = 42


# ── Prompt splitting ──────────────────────────────────────────────────────────

def split_prompt(text: str) -> tuple[str, str]:
    """Return (context, question) by splitting at the last sentence-final '?'.

    context  = everything up to and including the period before the question.
    question = the final question sentence (including '?').
    """
    q_idx = text.rfind("?")
    if q_idx == -1:
        return text, ""
    # Walk back to find start of the question sentence
    prev_period = text.rfind(".", 0, q_idx)
    if prev_period == -1:
        return "", text[:q_idx + 1].strip()
    context  = text[:prev_period + 1].strip()
    question = text[prev_period + 1 : q_idx + 1].strip()
    return context, question


def make_foil(context_prompt: str, question_prompt: str) -> str:
    """Build a foil prompt: context from one prompt, question from another.

    The answer is 'no' because the question asks about the person who did
    NOT perform the action described in the context.
    """
    context, _        = split_prompt(context_prompt)
    _,       question = split_prompt(question_prompt)
    if not context or not question:
        return ""
    return f"{context} {question}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)

    with open(PAIRS_FILE) as f:
        pairs = json.load(f)

    # Stratified sampling by category
    by_cat = defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)

    total = len(pairs)
    quotas = {cat: max(1, round(len(lst) / total * N_PAIRS))
              for cat, lst in by_cat.items()}

    # Adjust to exactly N_PAIRS
    diff = N_PAIRS - sum(quotas.values())
    cats_sorted = sorted(quotas, key=lambda c: len(by_cat[c]), reverse=True)
    for i in range(abs(diff)):
        cat = cats_sorted[i % len(cats_sorted)]
        quotas[cat] += 1 if diff > 0 else -1

    print("Sampling quotas (pairs):")
    for cat in sorted(quotas):
        print(f"  {cat}: {quotas[cat]} / {len(by_cat[cat])}")
    print(f"  Total: {sum(quotas.values())} pairs -> {sum(quotas.values()) * 4} items "
          f"(50% yes / 50% no)")

    sampled = []
    for cat, n in quotas.items():
        pool = by_cat[cat][:]
        random.shuffle(pool)
        sampled.extend(pool[:n])

    # Build items
    items = []
    skipped_foils = 0

    for pair in sampled:
        st = pair["prompts"]["stereotyped"]
        ct = pair["prompts"]["contrast"]

        # (1) Real stereotyped — yes
        items.append({
            "task_id":   f"{pair['id']}__stereotyped",
            "pair_id":   pair["id"],
            "condition": "stereotyped",
            "item_type": "real",
            "category":  pair["category"],
            "prompt":    st["text"],
            "expected":  "yes",
        })

        # (2) Real contrast — yes
        items.append({
            "task_id":   f"{pair['id']}__contrast",
            "pair_id":   pair["id"],
            "condition": "contrast",
            "item_type": "real",
            "category":  pair["category"],
            "prompt":    ct["text"],
            "expected":  "yes",
        })

        # (3) Foil: stereo context + contrast question — no
        foil_a = make_foil(st["text"], ct["text"])
        if foil_a:
            items.append({
                "task_id":   f"{pair['id']}__foil_st_ctx",
                "pair_id":   pair["id"],
                "condition": "foil_stereo_ctx",
                "item_type": "foil",
                "category":  pair["category"],
                "prompt":    foil_a,
                "expected":  "no",
            })
        else:
            skipped_foils += 1

        # (4) Foil: contrast context + stereo question — no
        foil_b = make_foil(ct["text"], st["text"])
        if foil_b:
            items.append({
                "task_id":   f"{pair['id']}__foil_ct_ctx",
                "pair_id":   pair["id"],
                "condition": "foil_contrast_ctx",
                "item_type": "real",
                "category":  pair["category"],
                "prompt":    foil_b,
                "expected":  "no",
            })
        else:
            skipped_foils += 1

    if skipped_foils:
        print(f"  Warning: {skipped_foils} foils skipped (bad prompt split)")

    random.shuffle(items)

    n_yes = sum(1 for i in items if i["expected"] == "yes")
    n_no  = sum(1 for i in items if i["expected"] == "no")
    print(f"\nFinal: {len(items)} items  |  yes={n_yes} ({n_yes/len(items):.0%})  "
          f"no={n_no} ({n_no/len(items):.0%})")

    # Spot-check a foil
    foil_examples = [i for i in items if i["item_type"] == "foil"]
    if foil_examples:
        ex = foil_examples[0]
        print(f"\nFoil example (expected='{ex['expected']}'):")
        print(f"  {ex['prompt']}")

    with open(OUT_FILE, "w") as f:
        json.dump(items, f, indent=2)
    print(f"\nSaved {len(items)} items to {OUT_FILE}")


if __name__ == "__main__":
    main()
