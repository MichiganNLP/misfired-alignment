"""
Build a second wave of human-annotation batches.

In the same spirit as `sample_annotation_data.py` + `generate_csv_batches.py`,
but:
  - samples 100 FRESH pairs from prompt_pairs_bbq.json, excluding any pair_id
    already used in the original data/annotation_task.json
  - emits 3 batches instead of 5 (batch_6.csv … batch_8.csv)
  - doubles attention-control foils per batch (10 → 20) for stricter screening
  - fixes the item_type='real' label on foil B (kept as 'foil')

Reads:
  data/prompt_pairs_bbq.json
  data/annotation_task.json     (to read previously-used pair_ids)

Writes:
  data/annotation_task_v2.json
  data/annotation_batches_v2/batch_6.csv … batch_8.csv
  data/annotation_batches_v2/answer_key.csv      (researcher only)
  data/annotation_batches_v2/guidelines.txt
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

PROJ_DIR     = Path(__file__).parent.parent.parent
PAIRS_FILE   = PROJ_DIR / "data" / "prompt_pairs_bbq.json"
PRIOR_TASK   = PROJ_DIR / "data" / "annotation_task.json"
TASK_OUT     = PROJ_DIR / "data" / "annotation_task_v2.json"
BATCH_OUT    = PROJ_DIR / "data" / "annotation_batches_v2"
PRIOR_GUIDE  = PROJ_DIR / "data" / "annotation_batches" / "guidelines.txt"

N_PAIRS    = 100
N_BATCHES  = 3
N_CORE     = 10        # core yes-items shared across all 3 batches
N_ATTN     = 20        # attention-control foil items shared across all 3 batches
SEED       = 43
BATCH_OFFSET = 5       # so this wave's batch numbers continue from 6


# ── helpers ──────────────────────────────────────────────────────────────────

def split_prompt(text: str) -> tuple[str, str]:
    """Return (passage, question) split at the last sentence-final '?'."""
    q_idx = text.rfind("?")
    if q_idx == -1:
        return text.strip(), ""
    prev_period = text.rfind(".", 0, q_idx)
    if prev_period == -1:
        return "", text[: q_idx + 1].strip()
    return text[: prev_period + 1].strip(), text[prev_period + 1 : q_idx + 1].strip()


def make_foil(context_prompt: str, question_prompt: str) -> str:
    """Build a foil: context from one prompt, question from another."""
    context, _ = split_prompt(context_prompt)
    _, question = split_prompt(question_prompt)
    if not context or not question:
        return ""
    return f"{context} {question}"


# ── Step 1: sample 100 fresh pairs ──────────────────────────────────────────

def sample_fresh_pairs() -> list[dict]:
    rng = random.Random(SEED)

    # Pair_ids already used in the prior wave
    with open(PRIOR_TASK) as f:
        prior_items = json.load(f)
    used_ids = {it["pair_id"] for it in prior_items}
    print(f"Excluding {len(used_ids)} pair_ids from prior wave")

    with open(PAIRS_FILE) as f:
        all_pairs = json.load(f)
    available = [p for p in all_pairs if p["id"] not in used_ids]
    print(f"Available pool: {len(available)} / {len(all_pairs)} pairs")

    # Stratified sampling proportional to category size of available pool
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for p in available:
        by_cat[p["category"]].append(p)
    total = len(available)
    quotas = {cat: max(1, round(len(lst) / total * N_PAIRS))
              for cat, lst in by_cat.items()}
    diff = N_PAIRS - sum(quotas.values())
    cats_sorted = sorted(quotas, key=lambda c: len(by_cat[c]), reverse=True)
    for i in range(abs(diff)):
        cat = cats_sorted[i % len(cats_sorted)]
        quotas[cat] += 1 if diff > 0 else -1

    print("Quotas (pairs):")
    for cat in sorted(quotas):
        print(f"  {cat:<22}  {quotas[cat]:>3} / {len(by_cat[cat])}")
    assert sum(quotas.values()) == N_PAIRS

    sampled: list[dict] = []
    for cat, n in quotas.items():
        pool = by_cat[cat][:]
        rng.shuffle(pool)
        sampled.extend(pool[:n])
    return sampled


# ── Step 2: build the 4-items-per-pair task ─────────────────────────────────

def build_items(sampled_pairs: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    items: list[dict] = []
    skipped_foils = 0

    for pair in sampled_pairs:
        st = pair["prompts"]["stereotyped"]
        ct = pair["prompts"]["contrast"]

        items.append({
            "task_id":   f"{pair['id']}__stereotyped",
            "pair_id":   pair["id"],
            "condition": "stereotyped",
            "item_type": "real",
            "category":  pair["category"],
            "prompt":    st["text"],
            "expected":  "yes",
        })
        items.append({
            "task_id":   f"{pair['id']}__contrast",
            "pair_id":   pair["id"],
            "condition": "contrast",
            "item_type": "real",
            "category":  pair["category"],
            "prompt":    ct["text"],
            "expected":  "yes",
        })

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

        foil_b = make_foil(ct["text"], st["text"])
        if foil_b:
            items.append({
                "task_id":   f"{pair['id']}__foil_ct_ctx",
                "pair_id":   pair["id"],
                "condition": "foil_contrast_ctx",
                # Fix: original sample_annotation_data.py mislabelled foil B as "real"
                "item_type": "foil",
                "category":  pair["category"],
                "prompt":    foil_b,
                "expected":  "no",
            })
        else:
            skipped_foils += 1

    if skipped_foils:
        print(f"  Warning: {skipped_foils} foils skipped (bad prompt split)")

    rng.shuffle(items)
    n_yes = sum(1 for i in items if i["expected"] == "yes")
    n_no  = sum(1 for i in items if i["expected"] == "no")
    print(f"\nTask items: {len(items)}  |  yes={n_yes} ({n_yes/len(items):.0%})  "
          f"no={n_no} ({n_no/len(items):.0%})")
    return items


# ── Step 3: split into 3 batches ─────────────────────────────────────────────

def write_batches(items: list[dict]):
    rng = random.Random(SEED)

    main_pool = [it for it in items if it["expected"] == "yes"]
    foil_pool = [it for it in items if it["expected"] == "no"]
    rng.shuffle(main_pool)
    rng.shuffle(foil_pool)

    if len(foil_pool) < N_ATTN:
        raise SystemExit(f"Not enough foils ({len(foil_pool)} < {N_ATTN})")

    core_items  = main_pool[:N_CORE]
    core_ids    = {it["task_id"] for it in core_items}
    remaining   = main_pool[N_CORE:]
    n_unique    = len(remaining) // N_BATCHES
    unique_groups: list[list[dict]] = []
    for i in range(N_BATCHES):
        start = i * n_unique
        end   = start + n_unique if i < N_BATCHES - 1 else len(remaining)
        unique_groups.append(remaining[start:end])

    attn_items  = foil_pool[:N_ATTN]
    attn_ids    = {it["task_id"] for it in attn_items}

    BATCH_OUT.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict] = []

    for b in range(N_BATCHES):
        batch_num   = b + 1 + BATCH_OFFSET     # batches 6, 7, 8
        main_items  = core_items + unique_groups[b]
        all_batch   = main_items + attn_items
        batch_rng   = random.Random(SEED + batch_num * 97)
        batch_rng.shuffle(all_batch)

        csv_path = BATCH_OUT / f"batch_{batch_num}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["#", "passage", "question", "answer"])
            for pos, item in enumerate(all_batch, 1):
                passage, question = split_prompt(item["prompt"])
                writer.writerow([pos, passage, question, ""])

                if item["task_id"] in attn_ids:
                    itype = "attention_check"
                elif item["task_id"] in core_ids:
                    itype = "core"
                else:
                    itype = "unique"

                key_rows.append({
                    "batch":     batch_num,
                    "#":         pos,
                    "task_id":   item["task_id"],
                    "pair_id":   item["pair_id"],
                    "condition": item["condition"],
                    "category":  item["category"],
                    "item_type": itype,
                    "expected":  item["expected"],
                })

        n_main = len(main_items)
        print(f"  batch_{batch_num}.csv — {n_main} main "
              f"({N_CORE} core + {len(unique_groups[b])} unique) "
              f"+ {N_ATTN} attention check = {len(all_batch)} items")

    key_path = BATCH_OUT / "answer_key.csv"
    fieldnames = ["batch", "#", "task_id", "pair_id", "condition",
                  "category", "item_type", "expected"]
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"\nAnswer key (researcher only) → {key_path}")

    # Reuse the original guidelines verbatim
    guide_dst = BATCH_OUT / "guidelines.txt"
    if PRIOR_GUIDE.exists():
        shutil.copyfile(PRIOR_GUIDE, guide_dst)
        print(f"Guidelines (copied from wave 1) → {guide_dst}")
    else:
        print(f"  WARNING: prior guidelines missing at {PRIOR_GUIDE}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    sampled = sample_fresh_pairs()
    items   = build_items(sampled)

    with open(TASK_OUT, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Wrote {TASK_OUT}")

    write_batches(items)


if __name__ == "__main__":
    main()
