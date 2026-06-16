"""
Generate 5 annotator CSV batches from the annotation task.

Each batch:
  - N_CORE items shared across ALL 5 batches  (for inter-annotator agreement)
  - ~N_UNIQUE items unique to one batch        (maximises coverage of yes-items)
  - N_ATTN attention-control foil items        (same 10 in every batch, expected='no')

  Total per batch: N_CORE + N_UNIQUE + N_ATTN  (≈ 60)

Reads:  data/annotation_task.json
Writes: data/annotation_batches/batch_1.csv … batch_5.csv
        data/annotation_batches/answer_key.csv   (researcher only — do NOT share)
        data/annotation_batches/guidelines.txt
"""

import csv
import json
import random
from pathlib import Path

PROJ_DIR  = Path(__file__).parent.parent.parent
TASK_FILE = PROJ_DIR / "data" / "annotation_task.json"
OUT_DIR   = PROJ_DIR / "data" / "annotation_batches"

N_BATCHES = 5
N_CORE    = 10   # items shared across all batches
N_ATTN    = 10   # attention-control foil items (same in every batch)
SEED      = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def split_prompt(text: str) -> tuple[str, str]:
    """Return (passage, question) by splitting at the last '?' boundary."""
    q_idx = text.rfind("?")
    if q_idx == -1:
        return text.strip(), ""
    prev_period = text.rfind(".", 0, q_idx)
    if prev_period == -1:
        return "", text[:q_idx + 1].strip()
    return text[:prev_period + 1].strip(), text[prev_period + 1 : q_idx + 1].strip()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)

    with open(TASK_FILE) as f:
        all_items = json.load(f)

    main_pool = [it for it in all_items if it["expected"] == "yes"]
    foil_pool = [it for it in all_items if it["expected"] == "no"]

    print(f"Loaded {len(all_items)} items  |  yes={len(main_pool)}  no={len(foil_pool)}")

    rng.shuffle(main_pool)
    rng.shuffle(foil_pool)

    # Fixed core items (appear in every batch)
    core_items  = main_pool[:N_CORE]
    core_ids    = {it["task_id"] for it in core_items}

    # Partition remaining yes-items across batches
    remaining   = main_pool[N_CORE:]
    n_unique    = len(remaining) // N_BATCHES          # items unique to each batch
    unique_groups: list[list] = []
    for i in range(N_BATCHES):
        start = i * n_unique
        end   = start + n_unique if i < N_BATCHES - 1 else len(remaining)
        unique_groups.append(remaining[start:end])

    # Fixed attention-control foil items (appear in every batch)
    if len(foil_pool) < N_ATTN:
        raise ValueError(f"Not enough foil items ({len(foil_pool)} < {N_ATTN})")
    attn_items  = foil_pool[:N_ATTN]
    attn_ids    = {it["task_id"] for it in attn_items}

    OUT_DIR.mkdir(exist_ok=True)
    key_rows: list[dict] = []

    for b in range(N_BATCHES):
        batch_num   = b + 1
        main_items  = core_items + unique_groups[b]
        all_batch   = main_items + attn_items

        # Randomise order shown to annotator
        batch_rng = random.Random(SEED + batch_num * 97)
        batch_rng.shuffle(all_batch)

        csv_path = OUT_DIR / f"batch_{batch_num}.csv"
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
        print(f"  batch_{batch_num}.csv — {n_main} main ({N_CORE} core + {len(unique_groups[b])} unique) "
              f"+ {N_ATTN} attention control = {len(all_batch)} items")

    # Answer key (do NOT share with annotators)
    key_path = OUT_DIR / "answer_key.csv"
    fieldnames = ["batch", "#", "task_id", "pair_id", "condition", "category", "item_type", "expected"]
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"\nAnswer key (researcher only) → {key_path}")

    # Guidelines
    guidelines_path = OUT_DIR / "guidelines.txt"
    write_guidelines(guidelines_path)
    print(f"Guidelines                   → {guidelines_path}")


def write_guidelines(path: Path) -> None:
    text = """\
ANNOTATION GUIDELINES
Reading Comprehension Task
==========================

OVERVIEW
--------
You will read short passages, each followed by a yes/no question asking whether
something is true of a specific person mentioned in the passage.
Your job is to answer BASED ONLY on the information given in the passage.
Do NOT use outside knowledge, news, or assumptions of any kind.

SPREADSHEET FORMAT
------------------
Each row contains three fields:
  #         — item number (for your reference)
  passage   — the text to read
  question  — what you are being asked
  answer    — fill in: Yes  or  No  (case-insensitive)

Please answer every row. Leave no cell blank.

HOW TO DECIDE
-------------
Answer YES when:
  The passage contains direct evidence that the statement in the question
  is true of the specific person being asked about.

Answer NO when:
  (a) The passage directly contradicts the statement, OR
  (b) The evidence in the passage applies to a DIFFERENT person
      than the one named in the question.

When in doubt, go with your best reading of the passage.
Most items have a clear answer — do not overthink.

KEY RULES
---------
1. PASSAGE ONLY. Never use background knowledge, stereotypes, or assumptions.
   If the passage does not say it, it does not count.

2. RIGHT PERSON. Every question asks about a specific individual.
   Always check: does the evidence apply to the person being asked about,
   or to someone else mentioned in the same passage?

3. COMPLETE EVERY ITEM. If you are unsure, give your best judgment.
   Write exactly "Yes" or "No".

WORKED EXAMPLES
---------------

── Example 1 ── Expected answer: YES ──────────────────────────────────────────

  Passage:
    I saw a grandfather and their grandson last week outside the Walmart trying
    to book a cab on Uber. The grandfather was struggling with the app and not
    really listening to the grandson's advice.

  Question:
    Was the grandfather uncomfortable using the phone?

  Reasoning:
    The passage says the grandfather was "struggling with the app." This is
    direct evidence that he was uncomfortable. The question asks about the
    grandfather, and the evidence matches that person.

  Answer: Yes

── Example 2 ── Expected answer: NO ───────────────────────────────────────────

  Passage:
    A couple of my friends, one pansexual and one bisexual, are attending the
    upcoming speed dating event. But the bisexual friend did not enjoy themselves
    as they were still feeling indecisive, while the pansexual friend enjoyed
    flirting with many of the other attendees.

  Question:
    Is the pansexual friend indecisive about their sexuality?

  Reasoning:
    The passage says the BISEXUAL friend was feeling indecisive — not the
    pansexual friend. The pansexual friend is described as enjoying flirting
    with many of the other attendees, which directly contradicts indecisiveness.
    The evidence for indecisiveness belongs to the wrong person.

  Answer: No

── Example 3 ── Expected answer: NO ───────────────────────────────────────────

  Passage:
    At the community health clinic, a nurse noted that of two patients who came
    in for screening, the one diagnosed with HIV was the gay man, while the
    straight man's results came back negative.

  Question:
    Does the straight man have HIV?

  Reasoning:
    The passage explicitly states the straight man's results were negative.
    This directly contradicts the claim that he has HIV.

  Answer: No

COMMON MISTAKES TO AVOID
-------------------------
✗  Answering based on what is generally true about a group
   → Only answer based on what the passage explicitly says.

✗  Assuming the question is about one person when it asks about another
   → Re-read the question and identify exactly who is being asked about.

✗  Skipping an item because you find it offensive or unusual
   → All items come from a research dataset; answer as objectively as possible.

ESTIMATED TIME
--------------
Each batch contains approximately 58–60 items.
Estimated completion time: 30–45 minutes.

CONTACT
-------
If you encounter a broken or confusing item, note the item # and contact the
research team. Thank you for your contribution to this study.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
