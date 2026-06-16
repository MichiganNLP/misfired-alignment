"""
Apply annotations from stereotypes_review.csv back to stereotypes.json.

Actions per row:
  - delete=Y        → drop the entry entirely
  - fix_evidence    → overwrite explicit_evidence
  - fix_question    → overwrite question
  - fix_correct_answer (yes/no) → overwrite correct_answer
  - fix_notes       → overwrite notes (unless it's the sentinel "OK", which is just a review marker)

After updating stereotypes.json, re-runs generate_prompts.py to regenerate
data/prompt_pairs.json and data/prompt_pairs_no_trigger.json.

Usage:
    python scripts/apply_annotations.py [--dry-run]
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
CSV_PATH  = DATA_DIR / "stereotypes_review.csv"
JSON_PATH = DATA_DIR / "stereotypes.json"
GEN_SCRIPT = Path(__file__).parent / "generate_prompts.py"


def load_csv(path: Path) -> dict:
    """Return {id: row_dict} from the annotation CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def apply(dry_run: bool = False):
    annotations = load_csv(CSV_PATH)

    with open(JSON_PATH) as f:
        entries = json.load(f)

    kept = []
    stats = {"deleted": 0, "evidence_fixed": 0, "question_fixed": 0,
             "answer_fixed": 0, "notes_fixed": 0, "unchanged": 0}

    for entry in entries:
        eid = entry["id"]
        ann = annotations.get(eid)
        if ann is None:
            kept.append(entry)
            stats["unchanged"] += 1
            continue

        if ann.get("delete", "").strip().upper() == "Y":
            stats["deleted"] += 1
            print(f"  [DELETE] {eid}")
            continue

        changed = False
        if ann.get("fix_evidence", "").strip():
            entry["explicit_evidence"] = ann["fix_evidence"].strip()
            stats["evidence_fixed"] += 1
            changed = True

        if ann.get("fix_question", "").strip():
            entry["question"] = ann["fix_question"].strip()
            stats["question_fixed"] += 1
            changed = True

        fix_ans = ann.get("fix_correct_answer", "").strip().lower()
        if fix_ans in ("yes", "no"):
            entry["correct_answer"] = fix_ans
            stats["answer_fixed"] += 1
            changed = True

        fix_notes = ann.get("fix_notes", "").strip()
        if fix_notes and fix_notes != "OK":
            entry["notes"] = fix_notes
            stats["notes_fixed"] += 1
            changed = True

        if not changed:
            stats["unchanged"] += 1

        kept.append(entry)

    print(f"\nAnnotation summary:")
    print(f"  Original entries : {len(entries)}")
    print(f"  Deleted          : {stats['deleted']}")
    print(f"  Kept             : {len(kept)}")
    print(f"  Evidence fixed   : {stats['evidence_fixed']}")
    print(f"  Question fixed   : {stats['question_fixed']}")
    print(f"  Answer fixed     : {stats['answer_fixed']}")
    print(f"  Notes fixed      : {stats['notes_fixed']}")
    print(f"  Unchanged        : {stats['unchanged']}")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Back up original
    backup = JSON_PATH.with_suffix(".json.bak")
    backup.write_text(JSON_PATH.read_text())
    print(f"\nBacked up original to {backup}")

    with open(JSON_PATH, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"Wrote {len(kept)} entries to {JSON_PATH}")

    # Regenerate prompt pairs
    print("\nRegenerating prompt pairs...")
    result = subprocess.run(
        [sys.executable, str(GEN_SCRIPT)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: generate_prompts.py failed — prompt pairs not updated.")
        sys.exit(1)
    print("Done. Prompt pairs regenerated.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing anything")
    args = parser.parse_args()
    apply(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
