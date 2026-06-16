"""
Human annotation web app for fairness-logic prompt pairs.

Supports free navigation: annotators can go forward/backward or jump to
any item. Answers are stored in a per-annotator JSON dict on disk so the
session cookie stays tiny and progress survives page reloads.

Usage:
  python scripts/annotation/app.py [--host 127.0.0.1] [--port 5000]

Results:
  data/annotation_results/<name>_progress.json  — live dict {task_id: answer}
  data/annotation_results/<name>_final.json      — written on completion
"""

import hashlib
import json
import os
import random
import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, session, request, redirect, url_for, render_template, send_file

PROJ_DIR    = Path(__file__).parent.parent.parent
TASK_FILE   = PROJ_DIR / "data" / "annotation_task.json"
RESULTS_DIR = PROJ_DIR / "data" / "annotation_results"
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.urandom(32)

with open(TASK_FILE) as f:
    ALL_ITEMS = json.load(f)
TOTAL = len(ALL_ITEMS)


# ── Per-annotator helpers ─────────────────────────────────────────────────────

def get_order(name: str) -> list[int]:
    """Deterministic shuffle from annotator name — stable across page reloads."""
    seed = int(hashlib.md5(name.lower().strip().encode()).hexdigest(), 16) % (2 ** 32)
    rng  = random.Random(seed)
    order = list(range(TOTAL))
    rng.shuffle(order)
    return order


def progress_file(name: str) -> Path:
    safe = name.strip().lower().replace(" ", "_")
    return RESULTS_DIR / f"{safe}_progress.json"


def load_answers(name: str) -> dict:
    """Return {task_id: answer} for all answered items."""
    path = progress_file(name)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_answers(name: str, answers: dict) -> None:
    with open(progress_file(name), "w") as f:
        json.dump(answers, f)


def finalize(name: str) -> Path:
    order   = get_order(name)
    answers = load_answers(name)
    records = []
    for pos, item_idx in enumerate(order):
        item   = ALL_ITEMS[item_idx]
        answer = answers.get(item["task_id"])
        if answer is None:
            continue
        records.append({
            "position":  pos + 1,
            "task_id":   item["task_id"],
            "pair_id":   item["pair_id"],
            "condition": item["condition"],
            "item_type": item["item_type"],
            "category":  item["category"],
            "expected":  item["expected"],
            "answer":    answer,
            "correct":   answer == item["expected"],
        })
    safe = name.strip().lower().replace(" ", "_")
    path = RESULTS_DIR / f"{safe}_final.json"
    with open(path, "w") as f:
        json.dump({"annotator": name, "timestamp": datetime.now().isoformat(),
                   "n_items": len(records), "results": records}, f, indent=2)
    return path


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def start():
    error = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            error = "Please enter your name before continuing."
        else:
            session.clear()
            session["name"] = name
            return redirect(url_for("annotate"))
    return render_template("start.html", error=error, total=TOTAL)


@app.route("/annotate", methods=["GET", "POST"])
def annotate():
    if "name" not in session:
        return redirect(url_for("start"))

    name    = session["name"]
    order   = get_order(name)
    answers = load_answers(name)

    # ── Handle answer submission ──────────────────────────────────────────────
    if request.method == "POST":
        answer      = request.form.get("answer", "").lower()
        current_idx = int(request.form.get("current_idx", 0))
        next_idx    = request.form.get("next_idx")   # explicit nav target, may be None

        if answer in ("yes", "no"):
            task_id = ALL_ITEMS[order[current_idx]]["task_id"]
            answers[task_id] = answer
            save_answers(name, answers)

        # All done?
        if len(answers) == TOTAL:
            path = finalize(name)
            session["final_file"] = str(path)
            return redirect(url_for("done"))

        # Navigate to explicit target, or to next unanswered after current
        if next_idx is not None:
            return redirect(url_for("annotate", idx=int(next_idx)))

        for i in range(current_idx + 1, TOTAL):
            if ALL_ITEMS[order[i]]["task_id"] not in answers:
                return redirect(url_for("annotate", idx=i))
        # Wrap around to first unanswered
        for i in range(0, current_idx + 1):
            if ALL_ITEMS[order[i]]["task_id"] not in answers:
                return redirect(url_for("annotate", idx=i))
        return redirect(url_for("done"))

    # ── Render page ───────────────────────────────────────────────────────────
    # Determine which item to show
    try:
        idx = int(request.args.get("idx", -1))
    except (TypeError, ValueError):
        idx = -1

    if idx < 0 or idx >= TOTAL:
        # Default: first unanswered, or item 0
        idx = next((i for i in range(TOTAL)
                    if ALL_ITEMS[order[i]]["task_id"] not in answers), 0)

    item = ALL_ITEMS[order[idx]]

    # Build status list for sidebar: list of (display_pos, answer_or_None)
    status = [answers.get(ALL_ITEMS[order[i]]["task_id"]) for i in range(TOTAL)]
    n_done = sum(1 for a in status if a is not None)

    return render_template("annotate.html",
                           prompt=item["prompt"],
                           current_idx=idx,
                           total=TOTAL,
                           n_done=n_done,
                           progress=round(n_done / TOTAL * 100),
                           status=status,         # list of "yes"/"no"/None per position
                           current_answer=answers.get(item["task_id"]))


@app.route("/done")
def done():
    name    = session.get("name", "Annotator")
    answers = load_answers(name)
    n_yes   = sum(1 for a in answers.values() if a == "yes")
    n_no    = sum(1 for a in answers.values() if a == "no")
    return render_template("done.html", name=name, n_yes=n_yes, n_no=n_no, total=TOTAL)


@app.route("/download")
def download():
    path = session.get("final_file")
    if not path or not Path(path).exists():
        return "Result file not found.", 404
    return send_file(path, as_attachment=True,
                     download_name=Path(path).name,
                     mimetype="application/json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Annotation app running at http://{args.host}:{args.port}")
    print(f"Task: {TOTAL} items  |  Results → {RESULTS_DIR}")
    app.run(host=args.host, port=args.port, debug=False)
