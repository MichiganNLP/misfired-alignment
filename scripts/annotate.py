"""
Annotation web interface for stereotypes_review.csv.

Usage:
    python scripts/annotate.py
    # then open http://localhost:7860 in your browser
    # (or port-forward: ssh -L 7860:localhost:7860 <server>)

Keyboard shortcuts (when focus is NOT in a text field):
    j / ArrowDown  — next entry
    k / ArrowUp    — previous entry
    d              — toggle delete
    a              — approve (mark reviewed, no changes)
    s              — save to disk
"""

import csv
import json
import os
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string

DATA_DIR = Path(__file__).parent.parent / "data"
CSV_PATH = DATA_DIR / "stereotypes_review.csv"

app = Flask(__name__)

# ── Data layer ────────────────────────────────────────────────────────────────

def load_entries():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_entries(entries):
    if not entries:
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=entries[0].keys())
        w.writeheader()
        w.writerows(entries)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/entries")
def get_entries():
    return jsonify(load_entries())

@app.post("/api/entry/<entry_id>")
def update_entry(entry_id):
    entries = load_entries()
    payload = request.json
    for e in entries:
        if e["id"] == entry_id:
            for k in ("delete", "fix_evidence", "fix_question",
                      "fix_correct_answer", "fix_notes"):
                if k in payload:
                    e[k] = payload[k]
            break
    save_entries(entries)
    return jsonify({"ok": True})

@app.post("/api/save")
def save_all():
    entries = request.json
    # Merge incoming edits back (entries come from client state)
    existing = {e["id"]: e for e in load_entries()}
    for e in entries:
        if e["id"] in existing:
            for k in ("delete", "fix_evidence", "fix_question",
                      "fix_correct_answer", "fix_notes"):
                existing[e["id"]][k] = e.get(k, "")
    save_entries(list(existing.values()))
    return jsonify({"ok": True, "saved": len(entries)})

@app.get("/api/stats")
def stats():
    entries = load_entries()
    n = len(entries)
    deleted   = sum(1 for e in entries if e.get("delete", "").strip().upper() == "Y")
    edited    = sum(1 for e in entries if any(
        e.get(k, "").strip()
        for k in ("fix_evidence", "fix_question", "fix_correct_answer", "fix_notes")
    ))
    reviewed  = sum(1 for e in entries if e.get("fix_notes", "").strip() == "OK")
    return jsonify({"total": n, "deleted": deleted, "edited": edited, "approved": reviewed})

# ── Main page ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fairness-Logic Annotation</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; }
  .entry-row { cursor: pointer; transition: background 0.1s; }
  .entry-row:hover { background: #f0f9ff; }
  .entry-row.selected { background: #dbeafe !important; }
  .entry-row.deleted { background: #fee2e2 !important; opacity: 0.7; }
  .entry-row.edited  { background: #fef9c3 !important; }
  .entry-row.approved{ background: #dcfce7 !important; }
  .badge { padding: 2px 6px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .prompt-box { font-size: 12px; line-height: 1.5; background: #f8fafc;
                border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px;
                white-space: pre-wrap; word-break: break-word; max-height: 140px; overflow-y: auto; }
  #detail-panel { position: sticky; top: 0; height: 100vh; overflow-y: auto; }
  #list-panel   { height: 100vh; overflow-y: auto; }
  textarea { font-size: 13px; }
</style>
</head>
<body class="bg-gray-50">

<!-- Top bar -->
<div class="bg-white border-b px-4 py-2 flex items-center gap-4 sticky top-0 z-10 shadow-sm">
  <span class="font-bold text-blue-700 text-lg">Fairness-Logic Annotator</span>
  <div id="stats" class="text-sm text-gray-500 flex gap-3"></div>
  <div class="flex gap-2 ml-auto">
    <select id="cat-filter" class="border rounded px-2 py-1 text-sm">
      <option value="">All categories</option>
    </select>
    <select id="status-filter" class="border rounded px-2 py-1 text-sm">
      <option value="">All statuses</option>
      <option value="unreviewed">Unreviewed</option>
      <option value="deleted">Deleted</option>
      <option value="edited">Edited</option>
      <option value="approved">Approved</option>
    </select>
    <button onclick="saveAll()" class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">
      💾 Save
    </button>
  </div>
</div>

<div class="flex h-[calc(100vh-49px)]">

  <!-- Left: entry list -->
  <div id="list-panel" class="w-80 border-r bg-white flex-shrink-0 overflow-y-auto">
    <div id="entry-list" class="divide-y text-sm"></div>
  </div>

  <!-- Right: detail + edit -->
  <div id="detail-panel" class="flex-1 overflow-y-auto p-5">
    <div id="detail" class="text-gray-400 mt-20 text-center text-lg">← Select an entry</div>
  </div>

</div>

<script>
let ALL = [];       // full dataset
let FILTERED = [];  // currently shown
let CURRENT = -1;   // index into FILTERED
let DIRTY = {};     // {id: entry} with unsaved changes

// ── Load ──────────────────────────────────────────────────────────────────────
async function init() {
  const res = await fetch('/api/entries');
  ALL = await res.json();
  // Populate category filter
  const cats = [...new Set(ALL.map(e => e.category))].sort();
  const sel = document.getElementById('cat-filter');
  cats.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; sel.appendChild(o); });
  applyFilters();
  refreshStats();
  document.addEventListener('keydown', onKey);
}

function entryStatus(e) {
  if ((e.delete||'').trim().toUpperCase() === 'Y') return 'deleted';
  if ((e.fix_notes||'').trim() === 'OK') return 'approved';
  if (['fix_evidence','fix_question','fix_correct_answer','fix_notes'].some(k => (e[k]||'').trim())) return 'edited';
  return 'unreviewed';
}

// ── Filters ───────────────────────────────────────────────────────────────────
function applyFilters() {
  const cat    = document.getElementById('cat-filter').value;
  const status = document.getElementById('status-filter').value;
  FILTERED = ALL.filter(e => {
    if (cat    && e.category !== cat) return false;
    if (status && entryStatus(e) !== status) return false;
    return true;
  });
  renderList();
  if (FILTERED.length) selectEntry(0);
  else document.getElementById('detail').innerHTML = '<div class="text-gray-400 mt-20 text-center">No entries match filter</div>';
}
document.getElementById('cat-filter').addEventListener('change', applyFilters);
document.getElementById('status-filter').addEventListener('change', applyFilters);

// ── List render ───────────────────────────────────────────────────────────────
function renderList() {
  const el = document.getElementById('entry-list');
  el.innerHTML = '';
  FILTERED.forEach((e, i) => {
    const status = entryStatus(e);
    const row = document.createElement('div');
    row.className = `entry-row p-2 ${status}`;
    row.dataset.idx = i;
    row.innerHTML = `
      <div class="font-mono text-xs text-gray-400">${e.id}</div>
      <div class="font-medium text-xs truncate">${e.stereotyped_group} → ${e.contrast_group}</div>
      <div class="text-xs text-gray-500 truncate italic">${e.stereotype_phrase}</div>
      <span class="badge ${statusBadgeClass(status)}">${status}</span>
    `;
    row.addEventListener('click', () => selectEntry(i));
    el.appendChild(row);
  });
  highlightRow();
}

function statusBadgeClass(s) {
  return {deleted:'bg-red-100 text-red-700', edited:'bg-yellow-100 text-yellow-700',
          approved:'bg-green-100 text-green-700', unreviewed:'bg-gray-100 text-gray-500'}[s] || '';
}

function highlightRow() {
  document.querySelectorAll('.entry-row').forEach((r,i) => {
    r.classList.toggle('selected', i === CURRENT);
  });
  const el = document.querySelector(`.entry-row[data-idx="${CURRENT}"]`);
  if (el) el.scrollIntoView({block:'nearest'});
}

// ── Detail render ─────────────────────────────────────────────────────────────
function selectEntry(idx) {
  CURRENT = idx;
  highlightRow();
  renderDetail(FILTERED[idx]);
}

function renderDetail(e) {
  const status = entryStatus(e);
  document.getElementById('detail').innerHTML = `
    <div class="max-w-3xl mx-auto space-y-4">

      <!-- Header -->
      <div class="flex items-center gap-3 flex-wrap">
        <span class="font-mono text-xs text-gray-400">${e.id}</span>
        <span class="badge bg-blue-100 text-blue-700">${e.category}</span>
        <span class="badge ${statusBadgeClass(status)}">${status}</span>
        <div class="ml-auto flex gap-2">
          <button onclick="toggleDelete()" class="px-3 py-1 rounded text-sm border ${(e.delete||'').toUpperCase()==='Y' ? 'bg-red-500 text-white' : 'border-red-400 text-red-600 hover:bg-red-50'}">
            🗑 ${(e.delete||'').toUpperCase()==='Y' ? 'Undelete' : 'Delete'}
          </button>
          <button onclick="approve()" class="px-3 py-1 rounded text-sm border border-green-500 text-green-700 hover:bg-green-50">
            ✓ Approve
          </button>
        </div>
      </div>

      <!-- Groups + stereotype -->
      <div class="grid grid-cols-2 gap-3 text-sm">
        <div><span class="text-gray-500">Stereotyped group:</span> <strong>${e.stereotyped_group}</strong></div>
        <div><span class="text-gray-500">Contrast group:</span> <strong>${e.contrast_group}</strong></div>
        <div class="col-span-2"><span class="text-gray-500">Stereotype phrase:</span> <em>${e.stereotype_phrase}</em></div>
      </div>

      <!-- Prompts side-by-side -->
      <div class="grid grid-cols-2 gap-3">
        <div>
          <div class="text-xs font-semibold text-gray-500 mb-1">STEREOTYPED PROMPT</div>
          <div class="prompt-box">${escHtml(e.stereo_prompt)}</div>
        </div>
        <div>
          <div class="text-xs font-semibold text-gray-500 mb-1">CONTRAST PROMPT</div>
          <div class="prompt-box">${escHtml(e.contrast_prompt)}</div>
        </div>
      </div>

      <!-- Core fields -->
      <div class="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div class="text-gray-500 text-xs mb-1">Evidence</div>
          <div class="prompt-box text-xs">${escHtml(e.explicit_evidence)}</div>
        </div>
        <div>
          <div class="text-gray-500 text-xs mb-1">Question template</div>
          <div class="prompt-box text-xs">${escHtml(e.question)}</div>
        </div>
      </div>

      <div class="flex items-center gap-4 text-sm">
        <div><span class="text-gray-500">Correct answer:</span>
          <span class="font-bold ${e.correct_answer==='yes'?'text-green-600':'text-red-600'}">${e.correct_answer}</span>
        </div>
        ${e.notes ? `<div class="text-gray-400 text-xs italic">${escHtml(e.notes)}</div>` : ''}
      </div>

      <!-- Edit section -->
      <div class="border-t pt-4 space-y-3">
        <div class="font-semibold text-sm text-gray-700">Fixes (leave blank if no change needed)</div>

        <div>
          <label class="text-xs text-gray-500">Fix evidence</label>
          <textarea id="fix_evidence" rows="3" class="w-full border rounded p-2 mt-1" placeholder="Corrected evidence text…">${escHtml(e.fix_evidence||'')}</textarea>
        </div>
        <div>
          <label class="text-xs text-gray-500">Fix question</label>
          <textarea id="fix_question" rows="2" class="w-full border rounded p-2 mt-1" placeholder="Corrected question template…">${escHtml(e.fix_question||'')}</textarea>
        </div>
        <div>
          <label class="text-xs text-gray-500">Fix correct answer</label>
          <select id="fix_correct_answer" class="border rounded p-2 mt-1 text-sm">
            <option value="">— no change —</option>
            <option value="yes" ${e.fix_correct_answer==='yes'?'selected':''}>yes</option>
            <option value="no"  ${e.fix_correct_answer==='no' ?'selected':''}>no</option>
          </select>
        </div>
        <div>
          <label class="text-xs text-gray-500">Notes (write "OK" to mark as approved)</label>
          <textarea id="fix_notes" rows="2" class="w-full border rounded p-2 mt-1" placeholder='Notes, or "OK" to approve'>${escHtml(e.fix_notes||'')}</textarea>
        </div>

        <div class="flex gap-2">
          <button onclick="saveEntry()" class="bg-blue-600 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-700">
            Save entry  <span class="text-blue-200 text-xs">[Enter]</span>
          </button>
          <button onclick="saveEntryNext()" class="bg-blue-500 text-white px-4 py-1.5 rounded text-sm hover:bg-blue-600">
            Save + Next  <span class="text-blue-200 text-xs">[Shift+Enter]</span>
          </button>
        </div>
      </div>
    </div>
  `;
}

function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Actions ───────────────────────────────────────────────────────────────────
function currentEntry() { return FILTERED[CURRENT]; }

function toggleDelete() {
  const e = currentEntry();
  e.delete = (e.delete||'').toUpperCase() === 'Y' ? '' : 'Y';
  persistEntry(e);
}

function approve() {
  const e = currentEntry();
  e.fix_notes = 'OK';
  persistEntry(e);
  goNext();
}

function saveEntry() {
  const e = currentEntry();
  e.fix_evidence       = document.getElementById('fix_evidence').value;
  e.fix_question       = document.getElementById('fix_question').value;
  e.fix_correct_answer = document.getElementById('fix_correct_answer').value;
  e.fix_notes          = document.getElementById('fix_notes').value;
  persistEntry(e);
}

function saveEntryNext() { saveEntry(); goNext(); }

async function persistEntry(e) {
  await fetch(`/api/entry/${e.id}`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(e)
  });
  renderList();
  renderDetail(e);
  refreshStats();
}

async function saveAll() {
  await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(ALL)
  });
  const btn = document.querySelector('button[onclick="saveAll()"]');
  btn.textContent = '✓ Saved!';
  setTimeout(() => btn.textContent = '💾 Save', 1500);
}

async function refreshStats() {
  const s = await (await fetch('/api/stats')).json();
  document.getElementById('stats').innerHTML =
    `<span>Total: <b>${s.total}</b></span>` +
    `<span class="text-green-600">Approved: <b>${s.approved}</b></span>` +
    `<span class="text-yellow-600">Edited: <b>${s.edited}</b></span>` +
    `<span class="text-red-500">Deleted: <b>${s.deleted}</b></span>` +
    `<span class="text-gray-400">Unreviewed: <b>${s.total - s.approved - s.edited - s.deleted}</b></span>`;
}

// ── Navigation ────────────────────────────────────────────────────────────────
function goNext() { if (CURRENT < FILTERED.length-1) selectEntry(CURRENT+1); }
function goPrev() { if (CURRENT > 0) selectEntry(CURRENT-1); }

function onKey(ev) {
  const tag = document.activeElement.tagName.toLowerCase();
  const inInput = ['input','textarea','select'].includes(tag);
  if (ev.key === 'Enter' && inInput) {
    if (ev.shiftKey) { ev.preventDefault(); saveEntryNext(); }
    return;
  }
  if (inInput) return;
  if (ev.key === 'j' || ev.key === 'ArrowDown') { ev.preventDefault(); goNext(); }
  if (ev.key === 'k' || ev.key === 'ArrowUp')   { ev.preventDefault(); goPrev(); }
  if (ev.key === 'd') toggleDelete();
  if (ev.key === 'a') approve();
  if (ev.key === 's') { ev.preventDefault(); saveAll(); }
}

init();
</script>
</body>
</html>"""

@app.get("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n  Annotator running at http://localhost:{port}")
    print(f"  Data file: {CSV_PATH}")
    print(f"  Port-forward: ssh -L {port}:localhost:{port} <your-server>\n")
    app.run(host="0.0.0.0", port=port, debug=False)
