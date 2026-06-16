"""
Aggregate per-pair mechinterp JSONs into cross-pair claim figures.

Reads:
  data/mechinterp_pairs.json                       (role labels: failure / control)
  results/mechinterp/{model}/logit_lens/*.json     (per-layer logit_diff)
  results/mechinterp/{model}/activation_patching/*.json
  results/mechinterp/{model}/attention_analysis/*.json
  results/mechinterp/{model}/head_ablation/ablation_results.json

Writes (one set per model + a cross-model comparison):
  results/mechinterp/aggregate/
      {model}_logit_lens_profile.png      (failure vs control × stereo/contrast, mean±sem)
      {model}_recovery_profile.png        (activation patching, failures only)
      {model}_attn_diff_heatmap.png       (attention diff, failures-mean)
      cross_model_layer_profile.png       (instruct vs base, side by side)
      summary_per_pair.csv                (one row per pair × model)
      summary.json                        (all aggregate stats)

Usage:
  python scripts/mechinterp/aggregate.py
  python scripts/mechinterp/aggregate.py --models meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).parent.parent.parent
DATA_DIR = PROJ / "data"
MECH_DIR = PROJ / "results" / "mechinterp"


def safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def load_role_map(pairs_file: Path) -> dict[str, dict]:
    with open(pairs_file) as f:
        d = json.load(f)
    return {p["id"]: p for p in d["pairs"]}


# Per-model pairs-file mapping. The pair_id space is per-model: Llama failures
# don't match Mistral or Qwen3 failures, so cross-family role lookup must use
# each model's own pair set.
_PAIRS_FILE_BY_MODEL = {
    # Llama (Instruct, Base, and any future variant) — default pair set
    "meta-llama/Llama-3.1-8B-Instruct":            "mechinterp_pairs.json",
    "meta-llama/Llama-3.1-8B":                     "mechinterp_pairs.json",
    "mistralai/Mistral-7B-Instruct-v0.3":          "mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json",
    "mistralai/Mistral-7B-v0.3":                   "mechinterp_pairs_mistralai_Mistral-7B-Instruct-v0.3.json",
    "Qwen/Qwen3-8B":                               "mechinterp_pairs_Qwen_Qwen3-8B.json",
    "Qwen/Qwen3-8B-Base":                          "mechinterp_pairs_Qwen_Qwen3-8B.json",
    "Qwen/Qwen3.5-9B":                             "mechinterp_pairs_Qwen_Qwen3.5-9B.json",
    "Qwen/Qwen3.5-9B-Base":                        "mechinterp_pairs_Qwen_Qwen3.5-9B.json",
}


def role_map_for(model: str, default: dict[str, dict]) -> dict[str, dict]:
    """Return the role lookup keyed by this model's own pair set, falling back
    to `default` if the model has no specific mapping."""
    fname = _PAIRS_FILE_BY_MODEL.get(model)
    if fname is None:
        return default
    p = DATA_DIR / fname
    if not p.exists():
        return default
    return load_role_map(p)


def _stack(list_of_lists: list[list[float]]) -> np.ndarray:
    """Stack ragged lists into a 2D array, padding with NaN if needed."""
    if not list_of_lists:
        return np.zeros((0, 0))
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), np.nan, dtype=float)
    for i, x in enumerate(list_of_lists):
        arr[i, : len(x)] = x
    return arr


def mean_sem(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, sem) over axis=0, ignoring NaN."""
    if arr.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(~np.isnan(arr), axis=0), 1))
    return mean, sem


# ── Trajectory-shape analysis (Hypothesis B: two competing circuits) ────────

def _classify_trajectory(diffs: list[float]) -> dict:
    """Classify a per-layer logit-diff trajectory.

    For (B-yes) — alignment circuit suppresses factual circuit — failure pairs
    on the stereotyped prompt should show:
      - logit_diff > 0 at some early/mid layer (factual circuit reaches "yes")
      - logit_diff < 0 at the final layer (suppression vetoes)
    Equivalently: max(diffs) > 0 AND diffs[-1] < 0 — the trajectory crosses
    zero at least once. We call this a "handoff" pattern.

    Returns peak-positive layer, peak-negative layer, sign at final layer,
    and a boolean handoff flag.
    """
    if not diffs:
        return {}
    arr = np.array(diffs, dtype=float)
    peak_pos_layer = int(np.argmax(arr))
    peak_neg_layer = int(np.argmin(arr))
    final_sign = "pos" if arr[-1] > 0 else ("neg" if arr[-1] < 0 else "zero")
    early_pos = arr[: max(1, len(arr) // 2)].max() > 0  # any positive in first half
    final_neg = arr[-1] < 0
    handoff = early_pos and final_neg
    return {
        "peak_pos_layer":  peak_pos_layer,
        "peak_pos_value":  float(arr[peak_pos_layer]),
        "peak_neg_layer":  peak_neg_layer,
        "peak_neg_value":  float(arr[peak_neg_layer]),
        "final_sign":      final_sign,
        "handoff":         bool(handoff),
    }


def aggregate_trajectory(model: str, role_of: dict[str, dict]) -> dict:
    """Per-pair trajectory classification for the stereotyped prompt across
    pairs, plus role-level handoff rates. Reads existing logit-lens JSONs."""
    d = MECH_DIR / safe(model) / "logit_lens"
    if not d.exists():
        return {}
    pair_summary = {}
    by_role = defaultdict(lambda: {"handoff": 0, "n": 0,
                                    "peak_pos_layers": [],
                                    "peak_neg_layers": []})
    for jf in sorted(d.glob("*_logit_lens.json")):
        rec = json.load(open(jf))
        pid = rec["pair_id"]
        role = role_of.get(pid, {}).get("role", "unknown")
        s_traj = _classify_trajectory(rec["per_layer_logit_diff_stereo"])
        c_traj = _classify_trajectory(rec["per_layer_logit_diff_contrast"])
        pair_summary[pid] = {"role": role, "stereo": s_traj, "contrast": c_traj}
        by_role[role]["n"] += 1
        if s_traj.get("handoff"):
            by_role[role]["handoff"] += 1
        by_role[role]["peak_pos_layers"].append(s_traj.get("peak_pos_layer"))
        by_role[role]["peak_neg_layers"].append(s_traj.get("peak_neg_layer"))
    out = {"pair_summary": pair_summary, "by_role": {}}
    for role, d2 in by_role.items():
        n = d2["n"]
        out["by_role"][role] = {
            "n":             n,
            "handoff_count": d2["handoff"],
            "handoff_rate":  d2["handoff"] / n if n else 0.0,
            "mean_peak_pos_layer":
                float(np.mean([x for x in d2["peak_pos_layers"] if x is not None]))
                if d2["peak_pos_layers"] else None,
            "mean_peak_neg_layer":
                float(np.mean([x for x in d2["peak_neg_layers"] if x is not None]))
                if d2["peak_neg_layers"] else None,
        }
    return out


# ── Head-ablation recovery — parsed from logs (not in JSON) ──────────────────

_ABL_PAIR_RE  = re.compile(r"^\[([^\]]+)\] Head ablation sweep")
_ABL_BASE_RE  = re.compile(r"^\s*Baseline: answer='(\w+)', logit_diff=([+-]?\d+\.\d+)")
_ABL_TOPN_RE  = re.compile(r"^\s*Ablate top-(\d+): answer='(\w+)', logit_diff=([+-]?\d+\.\d+)\s+(✓ RECOVERED|✗)")


def parse_ablation_logs(log_paths: list[Path]) -> dict[str, dict]:
    """Parse one or more run logs to extract per-pair multi-head ablation
    recovery (top-1 / top-3 / top-5 / top-N). Returns {pair_id: {...}}."""
    pair_data: dict[str, dict] = {}
    for log in log_paths:
        if not log.exists():
            continue
        cur_pid = None
        with open(log) as fh:
            in_recovery_block = False
            for line in fh:
                m = _ABL_PAIR_RE.search(line)
                if m:
                    cur_pid = m.group(1)
                    in_recovery_block = False
                    continue
                if cur_pid is None:
                    continue
                if "Multi-head ablation recovery" in line:
                    in_recovery_block = True
                    pair_data.setdefault(cur_pid, {"recovery": {}})
                    continue
                if not in_recovery_block:
                    continue
                m = _ABL_BASE_RE.search(line)
                if m:
                    pair_data[cur_pid]["base_answer"] = m.group(1)
                    pair_data[cur_pid]["base_logit_diff"] = float(m.group(2))
                    continue
                m = _ABL_TOPN_RE.search(line)
                if m:
                    n = int(m.group(1))
                    pair_data[cur_pid]["recovery"][f"top_{n}"] = {
                        "answer":     m.group(2),
                        "logit_diff": float(m.group(3)),
                        "recovered":  m.group(4).startswith("✓"),
                    }
                    continue
                # Blank line ends the recovery block
                if line.strip() == "":
                    in_recovery_block = False
    return pair_data


def aggregate_ablation_recovery(model: str, role_of: dict[str, dict]) -> dict:
    """Per-pair head-ablation recovery + role-level recovery rates."""
    pattern_map = {
        "meta-llama/Llama-3.1-8B-Instruct":   ["run_instruct_*.log"],
        "meta-llama/Llama-3.1-8B":            ["run_base_*.log"],
        "mistralai/Mistral-7B-Instruct-v0.3": ["wave1_mistral_instruct_*.log"],
        "mistralai/Mistral-7B-v0.3":          ["wave2_mistral_base_*.log"],
        "Qwen/Qwen3-8B":                      ["wave1_qwen3_8b_*.log"],
        "Qwen/Qwen3-8B-Base":                 ["wave2_qwen3_8b_base_*.log"],
    }
    pats = pattern_map.get(model, [f"*{safe(model)}*.log"])
    candidate_logs = []
    for pat in pats:
        candidate_logs += sorted(MECH_DIR.glob(pat))
    pair_data = parse_ablation_logs(list({str(p): p for p in candidate_logs}.values()))
    if not pair_data:
        return {}

    by_role = defaultdict(lambda: {"n": 0, "top_1": 0, "top_3": 0, "top_5": 0, "top_10": 0})
    pair_summary = {}
    for pid, d in pair_data.items():
        role = role_of.get(pid, {}).get("role", "unknown")
        rec = d.get("recovery", {})
        pair_summary[pid] = {"role": role, **d}
        by_role[role]["n"] += 1
        for k in ("top_1", "top_3", "top_5", "top_10"):
            if rec.get(k, {}).get("recovered"):
                by_role[role][k] += 1
    out = {"pair_summary": pair_summary, "by_role": {}}
    for role, d2 in by_role.items():
        n = d2["n"]
        out["by_role"][role] = {"n": n, **{
            f"recovery_rate_{k}": d2[f"top_{k}"] / n if n else 0.0
            for k in (1, 3, 5, 10)
        }}
    return out


# ── Logit lens aggregation ───────────────────────────────────────────────────

def aggregate_logit_lens(model: str, role_of: dict[str, dict]) -> dict:
    """Mean ± sem of per-layer logit_diff for {stereo, contrast} × {failure, control}."""
    d = MECH_DIR / safe(model) / "logit_lens"
    if not d.exists():
        return {}
    by_role = defaultdict(lambda: {"stereo": [], "contrast": []})
    pair_summary = {}
    for jf in sorted(d.glob("*_logit_lens.json")):
        rec = json.load(open(jf))
        pid = rec["pair_id"]
        role = role_of.get(pid, {}).get("role", "unknown")
        by_role[role]["stereo"].append(rec["per_layer_logit_diff_stereo"])
        by_role[role]["contrast"].append(rec["per_layer_logit_diff_contrast"])
        pair_summary[pid] = {
            "role": role,
            "category": rec.get("category"),
            "final_logit_diff_stereo": rec["final_logit_diff_stereo"],
            "final_logit_diff_contrast": rec["final_logit_diff_contrast"],
            "divergence_layer": rec["divergence_layer"],
        }
    out = {"pair_summary": pair_summary, "by_role": {}}
    for role, d2 in by_role.items():
        s = _stack(d2["stereo"]); c = _stack(d2["contrast"])
        sm, ss = mean_sem(s); cm, cs = mean_sem(c)
        out["by_role"][role] = {
            "n": s.shape[0],
            "stereo_mean": sm.tolist(), "stereo_sem": ss.tolist(),
            "contrast_mean": cm.tolist(), "contrast_sem": cs.tolist(),
            "diff_mean": (cm - sm).tolist(),  # contrast − stereo
        }
    return out


def plot_logit_lens_profile(model: str, agg: dict, out_path: Path):
    if not agg.get("by_role"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    titles = {"failure": "Failure pairs (stereo wrong, contrast right)",
              "control": "Control pairs (both right)"}
    for ax, role in zip(axes, ["failure", "control"]):
        st = agg["by_role"].get(role)
        if not st:
            ax.set_visible(False); continue
        L = len(st["stereo_mean"])
        x = np.arange(L)
        ax.plot(x, st["stereo_mean"], color="tomato", lw=2,
                label=f"Stereotyped (n={st['n']})")
        ax.fill_between(x,
                        np.array(st["stereo_mean"]) - np.array(st["stereo_sem"]),
                        np.array(st["stereo_mean"]) + np.array(st["stereo_sem"]),
                        alpha=0.2, color="tomato")
        ax.plot(x, st["contrast_mean"], color="steelblue", lw=2,
                label=f"Contrast (n={st['n']})")
        ax.fill_between(x,
                        np.array(st["contrast_mean"]) - np.array(st["contrast_sem"]),
                        np.array(st["contrast_mean"]) + np.array(st["contrast_sem"]),
                        alpha=0.2, color="steelblue")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Layer"); ax.set_title(titles.get(role, role), fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean Logit(yes) − Logit(no)")
    fig.suptitle(f"Logit Lens — {model}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cross_model_layer_profile(per_model: dict, out_path: Path):
    """Side-by-side: stereotyped−contrast gap (Δ logit-diff) for each model,
    failure pairs only — the headline figure."""
    n = len(per_model)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    for ax, (model, agg) in zip(axes, per_model.items()):
        if not agg.get("by_role", {}).get("failure"): ax.set_visible(False); continue
        f = agg["by_role"]["failure"]; c = agg["by_role"].get("control")
        L = len(f["stereo_mean"]); x = np.arange(L)
        # Plot the contrast−stereo gap: positive = "yes is harder for stereo", which
        # is exactly the alignment-suppression signature.
        gap_f = np.array(f["contrast_mean"]) - np.array(f["stereo_mean"])
        ax.plot(x, gap_f, color="purple", lw=2.4, label=f"Failures (n={f['n']})")
        if c:
            gap_c = np.array(c["contrast_mean"]) - np.array(c["stereo_mean"])
            ax.plot(x, gap_c, color="gray", lw=1.8, ls="--",
                    label=f"Controls (n={c['n']})")
        ax.axhline(0, color="black", lw=0.7, ls="-", alpha=0.4)
        ax.set_xlabel("Layer"); ax.set_title(model, fontsize=10)
        ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ Logit-diff (contrast − stereotyped)")
    fig.suptitle("Cross-model layer profile: where does the model split groups?", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Activation patching aggregation ─────────────────────────────────────────

def aggregate_patching(model: str, role_of: dict[str, dict]) -> dict:
    d = MECH_DIR / safe(model) / "activation_patching"
    if not d.exists():
        return {}
    by_role = defaultdict(lambda: {"final": [], "group": []})
    pair_summary = {}
    for jf in sorted(d.glob("*_layer_sweep.json")):
        rec = json.load(open(jf))
        pid = rec["pair_id"]
        role = role_of.get(pid, {}).get("role", "unknown")
        by_role[role]["final"].append(rec["recovery_per_layer_final"])
        if rec.get("recovery_per_layer_group"):
            by_role[role]["group"].append(rec["recovery_per_layer_group"])
        pair_summary[pid] = {
            "role": role,
            "top_recovery_layer_final": int(np.argmax(rec["recovery_per_layer_final"])),
            "top_recovery_layer_group": int(np.argmax(rec["recovery_per_layer_group"]))
                if rec.get("recovery_per_layer_group") else None,
            "max_recovery_final": float(max(rec["recovery_per_layer_final"])),
        }
    out = {"pair_summary": pair_summary, "by_role": {}}
    for role, d2 in by_role.items():
        f = _stack(d2["final"]); g = _stack(d2["group"])
        fm, fs = mean_sem(f); gm, gs = mean_sem(g)
        out["by_role"][role] = {
            "n_final": f.shape[0], "n_group": g.shape[0],
            "final_mean": fm.tolist(), "final_sem": fs.tolist(),
            "group_mean": gm.tolist(), "group_sem": gs.tolist(),
        }
    return out


def plot_recovery_profile(model: str, agg: dict, out_path: Path):
    if not agg.get("by_role", {}).get("failure"):
        return
    f = agg["by_role"]["failure"]
    L = len(f["final_mean"]); x = np.arange(L)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, f["final_mean"], color="firebrick", lw=2,
            label=f"Patch at final token (n={f['n_final']})")
    ax.fill_between(x,
                    np.array(f["final_mean"]) - np.array(f["final_sem"]),
                    np.array(f["final_mean"]) + np.array(f["final_sem"]),
                    alpha=0.2, color="firebrick")
    if f.get("group_mean"):
        ax.plot(x, f["group_mean"], color="darkorange", lw=2,
                label=f"Patch at group token (n={f['n_group']})")
        ax.fill_between(x,
                        np.array(f["group_mean"]) - np.array(f["group_sem"]),
                        np.array(f["group_mean"]) + np.array(f["group_sem"]),
                        alpha=0.2, color="darkorange")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Recovery score")
    ax.set_title(f"Activation patching, failure pairs — {model}", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Attention aggregation ───────────────────────────────────────────────────

def aggregate_attention(model: str, role_of: dict[str, dict]) -> dict:
    d = MECH_DIR / safe(model) / "attention_analysis"
    if not d.exists():
        return {}
    by_role = defaultdict(list)
    pair_summary = {}
    for jf in sorted(d.glob("*_group_attn.json")):
        rec = json.load(open(jf))
        pid = rec["pair_id"]
        role = role_of.get(pid, {}).get("role", "unknown")
        diff = np.array(rec["attn_diff_stereo_minus_contrast"])  # [L,H]
        by_role[role].append(diff)
        # top differential head
        flat = diff.flatten()
        top_idx = int(np.argmax(flat))
        L, H = diff.shape
        pair_summary[pid] = {
            "role": role,
            "top_diff_layer": top_idx // H,
            "top_diff_head":  top_idx %  H,
            "top_diff_value": float(flat[top_idx]),
        }
    out = {"pair_summary": pair_summary, "by_role": {}}
    for role, arrs in by_role.items():
        if not arrs: continue
        stk = np.stack(arrs)  # [N, L, H]
        out["by_role"][role] = {"n": stk.shape[0], "mean": stk.mean(0).tolist(),
                                  "shape": list(stk.shape[1:])}
    return out


def plot_attn_diff_heatmap(model: str, agg: dict, out_path: Path):
    if not agg.get("by_role", {}).get("failure"):
        return
    f = agg["by_role"]["failure"]
    arr = np.array(f["mean"])  # [L, H]
    fig, ax = plt.subplots(figsize=(10, max(4, arr.shape[0] * 0.18)))
    vmax = np.abs(arr).max()
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Δ attn (stereo − contrast)")
    # Highlight top-5 across all layers/heads
    flat = arr.flatten()
    top5 = np.argpartition(flat, -5)[-5:]
    for idx in top5:
        li, hi = divmod(int(idx), arr.shape[1])
        ax.add_patch(plt.Rectangle((hi - 0.5, li - 0.5), 1, 1,
                                     fill=False, edgecolor="black", lw=1.5))
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title(f"Attention to group token, mean over {f['n']} failures — {model}",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["meta-llama/Llama-3.1-8B-Instruct",
                             "meta-llama/Llama-3.1-8B"])
    ap.add_argument("--pairs_file", default=str(DATA_DIR / "mechinterp_pairs.json"))
    ap.add_argument("--out", default=str(MECH_DIR / "aggregate"))
    args = ap.parse_args()

    role_of = load_role_map(Path(args.pairs_file))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"pairs_file": args.pairs_file, "models": {}}
    csv_rows = []

    for model in args.models:
        print(f"\n=== {model} ===")
        # Use the model's OWN pair set for role lookup (cross-family).
        rm = role_map_for(model, role_of)
        ll  = aggregate_logit_lens(model, rm)
        ap_ = aggregate_patching(model, rm)
        att = aggregate_attention(model, rm)
        traj = aggregate_trajectory(model, rm)
        abl  = aggregate_ablation_recovery(model, rm)

        summary["models"][model] = {
            "logit_lens": {k: v for k, v in ll.items() if k != "pair_summary"},
            "patching":   {k: v for k, v in ap_.items() if k != "pair_summary"},
            "attention":  {k: v for k, v in att.items() if k != "pair_summary"},
            "trajectory": {k: v for k, v in traj.items() if k != "pair_summary"},
            "ablation_recovery": {k: v for k, v in abl.items() if k != "pair_summary"},
        }

        if ll:
            plot_logit_lens_profile(model, ll, out_dir / f"{safe(model)}_logit_lens_profile.png")
            print(f"  logit lens: {sum(b['n'] for b in ll['by_role'].values())} pairs")
        if ap_:
            plot_recovery_profile(model, ap_, out_dir / f"{safe(model)}_recovery_profile.png")
            print(f"  patching:   {sum(b['n_final'] for b in ap_['by_role'].values())} pairs")
        if att:
            plot_attn_diff_heatmap(model, att, out_dir / f"{safe(model)}_attn_diff_heatmap.png")
            print(f"  attention:  {sum(b['n'] for b in att['by_role'].values())} pairs")
        if traj:
            tr = traj["by_role"]
            for role, st in tr.items():
                print(f"  trajectory [{role}]:  handoff {st['handoff_count']}/{st['n']} = {st['handoff_rate']:.1%}  "
                      f"(peak+ ≈ L{st['mean_peak_pos_layer']:.1f}, peak- ≈ L{st['mean_peak_neg_layer']:.1f})"
                      if st.get('mean_peak_pos_layer') is not None else
                      f"  trajectory [{role}]:  handoff {st['handoff_count']}/{st['n']} = {st['handoff_rate']:.1%}")
        if abl:
            ab = abl["by_role"]
            for role, st in ab.items():
                rates = " ".join(f"top-{k}:{st[f'recovery_rate_{k}']:.0%}" for k in (1,3,5,10))
                print(f"  ablation rec [{role}]:  n={st['n']}  {rates}")

        # Per-pair CSV rows
        for pid, info in (ll.get("pair_summary") or {}).items():
            traj_p = (traj.get("pair_summary") or {}).get(pid, {})
            abl_p  = (abl.get("pair_summary") or {}).get(pid, {})
            csv_rows.append({
                "model": model, "pair_id": pid,
                "role": info["role"], "category": info["category"],
                "final_logit_diff_stereo": info["final_logit_diff_stereo"],
                "final_logit_diff_contrast": info["final_logit_diff_contrast"],
                "divergence_layer": info["divergence_layer"],
                "top_recovery_layer_final": (ap_.get("pair_summary") or {}).get(pid, {}).get("top_recovery_layer_final"),
                "max_recovery_final":       (ap_.get("pair_summary") or {}).get(pid, {}).get("max_recovery_final"),
                "top_diff_layer":           (att.get("pair_summary") or {}).get(pid, {}).get("top_diff_layer"),
                "top_diff_head":            (att.get("pair_summary") or {}).get(pid, {}).get("top_diff_head"),
                "stereo_traj_handoff":      (traj_p.get("stereo") or {}).get("handoff"),
                "stereo_peak_pos_layer":    (traj_p.get("stereo") or {}).get("peak_pos_layer"),
                "stereo_peak_neg_layer":    (traj_p.get("stereo") or {}).get("peak_neg_layer"),
                "ablation_top_10_recovered":
                    (abl_p.get("recovery") or {}).get("top_10", {}).get("recovered"),
            })

    # Cross-model headline figure
    per_model_ll = {m: aggregate_logit_lens(m, role_of) for m in args.models}
    plot_cross_model_layer_profile(per_model_ll, out_dir / "cross_model_layer_profile.png")

    # Write summary + CSV
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if csv_rows:
        with open(out_dir / "summary_per_pair.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            w.writeheader(); w.writerows(csv_rows)
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
