"""
Eval-aligned logit lens. Reuses the same per-pair probe as logit_lens.py but
under prompt formatting that matches the behavioural-eval pipeline:

  - SYSTEM_PROMPT = eval's JSON-format scaffolding (instead of mechinterp's
                    bare yes/no instruction)
  - User message  = pair's user message with `apply_direct_json` applied
                    (i.e. tail rewritten to "Respond with JSON: {...}")
  - Chat template = with add_generation_prompt=True
  - Then APPEND `{"answer": "` so the very next token in the generation
                    sequence is yes or no.
  - Logit lens at every layer, projecting the residual at the FINAL token
                    (the `"`) through final-norm + lm_head.

Output: results/mechinterp/<model>/logit_lens_eval_aligned/<pair_id>.json
        Same schema as the bare logit_lens output for easy comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import utils
from utils import (
    apply_chat_template, build_prompt_pairs, get_decoder_handles,
    get_logit_diff, get_yes_no_ids, load_model, set_run_context,
    DEFAULT_PAIRS_FILE, ANALYSIS_EXAMPLES, load_pair_dicts,
)

# Import eval's JSON system prompt + tail-rewrite logic
sys.path.insert(0, str(Path(__file__).parent.parent))
import evaluate as ev   # noqa: E402

JSON_PREFIX = '{"answer": "'   # what we append after the assistant role start


def build_eval_aligned_prompt(tokenizer, user_message: str) -> str:
    """Return the prompt string for the (B-eval-aligned) probe.

    Equivalent to: chat-template(eval-system-prompt + apply_direct_json(user))
                   + JSON_PREFIX
    """
    user_with_json_tail = ev.apply_direct_json(user_message)
    messages = [
        {"role": "system",  "content": ev.SYSTEM_PROMPT},
        {"role": "user",    "content": user_with_json_tail},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return text + JSON_PREFIX


def compute_logit_lens_at_last_token(model, tokenizer, prompt: str,
                                     yes_ids, no_ids):
    """Per-layer (yes-max − no-max) logit-diff at the final token, plus
    final-layer top answer."""
    H = get_decoder_handles(model)
    residuals: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            residuals[idx] = h.detach()
        return hook_fn

    hooks = [layer.register_forward_hook(make_hook(i))
             for i, layer in enumerate(H.layers)]
    inputs = tokenizer(prompt, return_tensors="pt").to(H.embed.weight.device)
    with torch.no_grad():
        out = model(**inputs)
    for h in hooks: h.remove()

    final_norm = H.norm
    lm_head    = H.lm_head
    diffs = []
    with torch.no_grad():
        for i in range(H.n_layers):
            h = residuals[i].to(final_norm.weight.device)
            normed = final_norm(h)
            logits = lm_head(normed.to(lm_head.weight.device))
            diff = get_logit_diff(logits, yes_ids, no_ids)
            diffs.append(diff)
    final_logits = out.logits
    final_diff = get_logit_diff(final_logits, yes_ids, no_ids)
    last = final_logits[0, -1]
    yes_max = max(last[i].item() for i in yes_ids)
    no_max  = max(last[i].item() for i in no_ids)
    final_answer = "yes" if yes_max > no_max else "no"
    return diffs, final_diff, final_answer


def run(model_name: str, pairs_file: str, device: str,
        tokenizer_override: str | None = None):
    set_run_context(model_name)
    out_dir = utils.RESULTS_DIR / "logit_lens_eval_aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    t0 = time.time()
    model, tokenizer = load_model(model_name, device=device,
                                  tokenizer_name=tokenizer_override)
    print(f"Model loaded in {time.time() - t0:.0f}s")

    pair_dicts = load_pair_dicts(pairs_file)
    yes_ids, no_ids = get_yes_no_ids(tokenizer)
    print(f"yes ids: {yes_ids}\nno  ids: {no_ids}")
    print(f"Processing {len(pair_dicts)} pairs ...\n" + "=" * 60)

    # Build pair objects but with the eval-aligned prompt
    role_map = {p["id"]: p.get("role", "unknown") for p in pair_dicts}
    for i, ex in enumerate(pair_dicts, 1):
        out_path = out_dir / f"{ex['id']}_logit_lens.json"
        if out_path.exists(): continue   # resume

        t_pair = time.time()
        s_prompt = build_eval_aligned_prompt(tokenizer, ex["stereotyped_user"])
        c_prompt = build_eval_aligned_prompt(tokenizer, ex["contrast_user"])

        s_diffs, s_final, s_ans = compute_logit_lens_at_last_token(
            model, tokenizer, s_prompt, yes_ids, no_ids)
        c_diffs, c_final, c_ans = compute_logit_lens_at_last_token(
            model, tokenizer, c_prompt, yes_ids, no_ids)

        # Layer of max divergence (contrast - stereo gap)
        gap = [c - s for s, c in zip(s_diffs, c_diffs)]
        div_layer = int(max(range(len(gap)), key=lambda L: gap[L]))

        record = {
            "pair_id":            ex["id"],
            "category":           ex["category"],
            "stereotyped_group":  ex["stereotyped_group"],
            "contrast_group":     ex["contrast_group"],
            "role":               role_map.get(ex["id"], "unknown"),
            "final_answer_stereo":     s_ans,
            "final_answer_contrast":   c_ans,
            "final_logit_diff_stereo":   round(s_final, 4),
            "final_logit_diff_contrast": round(c_final, 4),
            "per_layer_logit_diff_stereo":   [round(d, 4) for d in s_diffs],
            "per_layer_logit_diff_contrast": [round(d, 4) for d in c_diffs],
            "divergence_layer":   div_layer,
            "probe":              "eval_aligned_json_prefix",
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        print(f"[{i}/{len(pair_dicts)}] {ex['id']:<26} "
              f"role={role_map.get(ex['id'],'?'):<7} "
              f"stereo final={s_final:+.2f} ans={s_ans} | "
              f"contrast final={c_final:+.2f} ans={c_ans}  "
              f"div_layer=L{div_layer}  ({time.time()-t_pair:.1f}s)")

    print("\nDone.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pairs_file", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer", default=None)
    args = p.parse_args()
    run(args.model, args.pairs_file, args.device, args.tokenizer)


if __name__ == "__main__":
    main()
