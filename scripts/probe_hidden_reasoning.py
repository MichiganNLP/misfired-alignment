"""
Diagnostic: probe each closed-source model with one of our actual eval
prompts and inspect the response metadata to determine whether hidden
reasoning is happening by default.

For each model we record:
  - which provider / endpoint it's routed through
  - the parsed final answer
  - usage.prompt_tokens, completion_tokens, total_tokens
  - usage.completion_tokens_details.reasoning_tokens (OpenAI / OpenRouter)
  - any reasoning_content or extended-thinking blocks in the response
  - any provider-specific thinking metadata (Gemini thoughts_token_count, etc.)

Output: results/probe_hidden_reasoning.json (one entry per model)
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

PROJ = Path(__file__).parent.parent
DATA = PROJ / "data"
OUT  = PROJ / "results" / "probe_hidden_reasoning.json"

# ── Test prompt: stereo half of age_q1_p0 ────────────────────────────────────
PAIRS = json.load(open(DATA / "prompt_pairs_bbq.json"))
PAIR  = next(p for p in PAIRS if p["id"] == "age_q1_p0")
USER_PROMPT = (
    PAIR["prompts"]["stereotyped"]["text"]
    + " Respond with JSON: {\"answer\": \"yes\"} or {\"answer\": \"no\"}."
)
SYSTEM_PROMPT = "You are a careful assistant. Answer factual yes/no questions strictly from the supplied context."

print(f"Test prompt ({len(USER_PROMPT)} chars):")
print(f"  {USER_PROMPT[:140]}...\n")

# ── Models to probe ──────────────────────────────────────────────────────────
# Skip gpt-5.5 per user's instruction (already known to do hidden reasoning).
OPENROUTER_MODELS = [
    ("GPT-5.4",                "openai/gpt-5.4-20260305"),
    ("GPT-5.4-mini",           "openai/gpt-5.4-mini-20260317"),
    ("GPT-5.4-nano",           "openai/gpt-5.4-nano-20260317"),
    ("Claude-4.7-Opus",        "anthropic/claude-4.7-opus-20260416"),
    ("Claude-4.6-Sonnet",      "anthropic/claude-4.6-sonnet-20260217"),
    ("Gemini-3.1-Pro",         "google/gemini-3.1-pro-preview-20260219"),
    ("Gemini-3.1-Flash-Lite",  "google/gemini-3.1-flash-lite-preview-20260303"),
    ("Grok-4.20",              "x-ai/grok-4.20-20260309"),
]
DEEPSEEK_MODELS = [
    ("DeepSeek-V3-chat", "deepseek-chat"),
    ("DeepSeek-R1",      "deepseek-reasoner"),
]


def to_serialisable(obj):
    """Best-effort dump of an OpenAI-SDK response object to plain dicts."""
    if obj is None: return None
    if isinstance(obj, (bool, int, float, str)): return obj
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(x) for x in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return {k: to_serialisable(v) for k, v in vars(obj).items()
                if not k.startswith("_")}
    return str(obj)


def probe_via_openrouter(disp: str, model: str, ask_for_reasoning: bool) -> dict:
    """Probe a model via OpenRouter.

    Two modes:
      ask_for_reasoning=False  → mirrors evaluate.py exactly, no extras.
                                 Tells us what happens during our actual evals.
      ask_for_reasoning=True   → passes `reasoning: {enabled: True}` to expose
                                 reasoning if the underlying model can reason
                                 (whether on or off by default).
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    suffix = "(ask-for-reasoning)" if ask_for_reasoning else "(default, mirrors eval)"
    print(f"[{disp}] {suffix}  model={model}")
    t0 = time.time()
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT},
        ],
        max_tokens=2048,
    )
    if ask_for_reasoning:
        kwargs["extra_body"] = {"reasoning": {"enabled": True}}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return {"model": model, "provider": "openrouter",
                "ask_for_reasoning": ask_for_reasoning,
                "error": f"{type(e).__name__}: {e}"}
    dt = time.time() - t0
    payload = to_serialisable(resp)
    msg = payload["choices"][0]["message"]
    out = {
        "display":  disp,
        "provider": "openrouter",
        "ask_for_reasoning": ask_for_reasoning,
        "model":    model,
        "elapsed_s": round(dt, 2),
        "final_answer_text":   msg.get("content", ""),
        "reasoning_content":   msg.get("reasoning")
                                or msg.get("reasoning_content")
                                or None,
        "usage":               payload.get("usage"),
        "raw_message_keys":    sorted(msg.keys()),
        "raw_usage_keys":      sorted((payload.get("usage") or {}).keys()),
    }
    return out


def probe_via_deepseek(disp: str, model: str) -> dict:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    print(f"[{disp}]  POST /chat/completions  model={model}")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT},
            ],
            max_tokens=2048,
        )
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return {"model": model, "provider": "deepseek", "error": f"{type(e).__name__}: {e}"}
    dt = time.time() - t0
    payload = to_serialisable(resp)
    msg = payload["choices"][0]["message"]
    out = {
        "display":  disp,
        "provider": "deepseek",
        "model":    model,
        "elapsed_s": round(dt, 2),
        "final_answer_text":  msg.get("content", ""),
        "reasoning_content":  msg.get("reasoning_content"),
        "usage":              payload.get("usage"),
        "raw_message_keys":   sorted(msg.keys()),
        "raw_usage_keys":     sorted((payload.get("usage") or {}).keys()),
    }
    return out


def reasoning_tokens_from_usage(u):
    if not u: return None
    ctd = u.get("completion_tokens_details")
    if isinstance(ctd, dict) and "reasoning_tokens" in ctd:
        return ctd["reasoning_tokens"]
    if "reasoning_tokens" in u:
        return u["reasoning_tokens"]
    return None


def print_summary(rows: list[dict], title: str):
    print(f"\n{title}")
    print("=" * 100)
    hdr = f"{'Model':<22}  {'final[:25]':<25}  {'reason?':<8}  {'reason_tok':<10}  {'in/out':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r.get('display','?'):<22}  ERROR: {r['error'][:60]}")
            continue
        rc = r.get("reasoning_content")
        rc_flag = "YES" if rc else "no"
        u = r.get("usage") or {}
        rt = reasoning_tokens_from_usage(u)
        rt_str = str(rt) if rt is not None else "—"
        in_t  = u.get("prompt_tokens", "?")
        out_t = u.get("completion_tokens", "?")
        final = (r.get("final_answer_text") or "").replace("\n", " ")[:25]
        print(f"{r['display']:<22}  {final:<25}  {rc_flag:<8}  {rt_str:<10}  {in_t}/{out_t}")


def main():
    default_rows = []
    for disp, model in OPENROUTER_MODELS:
        default_rows.append(probe_via_openrouter(disp, model, ask_for_reasoning=False))
        time.sleep(0.4)
    for disp, model in DEEPSEEK_MODELS:
        default_rows.append(probe_via_deepseek(disp, model))
        time.sleep(0.4)

    asked_rows = []
    for disp, model in OPENROUTER_MODELS:
        asked_rows.append(probe_via_openrouter(disp, model, ask_for_reasoning=True))
        time.sleep(0.4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"default": default_rows, "with_reasoning_requested": asked_rows},
        indent=2,
    ))
    print(f"\nWrote {OUT}")

    print_summary(default_rows,
        "DEFAULT calls (no extra_body — mirrors evaluate.py exactly)")
    print_summary(asked_rows,
        "WITH `reasoning: {enabled: True}` requested (capability check)")


if __name__ == "__main__":
    main()
