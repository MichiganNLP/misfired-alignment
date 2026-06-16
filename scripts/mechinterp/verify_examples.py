import sys
sys.path.insert(0, "scripts/mechinterp")
from utils import load_model, build_prompt_pairs, get_yes_no_ids, get_answer, get_decoder_handles
import torch

model, tokenizer = load_model()
pairs = build_prompt_pairs(tokenizer)
yes_ids, no_ids = get_yes_no_ids(tokenizer)
in_dev = get_decoder_handles(model).embed.weight.device

all_ok = True
for p in pairs:
    s_in = tokenizer(p.stereotyped_prompt, return_tensors="pt").to(in_dev)
    c_in = tokenizer(p.contrast_prompt, return_tensors="pt").to(in_dev)
    with torch.no_grad():
        ans_s = get_answer(model(**s_in).logits, yes_ids, no_ids)
        ans_c = get_answer(model(**c_in).logits, yes_ids, no_ids)
    ok = ans_s == "no" and ans_c == "yes"
    tag = "[CONFIRMED FAIL]" if ok else f"[unexpected: stereo={ans_s} contrast={ans_c}]"
    print(f"  {p.id}: {tag}")
    if not ok:
        all_ok = False

print()
print("All confirmed!" if all_ok else "Some examples failed — update ANALYSIS_EXAMPLES.")
