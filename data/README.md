# Data

The `data/` and `results/` directories are **not** checked into this repository.
The datasets are distributed (gated) on the HuggingFace Hub:

| Dataset | What it is | Link |
|---|---|---|
| **VETO** (prompt pairs) | 2,032 BBQ-derived contrastive prompt pairs across 8 demographic categories, plus the priming-trigger variant. | `MichiganNLP/misfired-alignment` |
| **Raw evaluation results** | Per-model, per-condition raw model outputs used to compute MAR and the paper's tables/figures. | `MichiganNLP/misfired-alignment-eval-results` |

> ⚠️ Access is **gated** because the data contains sensitive stereotype-related
> content. Please read `../NOTICE.md` (responsible use) before requesting access.

## Building the prompt pairs locally

You can regenerate the prompt pairs directly from BBQ (no download needed):

```bash
python scripts/build_pairs_from_bbq.py
# -> data/prompt_pairs_bbq.json  (+ the trigger variant)
```

## Downloading from the Hub

```python
from datasets import load_dataset
ds = load_dataset("MichiganNLP/misfired-alignment")   # requires accepting the gated terms
```

The human-annotation data used for the human baseline is **not released** to
protect annotator privacy.
