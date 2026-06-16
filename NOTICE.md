# Responsible use & data provenance

## Content warning

This repository studies **stereotypes and social biases** in large language
models. The associated **VETO** benchmark contains prompts that reference
historically stereotyped demographic groups and, for measurement purposes,
includes potentially disturbing content. These examples exist **only** to
quantify and localize a failure mode (*misfired alignment*) — they are not
endorsements of any stereotype.

Our findings should **not** be read as an argument against alignment. The point
is the opposite: alignment is essential, and this work highlights the need for
alignment that preserves contextual, evidence-grounded reasoning.

## Intended use

This code and the VETO benchmark are released **for research and educational
purposes** — measuring, analyzing, and mitigating misfired alignment in LLMs.

## Prohibited / out-of-scope use

Do **not** use these materials to:

- train, fine-tune, or prompt systems to produce discriminatory outputs or to
  reinforce stereotypes;
- target, profile, or make consequential decisions about real individuals or
  groups;
- present the contrastive prompts as factual claims about any group.

## Data provenance & licensing

The VETO benchmark is **derived from BBQ** (Bias Benchmark for QA,
Parrish et al., 2022), which is distributed under **CC BY 4.0**. VETO inherits
that license for its data, and BBQ must be cited in any downstream use. The
**code** in this repository is released under the MIT License (see `LICENSE`).

> Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J.,
> Htut, P. M., & Bowman, S. R. (2022). *BBQ: A Hand-Built Bias Benchmark for
> Question Answering.* Findings of ACL 2022.

The full dataset is distributed (gated) on the HuggingFace Hub rather than in
this repository; see `data/README.md`.
