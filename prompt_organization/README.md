
# prompt_organization

This module implements step 4 prompt-organization baselines.

## Goal

Generate alternative layouts that:
- keep stable content (the sections the mutation does NOT touch) early in the prompt
- move volatile content (the section the mutation changes) to the end

The intent is to push the first divergence between the base and mutated
prompts as late as possible, so vLLM's prefix cache can reuse more KV blocks.

## Files

- `layout_strategies.py` — layout strategy implementations
- `apply_layouts.py` — CLI to rewrite an existing mutation JSONL using a chosen layout strategy

## Strategies

- `original` — leaves the prompt as the mutation step produced it. Baseline.
- `stable_first` — mutation-aware. Identifies the volatile section for the
  current mutation type (e.g. `retrieved_chunks` for `chunk_reorder`,
  `user_query` for `typo`) and moves it to the end of the prompt; all other
  sections stay in their canonical order at the front. Does NOT alter the
  content of any section.
- `volatile_last` — kept as a backward-compat alias of `stable_first`.
