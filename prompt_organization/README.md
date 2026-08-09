
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

There is currently **one** real layout strategy. `get_layout_strategy` accepts
three names, but only `stable_first` reorganizes anything:

- `original` — leaves the prompt as the mutation step produced it. Baseline,
  not a strategy.
- `stable_first` — the only real strategy. Identifies the volatile section for
  the current mutation type (e.g. `retrieved_chunks` for `chunk_reorder`,
  `user_query` for `typo`) and moves it to the end of the prompt; all other
  sections stay in their canonical order at the front. Does NOT alter the
  content of any section.
- `volatile_last` — a backward-compat **alias** of `stable_first`, kept for old
  configs. It returns the identical prompt text and differs only in the
  `strategy_name` field. Running both is not a two-strategy comparison.

### Limitations of `stable_first`

- **It is an oracle.** The volatile section is looked up from the ground-truth
  `mutation_type` (`RAG_VOLATILE_SECTION` / `SCIENTIFIC_VOLATILE_SECTION`). A
  deployed serving system does not know in advance which section a future
  request will change, so measured gains are an upper bound on what a practical
  layout policy could achieve.
- **It only reorders whole sections; it never edits content.** So it cannot
  help when the mutation permutes *within* the volatile section. This is why
  `chunk_reorder` is not recoverable here: the retrieved chunks are the bulk of
  the prompt, and moving that block does not lengthen the shared prefix. A
  content-normalizing strategy (canonical chunk order, whitespace/format
  normalization) would be needed instead.
- **Recovery scales with stable mass.** When the volatile section is small
  (typo, synonym substitution) the strategy recovers essentially the full
  exact-reuse ceiling; when the volatile section is the payload it recovers
  nothing. See `research.md` for the measured asymmetry.
