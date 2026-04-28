
# prompt_organization

This module implements step 4 prompt-organization baselines.

## Goal

Generate alternative layouts that:
- keep stable content early
- move volatile content later
- normalize formatting
- preserve chunk order when possible

## Files

- `layout_strategies.py` — layout strategy implementations
- `apply_layouts.py` — CLI to rewrite an existing mutation JSONL using a chosen layout strategy

## Strategies

- `original`
- `stable_first`
- `stable_first_normalized`
- `volatile_last`
