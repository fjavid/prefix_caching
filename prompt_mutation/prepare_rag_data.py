"""Step 0: pre-process raw RAG data into a JSONL of RAGExample dicts.

Runs on the LOGIN NODE (needs internet). The output JSONL is then consumed
offline by SLURM jobs via --load-processed-path.

Two safety mechanisms keep prompts under the model's context window:
1. --max-chunk-words: per-chunk word cap applied in data_loader.
2. --max-prompt-tokens + --tokenizer-path: after distractor padding, render each
   prompt and drop any whose tokenization exceeds the budget. Use this to match
   the model's actual context (e.g. 2048 for TinyLlama-1.1B).

Usage (login node, after activating .venv):
    python -m prompt_mutation.prepare_rag_data \\
        --dataset-name LLukas22/nq-simplified \\
        --split "train[:200]" \\
        --max-samples 200 \\
        --min-chunks 3 \\
        --max-chunks 4 \\
        --max-chunk-words 200 \\
        --tokenizer-path $HOME/work/prefix_caching/models/TinyLlama-1.1B-Chat-v1.0 \\
        --max-prompt-tokens 1800 \\
        --output-path $SCRATCH/prefix_caching/processed/rag_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .data_loader import DataLoadConfig, load_examples
from .prompt_generator import RAGExample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-name", default="LLukas22/nq-simplified")
    p.add_argument("--dataset-config-name", default=None)
    p.add_argument("--split", default="train[:200]")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--cache-dir", default=None,
                   help="HF datasets cache_dir; defaults to $HF_DATASETS_CACHE.")
    p.add_argument("--min-chunks", type=int, default=3,
                   help="Minimum retrieved_chunks per example (pads with distractors).")
    p.add_argument("--max-chunks", type=int, default=4)
    p.add_argument("--max-chunk-words", type=int, default=200,
                   help="Per-chunk word cap. Prevents a single Wikipedia paragraph "
                        "from filling the context window. 0 disables truncation.")
    p.add_argument("--distractor-seed", type=int, default=0)
    p.add_argument("--tokenizer-path", default=None,
                   help="Path or HF id of the tokenizer used to enforce "
                        "--max-prompt-tokens. If unset, the token filter is skipped.")
    p.add_argument("--max-prompt-tokens", type=int, default=1800,
                   help="Drop any rendered prompt that tokenizes to more than this "
                        "many tokens. Leave headroom for max_new_tokens. 0 disables.")
    p.add_argument("--output-path", required=True,
                   help="Destination JSONL file (one RAGExample dict per line).")
    return p.parse_args()


def _filter_by_token_budget(
    examples: List[RAGExample],
    tokenizer_path: str,
    max_prompt_tokens: int,
) -> List[RAGExample]:
    """Drop any example whose rendered prompt exceeds max_prompt_tokens.

    The tokenizer must match the model used at benchmark time, otherwise the
    budget is a guess. We load the tokenizer with trust_remote_code=True so
    non-standard model configs still work.
    """
    if max_prompt_tokens <= 0 or not tokenizer_path:
        return examples

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    kept: List[RAGExample] = []
    dropped_lengths: List[int] = []
    kept_lengths: List[int] = []
    for ex in examples:
        n = len(tok(ex.render()).input_ids)
        if n <= max_prompt_tokens:
            kept.append(ex)
            kept_lengths.append(n)
        else:
            dropped_lengths.append(n)

    total = len(examples)
    print(
        f"Token-budget filter (limit={max_prompt_tokens}, tokenizer={tokenizer_path}):\n"
        f"  kept    : {len(kept)} / {total}\n"
        f"  dropped : {len(dropped_lengths)} / {total}"
    )
    if kept_lengths:
        print(
            f"  kept tokens   : min={min(kept_lengths)} max={max(kept_lengths)} "
            f"mean={sum(kept_lengths)/len(kept_lengths):.1f}"
        )
    if dropped_lengths:
        print(
            f"  dropped tokens: min={min(dropped_lengths)} max={max(dropped_lengths)} "
            f"mean={sum(dropped_lengths)/len(dropped_lengths):.1f}"
        )
    return kept


def _save_jsonl(examples: List[RAGExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)

    # Note: we deliberately do NOT pass save_processed_path here. We want to
    # apply the token-budget filter BEFORE saving so the JSONL only contains
    # prompts that will actually run through the model.
    cfg = DataLoadConfig(
        workload="rag",
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        max_samples=args.max_samples,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        cache_dir=args.cache_dir,
        min_chunks=args.min_chunks,
        max_chunks=args.max_chunks,
        max_chunk_words=args.max_chunk_words,
        distractor_seed=args.distractor_seed,
    )
    examples = load_examples(cfg)
    print(f"Loaded {len(examples)} RAG examples after chunk extraction and distractor padding.")

    examples = _filter_by_token_budget(
        examples,
        tokenizer_path=args.tokenizer_path,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    _save_jsonl(examples, out_path)

    chunk_counts = [len(ex.retrieved_chunks) for ex in examples]
    print(f"Saved {len(examples)} RAG examples to: {out_path}")
    if chunk_counts:
        print(
            f"retrieved_chunks per example: "
            f"min={min(chunk_counts)} max={max(chunk_counts)} "
            f"mean={sum(chunk_counts)/len(chunk_counts):.2f}"
        )


if __name__ == "__main__":
    main()
