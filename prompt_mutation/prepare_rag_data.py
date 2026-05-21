"""Step 0: pre-process raw RAG data into a JSONL of RAGExample dicts.

Runs on the LOGIN NODE (needs internet). The output JSONL is then consumed
offline by SLURM jobs via --load-processed-path.

Usage (login node, after activating .venv):
    python -m prompt_mutation.prepare_rag_data \
        --dataset-name LLukas22/nq-simplified \
        --split "train[:200]" \
        --max-samples 200 \
        --min-chunks 3 \
        --max-chunks 4 \
        --output-path $SCRATCH/prefix_caching/processed/rag_examples.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data_loader import DataLoadConfig, load_examples


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
    p.add_argument("--distractor-seed", type=int, default=0)
    p.add_argument("--output-path", required=True,
                   help="Destination JSONL file (one RAGExample dict per line).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = DataLoadConfig(
        workload="rag",
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        split=args.split,
        max_samples=args.max_samples,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        cache_dir=args.cache_dir,
        save_processed_path=str(out_path),
        min_chunks=args.min_chunks,
        max_chunks=args.max_chunks,
        distractor_seed=args.distractor_seed,
    )
    examples = load_examples(cfg)

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
