#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=benchmark_cache_off
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
module load scipy-stack/2025a

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

INPUT_JSONL="$SCRATCH_ROOT/prompt_organization/rag_chunk_reorder_original.jsonl"

python -m inference_benchmark.benchmark_prefix_cache \
  --backend-name vllm \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --disable-prefix-caching \
  --mutation-jsonl-path "$INPUT_JSONL" \
  --output-dir "$SCRATCH_ROOT/benchmark_results" \
  --run-name rag_chunk_reorder_original_cache_off