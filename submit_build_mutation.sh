#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module load python/3.11

PROJECT_ROOT="$PROJECT/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

source "$PROJECT_ROOT/env/bin/activate"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"

cd "$PROJECT_ROOT/repo_root"

python prompt_mutation/build_mutation_dataset.py \
  --workload rag \
  --dataset-name LLukas22/nq-simplified \
  --split train \
  --max-samples 50 \
  --semantic-class meaning_preserving \
  --generation-class algorithmic \
  --mutation-type chunk_reorder \
  --validation-backend sentence_transformer \
  --severity-backend sentence_transformer \
  --save-processed-dir "$SCRATCH_ROOT/processed/rag_examples.jsonl" \
  --cache-dir "$PROJECT_ROOT/hf_cache" \
  --output-root "$SCRATCH_ROOT/mutation"