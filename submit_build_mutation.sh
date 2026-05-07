#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=build_mutation
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
module load scipy-stack/2025a

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

source "$PROJECT_ROOT/.venv/bin/activate"

export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p "$SCRATCH_ROOT"/{processed,mutation}

cd "$PROJECT_ROOT"

python prompt_mutation/build_mutation_dataset.py \
  --workload rag \
  --load-processed-dir "$SCRATCH_ROOT/processed/rag_examples.jsonl" \
  --semantic-class meaning_preserving \
  --generation-class algorithmic \
  --mutation-type chunk_reorder \
  --validation-backend sentence_transformer \
  --severity-backend sentence_transformer \
  --output-root "$SCRATCH_ROOT/mutation"