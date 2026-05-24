#!/bin/bash
# Pre-download and pre-process raw data on the LOGIN NODE (compute nodes have no internet).
# Run this once after prep_login.sh, before submitting any SLURM job that needs the data.
#
# Idempotent: re-running just rewrites the processed JSONL.
#
# Overridable env vars:
#   DATASET_NAME, SPLIT, MAX_SAMPLES, MIN_CHUNKS, MAX_CHUNKS
#   PROJECT_ROOT, SCRATCH_ROOT
#
# Usage (login node):
#   bash prepare_data.sh
#   DATASET_NAME=hotpot_qa SPLIT='train[:500]' MAX_SAMPLES=500 bash prepare_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

# Use the modules + venv but not the offline env: we need network here.
load_modules
activate_venv

export HF_HOME="$PROJECT_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE || true

DATASET_NAME="${DATASET_NAME:-LLukas22/nq-simplified}"
# Default sample budget is 1000 to give CIs that visibly tighten over the previous
# 200-sample runs (CIs scale as 1/sqrt(N), so 5x more data ~= 2.2x tighter intervals).
# The token-budget filter (--max-prompt-tokens) and synonym_substitution's ~45%
# survival rate mean the saved-record count is typically lower than this raw budget.
SPLIT="${SPLIT:-train[:1000]}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MIN_CHUNKS="${MIN_CHUNKS:-3}"
MAX_CHUNKS="${MAX_CHUNKS:-4}"
# Context-budget controls. Defaults are sized for TinyLlama-1.1B (2048-token window)
# with max_new_tokens=64 and a small margin for layout-strategy header overhead.
MAX_CHUNK_WORDS="${MAX_CHUNK_WORDS:-200}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1800}"

mkdir -p "$PROCESSED_DIR"
cd "$PROJECT_ROOT"

OUTPUT_PATH="$PROCESSED_DIR/${WORKLOAD}_examples.jsonl"

echo "Preparing $WORKLOAD data:"
echo "  dataset=$DATASET_NAME  split=$SPLIT  max_samples=$MAX_SAMPLES"
echo "  min_chunks=$MIN_CHUNKS  max_chunks=$MAX_CHUNKS  max_chunk_words=$MAX_CHUNK_WORDS"
echo "  tokenizer=$TOKENIZER_PATH  max_prompt_tokens=$MAX_PROMPT_TOKENS"
echo "  output=$OUTPUT_PATH"

python -m prompt_mutation.prepare_rag_data \
  --dataset-name "$DATASET_NAME" \
  --split "$SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --min-chunks "$MIN_CHUNKS" \
  --max-chunks "$MAX_CHUNKS" \
  --max-chunk-words "$MAX_CHUNK_WORDS" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --output-path "$OUTPUT_PATH"

echo "Done. Run sbatch jobs next (e.g. ./run_pipeline.sh)."
