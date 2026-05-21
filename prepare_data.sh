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
SPLIT="${SPLIT:-train[:200]}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MIN_CHUNKS="${MIN_CHUNKS:-3}"
MAX_CHUNKS="${MAX_CHUNKS:-4}"

mkdir -p "$PROCESSED_DIR"
cd "$PROJECT_ROOT"

OUTPUT_PATH="$PROCESSED_DIR/${WORKLOAD}_examples.jsonl"

echo "Preparing $WORKLOAD data:"
echo "  dataset=$DATASET_NAME  split=$SPLIT  max_samples=$MAX_SAMPLES"
echo "  min_chunks=$MIN_CHUNKS  max_chunks=$MAX_CHUNKS"
echo "  output=$OUTPUT_PATH"

python -m prompt_mutation.prepare_rag_data \
  --dataset-name "$DATASET_NAME" \
  --split "$SPLIT" \
  --max-samples "$MAX_SAMPLES" \
  --min-chunks "$MIN_CHUNKS" \
  --max-chunks "$MAX_CHUNKS" \
  --output-path "$OUTPUT_PATH"

echo "Done. Run sbatch jobs next (e.g. ./run_pipeline.sh)."
