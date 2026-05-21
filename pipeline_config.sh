#!/bin/bash
# Central pipeline configuration. Sourced by every SLURM script and run_pipeline.sh.
# All values are overridable by exporting them before sourcing this file.
#
# Knobs:
#   SLURM_ACCOUNT   - cluster account
#   PROJECT_ROOT    - repo root on the cluster (must contain .venv/, models/, etc.)
#   SCRATCH_ROOT    - cluster artifact root (mutation, layouts, benchmarks, analysis)
#   Local mirror    - repo outputs/ (see paths.py); sync from $SCRATCH after cluster runs
#   MODEL_PATH      - local model directory used by vLLM (no internet on compute nodes)
#   WORKLOAD        - rag | scientific
#   SEMANTIC_CLASS  - meaning_preserving | meaning_changing
#   GENERATION_CLASS- algorithmic | llm_generated
#   MUTATION_TYPE   - e.g. chunk_reorder, field_reorder, parameter_change, ...
#   STRATEGIES      - space-separated layout names (original stable_first stable_first_normalized volatile_last)
#   CACHE_MODES     - space-separated; subset of "off on"

: "${SLURM_ACCOUNT:=def-mmehride}"

: "${PROJECT_ROOT:=$HOME/work/prefix_caching}"
: "${SCRATCH_ROOT:=$SCRATCH/prefix_caching}"
: "${MODEL_PATH:=$PROJECT_ROOT/models/TinyLlama-1.1B-Chat-v1.0}"

: "${WORKLOAD:=rag}"
: "${SEMANTIC_CLASS:=meaning_preserving}"
: "${GENERATION_CLASS:=algorithmic}"
: "${MUTATION_TYPE:=chunk_reorder}"

: "${STRATEGIES:=original stable_first stable_first_normalized}"
: "${CACHE_MODES:=off on}"

# Derived paths
MUTATION_DIR="$SCRATCH_ROOT/mutation/$WORKLOAD/$SEMANTIC_CLASS/$GENERATION_CLASS/$MUTATION_TYPE"
LAYOUT_DIR="$SCRATCH_ROOT/prompt_organization"
BENCH_DIR="$SCRATCH_ROOT/benchmark_results"
ANALYSIS_DIR="$SCRATCH_ROOT/analysis"
PROCESSED_DIR="$SCRATCH_ROOT/processed"

# Common naming tag, e.g. rag_chunk_reorder
TAG="${WORKLOAD}_${MUTATION_TYPE}"

load_modules() {
  module --force purge
  module load StdEnv/2023
  module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
  module load scipy-stack/2025a
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
}

set_offline_env() {
  export HF_HOME="$PROJECT_ROOT/hf_cache"
  export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf_cache"
  export HF_DATASETS_CACHE="$PROJECT_ROOT/hf_cache/datasets"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
}

# Pick the newest .jsonl in the mutation directory (ignores .summary.json)
discover_mutation_jsonl() {
  local dir="${1:-$MUTATION_DIR}"
  ls -t "$dir"/*.jsonl 2>/dev/null | head -n 1
}

layout_jsonl_path() {
  local strategy="$1"
  echo "$LAYOUT_DIR/${TAG}_${strategy}.jsonl"
}

benchmark_run_name() {
  local strategy="$1"
  local cache_mode="$2"
  echo "${TAG}_${strategy}_cache_${cache_mode}"
}

benchmark_jsonl_path() {
  local strategy="$1"
  local cache_mode="$2"
  echo "$BENCH_DIR/$(benchmark_run_name "$strategy" "$cache_mode").jsonl"
}
