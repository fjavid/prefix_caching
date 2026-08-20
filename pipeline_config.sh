#!/bin/bash
# Central pipeline configuration. Sourced by every SLURM script and run_pipeline.sh.
# All values are overridable by exporting them before sourcing this file.
#
# Knobs:
#   SLURM_ACCOUNT   - cluster account
#   PROJECT_ROOT    - repo root on the cluster (must contain .venv/, models/, etc.)
#   SCRATCH_ROOT    - cluster artifact root (mutation, layouts, benchmarks, analysis)
#   Local mirror    - repo outputs/ (see paths.py); sync from $SCRATCH after cluster runs
#   MODEL_TAG       - selects the model from the registry below. Also namespaces
#                     every $SCRATCH_ROOT artifact directory, so two models never
#                     overwrite each other's results.
#   MODEL_PATH      - local model directory used by vLLM (no internet on compute
#                     nodes). Defaults to $PROJECT_ROOT/models/$MODEL_TAG; an
#                     explicit export still wins.
#   WORKLOAD        - rag | scientific
#   SEMANTIC_CLASS  - meaning_preserving | meaning_changing
#   GENERATION_CLASS- algorithmic | llm_generated
#   MUTATION_TYPE   - e.g. chunk_reorder, field_reorder, parameter_change, ...
#   STRATEGIES      - space-separated layout names (original stable_first volatile_last)
#   CACHE_MODES     - space-separated; subset of "off on"
#   MAX_CASES       - cap benchmark cases per run; empty means all. Use for the
#                     pilot that measures per-case cost before a full sweep.
#
# Login-node-only knobs (consumed by prepare_data.sh):
#   MAX_CHUNK_WORDS   - per-chunk word cap during extraction (default 200)
#   TOKENIZER_PATH    - tokenizer used for the prompt-budget filter (default $MODEL_PATH)
#
# Model registry fields (resolved from MODEL_TAG; each is overridable):
#   MODEL_REPO        - Hugging Face repo id, used by prep_login.sh
#   MAX_MODEL_LEN     - engine context cap passed to vLLM
#   MAX_PROMPT_TOKENS - prompt-token budget enforced by prepare_data.sh
#   N_PARAMS          - parameter count for analysis/plot_report.py roofline
#   GATED             - 1 if the HF repo requires license acceptance + HF_TOKEN
#   SBATCH_MEM        - host memory for the GPU benchmark job
#   SBATCH_TIME       - wall-clock limit for the GPU benchmark job
#   SBATCH_GRES       - GPU request for the benchmark job
#
# The SBATCH_* values are passed as sbatch COMMAND-LINE overrides by
# run_pipeline.sh, because #SBATCH directives inside a submit script are static
# comments and cannot read this registry. Command-line options win over
# directives, so the directives in submit_interface_benchmark.sh remain the
# fallback for a bare `sbatch submit_interface_benchmark.sh`.

: "${SLURM_ACCOUNT:=def-mmehride}"

: "${PROJECT_ROOT:=$HOME/work/prefix_caching}"
: "${SCRATCH_ROOT:=$SCRATCH/prefix_caching}"

# ---------------------------------------------------------------------------
# Model registry.
#
# MODEL_TAG is the single knob that selects a model. It resolves to the local
# weights directory, the tokenizer used for the prompt-token budget, the engine
# context cap, and the $SCRATCH_ROOT artifact namespace.
#
# MAX_MODEL_LEN is deliberately NOT the model's declared context window. The
# required window is MAX_PROMPT_TOKENS + chat-template overhead + max_new_tokens
# = 1800 + 15 + 64 = 1879, where 15 is the measured full-template overhead for
# TinyLlama-1.1B (leading role markers plus the trailing </s> and assistant
# marker). The longest templated prompt observed was 1554 tokens. Llama-3.1
# declares 131072; without the cap vLLM sizes the KV cache for the full window
# and fails allocation on a 40 GB device.
#
# To add a model: add a case arm and stage the weights with
#   MODEL_TAG=<tag> bash prep_login.sh
# ---------------------------------------------------------------------------
: "${MODEL_TAG:=TinyLlama-1.1B-Chat-v1.0}"

case "$MODEL_TAG" in
  TinyLlama-1.1B-Chat-v1.0)
    : "${MODEL_REPO:=TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
    : "${MAX_MODEL_LEN:=2048}"
    : "${MAX_PROMPT_TOKENS:=1800}"
    : "${N_PARAMS:=1.1e9}"
    : "${GATED:=0}"
    # 2.2 GB of weights; the stock SBATCH directives are already sized for this.
    : "${SBATCH_MEM:=32G}"
    : "${SBATCH_TIME:=02:30:00}"
    : "${SBATCH_GRES:=gpu:1}"
    ;;
  Llama-3.1-8B-Instruct)
    : "${MODEL_REPO:=meta-llama/Llama-3.1-8B-Instruct}"
    : "${MAX_MODEL_LEN:=2048}"
    : "${MAX_PROMPT_TOKENS:=1800}"
    : "${N_PARAMS:=8.0e9}"
    : "${GATED:=1}"
    # ~16 GB of weights in bf16, so host memory for loading roughly doubles and
    # the device needs >= 40 GB to hold weights plus KV cache at
    # gpu_memory_utilization=0.85.
    : "${SBATCH_MEM:=64G}"
    # PROVISIONAL. Estimated from the v5 TinyLlama timings scaled ~7x: the worst
    # mutation type needs ~4.4 h at 3000 cases, which the 02:30:00 default would
    # kill mid-run. Replace with the measured value from the pilot (Batch 8 of
    # TASKS/model-scale-8b.md) before the full sweep.
    : "${SBATCH_TIME:=06:00:00}"
    # A plain gpu:1 request may land on a device smaller than 40 GB depending on
    # the cluster and partition. Override with an explicit type when needed,
    # e.g. SBATCH_GRES=gpu:a100:1.
    : "${SBATCH_GRES:=gpu:1}"
    ;;
  *)
    echo "ERROR: unknown MODEL_TAG='$MODEL_TAG'." >&2
    echo "Valid tags:" >&2
    echo "  TinyLlama-1.1B-Chat-v1.0" >&2
    echo "  Llama-3.1-8B-Instruct" >&2
    echo "Add a case arm in pipeline_config.sh to register a new model." >&2
    exit 1
    ;;
esac

: "${MODEL_PATH:=$PROJECT_ROOT/models/$MODEL_TAG}"

: "${WORKLOAD:=rag}"
: "${SEMANTIC_CLASS:=meaning_preserving}"
: "${GENERATION_CLASS:=algorithmic}"
: "${MUTATION_TYPE:=chunk_reorder}"

# Cap the number of benchmark cases per run. Unset means all of them. Required
# for the pilot: a small run measures per-case cost without committing to a full
# sweep, and without rewriting the shared processed example set.
: "${MAX_CASES:=}"

: "${STRATEGIES:=original stable_first}"
: "${CACHE_MODES:=off on}"

# Derived paths.
#
# Every artifact directory is namespaced by MODEL_TAG. Without this, artifact
# names carry only TAG=${WORKLOAD}_${MUTATION_TYPE}, so running a second model
# would silently overwrite the first model's benchmark JSONLs. Each model also
# gets its own processed/ example set, because each filters the prompt-token
# budget with its own tokenizer.
MODEL_SCRATCH_ROOT="$SCRATCH_ROOT/$MODEL_TAG"
MUTATION_ROOT="$MODEL_SCRATCH_ROOT/mutation"
MUTATION_DIR="$MUTATION_ROOT/$WORKLOAD/$SEMANTIC_CLASS/$GENERATION_CLASS/$MUTATION_TYPE"
LAYOUT_DIR="$MODEL_SCRATCH_ROOT/prompt_organization"
BENCH_DIR="$MODEL_SCRATCH_ROOT/benchmark_results"
ANALYSIS_DIR="$MODEL_SCRATCH_ROOT/analysis"
PROCESSED_DIR="$MODEL_SCRATCH_ROOT/processed"

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
  # Quiet vLLM's per-request logger so the benchmark .out file stays readable.
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
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
