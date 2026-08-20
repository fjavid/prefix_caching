#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=benchmark
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# The --time, --mem, and --gres directives above are the TinyLlama-1.1B
# fallback, used only when this script is submitted directly with bare
# `sbatch submit_interface_benchmark.sh`. #SBATCH directives are static comments
# and cannot read the MODEL_TAG registry, so run_pipeline.sh passes the
# per-model values from pipeline_config.sh (SBATCH_MEM / SBATCH_TIME /
# SBATCH_GRES) as sbatch command-line options, which take precedence.
#
# Submitting this script directly for an 8B model will therefore under-request
# memory and time. Go through run_pipeline.sh, or pass the overrides yourself.
# Time budget rationale (per mutation chain, N=1000 records):
#   ~12000 inference calls = N * 3 categories * 2 layouts * 2 cache modes
#   * ~50 ms TTFT per call = ~10 min pure compute
#   + 2 engine startups (one per cache mode) * ~3 min = ~6 min
#   + per-case prefix-cache reset overhead = ~10-15 min
# Total ~30-45 min in the happy path; 2.5 h gives ~3-4x headroom for I/O
# and scheduling slack. Bump if you scale N or strategies.

set -euo pipefail

# Under sbatch, the script is run from a spool dir, so BASH_SOURCE is unreliable.
# Prefer PIPELINE_DIR (set by run_pipeline.sh), then SLURM_SUBMIT_DIR, then BASH_SOURCE.
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

load_modules
activate_venv
set_offline_env

mkdir -p "$BENCH_DIR"
cd "$PROJECT_ROOT"

# Optional single-config override (used when running a single sbatch outside the master pipeline).
# If STRATEGY and CACHE_MODE are both set, run only that one; otherwise loop over the full grid.
if [[ -n "${STRATEGY:-}" && -n "${CACHE_MODE:-}" ]]; then
  RUN_STRATEGIES="$STRATEGY"
  RUN_CACHE_MODES="$CACHE_MODE"
else
  RUN_STRATEGIES="$STRATEGIES"
  RUN_CACHE_MODES="$CACHE_MODES"
fi

# TTFT path: defaults to the async (AsyncLLMEngine) backend. Set USE_ASYNC_TTFT=0
# to fall back to the simpler offline LLM.generate path (TTFT will be None).
USE_ASYNC_TTFT="${USE_ASYNC_TTFT:-1}"
if [[ "$USE_ASYNC_TTFT" -eq 1 ]]; then
  async_flag="--use-async-ttft"
else
  async_flag="--no-async-ttft"
fi

# Reset the prefix KV cache between cases so each (base, followup) pair starts
# clean. Required for a meaningful unrelated_control noise floor. Set
# RESET_CACHE_BETWEEN_CASES=0 only to reproduce the older contaminated runs.
RESET_CACHE_BETWEEN_CASES="${RESET_CACHE_BETWEEN_CASES:-1}"
if [[ "$RESET_CACHE_BETWEEN_CASES" -eq 1 ]]; then
  reset_flag="--reset-cache-between-cases"
else
  reset_flag="--no-reset-cache-between-cases"
fi

# Validate that every (strategy) layout JSONL exists before we boot any engine.
layout_jsonls=()
for strategy in $RUN_STRATEGIES; do
  p="$(layout_jsonl_path "$strategy")"
  if [[ ! -f "$p" ]]; then
    echo "ERROR: layout JSONL not found: $p" >&2
    exit 1
  fi
  layout_jsonls+=("$p")
done

echo "Benchmark grid:"
echo "  model_tag                   = $MODEL_TAG"
echo "  strategies                  = $RUN_STRATEGIES"
echo "  cache_modes                 = $RUN_CACHE_MODES"
echo "  model                       = $MODEL_PATH"
echo "  max_model_len               = $MAX_MODEL_LEN"
echo "  use_async_ttft              = $USE_ASYNC_TTFT"
echo "  reset_cache_between_cases   = $RESET_CACHE_BETWEEN_CASES"
# Resource request actually granted, which may differ from the static #SBATCH
# directives above when run_pipeline.sh supplied command-line overrides.
echo "  granted mem / time / gres   = ${SLURM_MEM_PER_NODE:-?} / ${SLURM_JOB_ID:+see scontrol} / ${SLURM_JOB_GPUS:-?}"

# ONE engine per cache_mode (vLLM's enable_prefix_caching is set at engine
# construction time and can't be toggled at runtime, so we need a separate
# engine for cache_on vs cache_off). Inside each engine we sweep every
# layout back-to-back; this removes the per-layout engine-startup variance
# that contaminated earlier runs.
for cache_mode in $RUN_CACHE_MODES; do
  case "$cache_mode" in
    on)  cache_flag="--enable-prefix-caching" ;;
    off) cache_flag="--disable-prefix-caching" ;;
    *)   echo "ERROR: invalid cache_mode='$cache_mode'" >&2; exit 1 ;;
  esac

  run_name_template="${TAG}_{layout}_cache_${cache_mode}"
  echo "=== cache=$cache_mode  layouts=[$RUN_STRATEGIES]  template=$run_name_template ==="

  # Only pass --max-cases when MAX_CASES is non-empty; the flag's own default of
  # None means "all cases", and passing an empty string would be a parse error.
  max_cases_flag=()
  if [[ -n "${MAX_CASES:-}" ]]; then
    max_cases_flag=(--max-cases "$MAX_CASES")
  fi

  python -m inference_benchmark.benchmark_prefix_cache \
    --backend-name vllm \
    --model-name "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    ${max_cases_flag[@]+"${max_cases_flag[@]}"} \
    $cache_flag \
    $async_flag \
    $reset_flag \
    --layouts $RUN_STRATEGIES \
    --mutation-jsonls "${layout_jsonls[@]}" \
    --output-dir "$BENCH_DIR" \
    --run-name-template "$run_name_template"
done
