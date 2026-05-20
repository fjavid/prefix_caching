#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=benchmark
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
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

echo "Benchmark grid:"
echo "  strategies  = $RUN_STRATEGIES"
echo "  cache_modes = $RUN_CACHE_MODES"
echo "  model       = $MODEL_PATH"

for strategy in $RUN_STRATEGIES; do
  input_jsonl="$(layout_jsonl_path "$strategy")"
  if [[ ! -f "$input_jsonl" ]]; then
    echo "ERROR: layout JSONL not found: $input_jsonl" >&2
    exit 1
  fi
  for cache_mode in $RUN_CACHE_MODES; do
    case "$cache_mode" in
      on)  cache_flag="--enable-prefix-caching" ;;
      off) cache_flag="--disable-prefix-caching" ;;
      *)   echo "ERROR: invalid cache_mode='$cache_mode'" >&2; exit 1 ;;
    esac
    run_name="$(benchmark_run_name "$strategy" "$cache_mode")"
    echo "=== strategy=$strategy cache=$cache_mode -> $run_name ==="
    python -m inference_benchmark.benchmark_prefix_cache \
      --backend-name vllm \
      --model-name "$MODEL_PATH" \
      $cache_flag \
      --mutation-jsonl-path "$input_jsonl" \
      --output-dir "$BENCH_DIR" \
      --run-name "$run_name"
  done
done
