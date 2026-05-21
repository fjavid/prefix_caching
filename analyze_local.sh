#!/bin/bash
# Run the analysis stage WITHOUT SLURM (login node or MacBook).
# The analysis is pure pandas/numpy/matplotlib; no GPU, no LLM, no internet.
#
# It expects the benchmark JSONLs to be reachable from this machine. On
# Alliance Canada this works on the login node directly. On a MacBook, rsync
# the results first, e.g.:
#   rsync -av narval:/scratch/$USER/prefix_caching/benchmark_results outputs/
#
# Overridable env vars (default values from pipeline_config.sh):
#   TAG, STRATEGIES, CACHE_MODES, METRIC
#   RESULTS_ROOT  (overrides SCRATCH_ROOT/benchmark_results; defaults to
#                  outputs/benchmark_results if SCRATCH is unset)
#   ANALYSIS_DIR  (defaults to outputs/analysis)
#
# Usage:
#   bash analyze_local.sh
#   METRIC=ttft_gain_seconds bash analyze_local.sh
#   RESULTS_ROOT=outputs/benchmark_results ANALYSIS_DIR=outputs/analysis bash analyze_local.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Allow running off-cluster by stubbing SLURM-only env vars.
: "${SCRATCH:=$SCRIPT_DIR/outputs}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

# Override directories for local runs that don't have $SCRATCH/prefix_caching.
RESULTS_ROOT="${RESULTS_ROOT:-$SCRATCH_ROOT/benchmark_results}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$SCRATCH_ROOT/analysis}"
METRIC="${METRIC:-latency_gain_seconds}"

if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "ERROR: RESULTS_ROOT not found: $RESULTS_ROOT" >&2
  echo "Set RESULTS_ROOT=<path to benchmark_results/> and retry." >&2
  exit 1
fi

mkdir -p "$ANALYSIS_DIR"

INPUT_PATHS=()
MISSING=()
for strategy in $STRATEGIES; do
  for mode in $CACHE_MODES; do
    p="$RESULTS_ROOT/${TAG}_${strategy}_${mode}.jsonl"
    if [[ -f "$p" ]]; then
      INPUT_PATHS+=("$p")
    else
      MISSING+=("$p")
    fi
  done
done

if [[ ${#INPUT_PATHS[@]} -eq 0 ]]; then
  echo "ERROR: no benchmark JSONLs found under $RESULTS_ROOT for TAG=$TAG" >&2
  printf '  missing: %s\n' "${MISSING[@]}" >&2
  exit 1
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "WARNING: skipping ${#MISSING[@]} missing file(s):"
  printf '  %s\n' "${MISSING[@]}"
fi

PLOTS_DIR="$ANALYSIS_DIR/plots_${TAG}"
mkdir -p "$PLOTS_DIR"

echo "Analyzing ${#INPUT_PATHS[@]} result files with metric=$METRIC"
python -m analysis.analyze_prefix_cache \
  --input-paths "${INPUT_PATHS[@]}" \
  --output-dir "$ANALYSIS_DIR" \
  --prefix "$TAG" \
  --metric "$METRIC"

python -m analysis.plot_prefix_cache_results \
  --merged-csv "$ANALYSIS_DIR/${TAG}.merged.csv" \
  --output-dir "$PLOTS_DIR" \
  --metric "$METRIC"

echo "Done."
echo "  summary : $ANALYSIS_DIR/${TAG}.summary.json"
echo "  merged  : $ANALYSIS_DIR/${TAG}.merged.csv"
echo "  plots   : $PLOTS_DIR/  (subdir: cross_category/)"
