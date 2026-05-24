#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=analyze
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

# Under sbatch, the script is run from a spool dir, so BASH_SOURCE is unreliable.
# Prefer PIPELINE_DIR (set by run_pipeline.sh), then SLURM_SUBMIT_DIR, then BASH_SOURCE.
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

load_modules
activate_venv

mkdir -p "$ANALYSIS_DIR"
cd "$PROJECT_ROOT"

# Cross-strategy combined analysis.
# Loads every benchmark JSONL produced for $TAG and produces:
#   $ANALYSIS_DIR/${TAG}.flat.csv
#   $ANALYSIS_DIR/${TAG}.merged.csv
#   $ANALYSIS_DIR/${TAG}.summary.json
#   $ANALYSIS_DIR/plots_${TAG}/*.png

PREFIX="$TAG"
PLOTS_DIR="$ANALYSIS_DIR/plots_${PREFIX}"
mkdir -p "$PLOTS_DIR"

INPUT_PATHS=()
for strategy in $STRATEGIES; do
  for mode in $CACHE_MODES; do
    p="$(benchmark_jsonl_path "$strategy" "$mode")"
    if [[ ! -f "$p" ]]; then
      echo "ERROR: missing benchmark result: $p" >&2
      exit 1
    fi
    INPUT_PATHS+=("$p")
  done
done

METRIC="${METRIC:-wall_clock_gain_seconds}"

echo "Analyzing $((${#INPUT_PATHS[@]})) result files with metric=$METRIC"

python -m analysis.analyze_prefix_cache \
  --input-paths "${INPUT_PATHS[@]}" \
  --output-dir "$ANALYSIS_DIR" \
  --prefix "$PREFIX" \
  --metric "$METRIC"

python -m analysis.plot_prefix_cache_results \
  --merged-csv "$ANALYSIS_DIR/${PREFIX}.merged.csv" \
  --output-dir "$PLOTS_DIR" \
  --metric "$METRIC"
