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

# Single-strategy override (when running outside the master pipeline).
if [[ -n "${STRATEGY:-}" ]]; then
  RUN_STRATEGIES="$STRATEGY"
else
  RUN_STRATEGIES="$STRATEGIES"
fi

echo "Analyzing strategies: $RUN_STRATEGIES"

for strategy in $RUN_STRATEGIES; do
  prefix="${TAG}_${strategy}"
  plots_dir="$ANALYSIS_DIR/plots_${prefix}"
  mkdir -p "$plots_dir"

  input_off="$(benchmark_jsonl_path "$strategy" off)"
  input_on="$(benchmark_jsonl_path "$strategy" on)"

  for f in "$input_off" "$input_on"; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: missing benchmark result: $f" >&2
      exit 1
    fi
  done

  echo "=== analyzing strategy=$strategy ==="
  python -m analysis.analyze_prefix_cache \
    --input-paths "$input_off" "$input_on" \
    --output-dir "$ANALYSIS_DIR" \
    --prefix "$prefix"

  python -m analysis.plot_prefix_cache_results \
    --merged-csv "$ANALYSIS_DIR/${prefix}.merged.csv" \
    --output-dir "$plots_dir"
done
