#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=apply_layouts
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

mkdir -p "$LAYOUT_DIR"
cd "$PROJECT_ROOT"

# Allow caller to pin a specific mutation file via INPUT_JSONL, otherwise auto-discover.
INPUT_JSONL="${INPUT_JSONL:-$(discover_mutation_jsonl)}"
if [[ -z "$INPUT_JSONL" || ! -f "$INPUT_JSONL" ]]; then
  echo "ERROR: could not find mutation JSONL in $MUTATION_DIR" >&2
  exit 1
fi
echo "Using mutation JSONL: $INPUT_JSONL"
echo "Strategies: $STRATEGIES"

for strategy in $STRATEGIES; do
  out_path="$(layout_jsonl_path "$strategy")"
  echo "=== applying strategy=$strategy -> $out_path ==="
  python -m prompt_organization.apply_layouts \
    --input-path "$INPUT_JSONL" \
    --strategy-name "$strategy" \
    --output-path "$out_path"
done
