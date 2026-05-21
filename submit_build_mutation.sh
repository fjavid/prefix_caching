#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=build_mutation
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Under sbatch, the script is run from a spool dir, so BASH_SOURCE is unreliable.
# Prefer PIPELINE_DIR (set by run_pipeline.sh), then SLURM_SUBMIT_DIR, then BASH_SOURCE.
SCRIPT_DIR="${PIPELINE_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)}}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

load_modules
activate_venv
set_offline_env

mkdir -p "$PROCESSED_DIR" "$MUTATION_DIR"
cd "$PROJECT_ROOT"

echo "Building mutation dataset:"
echo "  workload=$WORKLOAD semantic_class=$SEMANTIC_CLASS"
echo "  generation_class=$GENERATION_CLASS mutation_type=$MUTATION_TYPE"
echo "  output_root=$SCRATCH_ROOT/mutation"

python -m prompt_mutation.build_mutation_dataset \
  --workload "$WORKLOAD" \
  --load-processed-path "$PROCESSED_DIR/${WORKLOAD}_examples.jsonl" \
  --semantic-class "$SEMANTIC_CLASS" \
  --generation-class "$GENERATION_CLASS" \
  --mutation-type "$MUTATION_TYPE" \
  --validation-backend sentence_transformer \
  --severity-backend sentence_transformer \
  --output-root "$SCRATCH_ROOT/mutation"
