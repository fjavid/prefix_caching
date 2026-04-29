#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=apply_layouts
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# # Robust scratch dir
# SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}}"
# mkdir -p "$SCRATCH_DIR"
# echo "SCRATCH_DIR=$SCRATCH_DIR"

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

mkdir -p "$SCRATCH_ROOT/prompt_organization"

INPUT_JSONL="$SCRATCH_ROOT/mutation/rag/meaning_preserving/algorithmic/chunk_reorder/<your_file>.jsonl"

python -m prompt_organization.apply_layouts \
  --input-path "$INPUT_JSONL" \
  --strategy-name original \
  --output-path "$SCRATCH_ROOT/prompt_organization/rag_chunk_reorder_original.jsonl"

python -m prompt_organization.apply_layouts \
  --input-path "$INPUT_JSONL" \
  --strategy-name stable_first \
  --output-path "$SCRATCH_ROOT/prompt_organization/rag_chunk_reorder_stable_first.jsonl"

python -m prompt_organization.apply_layouts \
  --input-path "$INPUT_JSONL" \
  --strategy-name stable_first_normalized \
  --output-path "$SCRATCH_ROOT/prompt_organization/rag_chunk_reorder_stable_first_normalized.jsonl"