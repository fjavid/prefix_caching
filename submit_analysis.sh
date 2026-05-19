#!/bin/bash
#SBATCH --account=def-mmehride
#SBATCH --job-name=analyze_prefix_cache
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3 arrow/24.0.0 opencv/4.13 python/3.12
module load scipy-stack/2025a

PROJECT_ROOT="$HOME/work/prefix_caching"
SCRATCH_ROOT="$SCRATCH/prefix_caching"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

mkdir -p "$SCRATCH_ROOT/analysis"
mkdir -p "$SCRATCH_ROOT/analysis/plots_rag_chunk_reorder_stable_first"

python -m analysis.analyze_prefix_cache \
  --input-paths \
    "$SCRATCH_ROOT/benchmark_results/rag_chunk_reorder_stable_first_cache_off.jsonl" \
    "$SCRATCH_ROOT/benchmark_results/rag_chunk_reorder_stable_first_cache_on.jsonl" \
  --output-dir "$SCRATCH_ROOT/analysis" \
  --prefix rag_chunk_reorder_stable_first

python -m analysis.plot_prefix_cache_results \
  --merged-csv "$SCRATCH_ROOT/analysis/rag_chunk_reorder_stable_first.merged.csv" \
  --output-dir "$SCRATCH_ROOT/analysis/plots_rag_chunk_reorder_stable_first"