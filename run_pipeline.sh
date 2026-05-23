#!/bin/bash
# Master orchestrator: submits the full prefix-caching pipeline as a SLURM job chain.
#
# Defaults (from pipeline_config.sh):
#   WORKLOAD          = rag
#   SEMANTIC_CLASS    = meaning_preserving
#   GENERATION_CLASS  = algorithmic
#   MUTATION_TYPE     = chunk_reorder
#   STRATEGIES        = "original stable_first stable_first_normalized"
#   CACHE_MODES       = "off on"
#
# All knobs are overridable by exporting env vars before invocation.
# Stages can be individually skipped via SKIP_MUTATION / SKIP_LAYOUTS /
# SKIP_BENCHMARK / SKIP_ANALYSIS (set to 1 to skip).
#
# Prerequisite (run once on the LOGIN NODE before any sbatch job):
#   bash prep_login.sh        # builds .venv, downloads models, runs prepare_data.sh
#   # or, to refresh just the data:
#   bash prepare_data.sh
#
# Examples:
#   ./run_pipeline.sh                                    # full pipeline with defaults
#   SKIP_MUTATION=1 ./run_pipeline.sh                    # reuse existing mutation jsonl
#   SKIP_MUTATION=1 SKIP_LAYOUTS=1 ./run_pipeline.sh     # only benchmark + analyze
#   STRATEGIES="original stable_first" ./run_pipeline.sh # subset of strategies
#   MUTATION_TYPE=field_reorder ./run_pipeline.sh        # different mutation
#
# After submission, monitor with:  squeue -u $USER
# Each stage's logs land in $PWD as <jobname>-<jobid>.out / .err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Pinned so child sbatch scripts can locate pipeline_config.sh regardless of submit CWD
# or SLURM script-spooling.
export PIPELINE_DIR="$SCRIPT_DIR"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/pipeline_config.sh"

SKIP_MUTATION="${SKIP_MUTATION:-0}"
SKIP_LAYOUTS="${SKIP_LAYOUTS:-0}"
SKIP_BENCHMARK="${SKIP_BENCHMARK:-0}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

echo "Pipeline configuration:"
echo "  PROJECT_ROOT      = $PROJECT_ROOT"
echo "  SCRATCH_ROOT      = $SCRATCH_ROOT"
echo "  MODEL_PATH        = $MODEL_PATH"
echo "  WORKLOAD          = $WORKLOAD"
echo "  SEMANTIC_CLASS    = $SEMANTIC_CLASS"
echo "  GENERATION_CLASS  = $GENERATION_CLASS"
echo "  MUTATION_TYPE     = $MUTATION_TYPE"
echo "  STRATEGIES        = $STRATEGIES"
echo "  CACHE_MODES       = $CACHE_MODES"
echo "  TAG               = $TAG"
echo "  SKIP_MUTATION=$SKIP_MUTATION  SKIP_LAYOUTS=$SKIP_LAYOUTS"
echo "  SKIP_BENCHMARK=$SKIP_BENCHMARK  SKIP_ANALYSIS=$SKIP_ANALYSIS"
echo ""

# Build a --export=ALL,KEY=VAL,... string that pins all our pipeline knobs into the child job env.
build_export_arg() {
  local vars=(PIPELINE_DIR PROJECT_ROOT SCRATCH_ROOT MODEL_PATH
              WORKLOAD SEMANTIC_CLASS GENERATION_CLASS MUTATION_TYPE
              STRATEGIES CACHE_MODES USE_ASYNC_TTFT VLLM_LOGGING_LEVEL
              RESET_CACHE_BETWEEN_CASES)
  local pairs=()
  for v in "${vars[@]}"; do
    # Only export if defined (so unset vars stay unset rather than become "").
    if [[ -n "${!v+x}" ]]; then
      pairs+=("$v=${!v}")
    fi
  done
  local IFS=,
  echo "ALL,${pairs[*]}"
}

EXPORT_ARG="$(build_export_arg)"

submit_with_dep() {
  # Submits an sbatch script with optional dependency, prints jobid only.
  # Usage: submit_with_dep <dep_arg_or_empty> <script_path>
  local dep="$1"; shift
  local script="$1"; shift
  local jid
  if [[ -n "$dep" ]]; then
    jid=$(sbatch --parsable "$dep" --export="$EXPORT_ARG" "$script" | cut -d';' -f1)
  else
    jid=$(sbatch --parsable --export="$EXPORT_ARG" "$script" | cut -d';' -f1)
  fi
  echo "$jid"
}

DEP=""

if [[ "$SKIP_MUTATION" -eq 0 ]]; then
  JID=$(submit_with_dep "$DEP" "$SCRIPT_DIR/submit_build_mutation.sh")
  echo "Submitted build_mutation:   $JID"
  DEP="--dependency=afterok:$JID"
fi

if [[ "$SKIP_LAYOUTS" -eq 0 ]]; then
  JID=$(submit_with_dep "$DEP" "$SCRIPT_DIR/submit_apply_layouts.sh")
  echo "Submitted apply_layouts:    $JID"
  DEP="--dependency=afterok:$JID"
fi

if [[ "$SKIP_BENCHMARK" -eq 0 ]]; then
  JID=$(submit_with_dep "$DEP" "$SCRIPT_DIR/submit_interface_benchmark.sh")
  echo "Submitted benchmark:        $JID"
  DEP="--dependency=afterok:$JID"
fi

if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
  JID=$(submit_with_dep "$DEP" "$SCRIPT_DIR/submit_analysis.sh")
  echo "Submitted analysis:         $JID"
fi

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Outputs land in:"
echo "  mutations  : $MUTATION_DIR/"
echo "  layouts    : $LAYOUT_DIR/"
echo "  benchmarks : $BENCH_DIR/"
echo "  analysis   : $ANALYSIS_DIR/"
