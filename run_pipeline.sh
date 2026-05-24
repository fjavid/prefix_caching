#!/bin/bash
# Master orchestrator: submits the full prefix-caching pipeline as one
# afterok-chained SLURM job sequence per mutation type.
#
# Defaults (from pipeline_config.sh):
#   WORKLOAD          = rag
#   SEMANTIC_CLASS    = meaning_preserving
#   GENERATION_CLASS  = algorithmic
#   MUTATION_TYPE     = chunk_reorder      # single-mutation mode
#   STRATEGIES        = "original stable_first"
#   CACHE_MODES       = "off on"
#
# Multi-mutation mode: set MUTATION_TYPES to a space-separated list. Each
# mutation type gets its own independent SLURM chain (build_mutation ->
# apply_layouts -> benchmark -> analyze). Chains run in parallel from the
# scheduler's perspective; the per-mutation outputs never overwrite each
# other because all derived paths embed the TAG = ${WORKLOAD}_${MUTATION_TYPE}.
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
#   ./run_pipeline.sh                                                  # full pipeline, single mutation
#   MUTATION_TYPE=typo ./run_pipeline.sh                               # different single mutation
#   MUTATION_TYPES="chunk_reorder typo formatting" ./run_pipeline.sh   # three parallel chains
#   SKIP_MUTATION=1 ./run_pipeline.sh                                  # reuse existing mutation jsonl
#   SKIP_MUTATION=1 SKIP_LAYOUTS=1 ./run_pipeline.sh                   # only benchmark + analyze
#   STRATEGIES="original stable_first" ./run_pipeline.sh               # subset of strategies
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

# Resolve the mutation list. MUTATION_TYPES (plural) takes precedence when set;
# otherwise fall back to the scalar MUTATION_TYPE for backward compatibility.
MUTATION_LIST="${MUTATION_TYPES:-$MUTATION_TYPE}"

echo "Pipeline configuration:"
echo "  PROJECT_ROOT      = $PROJECT_ROOT"
echo "  SCRATCH_ROOT      = $SCRATCH_ROOT"
echo "  MODEL_PATH        = $MODEL_PATH"
echo "  WORKLOAD          = $WORKLOAD"
echo "  SEMANTIC_CLASS    = $SEMANTIC_CLASS"
echo "  GENERATION_CLASS  = $GENERATION_CLASS"
echo "  MUTATION_LIST     = $MUTATION_LIST"
echo "  STRATEGIES        = $STRATEGIES"
echo "  CACHE_MODES       = $CACHE_MODES"
echo "  SKIP_MUTATION=$SKIP_MUTATION  SKIP_LAYOUTS=$SKIP_LAYOUTS"
echo "  SKIP_BENCHMARK=$SKIP_BENCHMARK  SKIP_ANALYSIS=$SKIP_ANALYSIS"
echo ""

# Build a --export=ALL,KEY=VAL,... string that pins all our pipeline knobs into the child job env.
# Captures the CURRENT shell value of MUTATION_TYPE, so call this AFTER setting MUTATION_TYPE per iteration.
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

submit_with_dep() {
  # Submits an sbatch script with optional dependency, prints jobid only.
  # Usage: submit_with_dep <dep_arg_or_empty> <export_arg> <script_path>
  local dep="$1"; shift
  local export_arg="$1"; shift
  local script="$1"; shift
  local jid
  if [[ -n "$dep" ]]; then
    jid=$(sbatch --parsable "$dep" --export="$export_arg" "$script" | cut -d';' -f1)
  else
    jid=$(sbatch --parsable --export="$export_arg" "$script" | cut -d';' -f1)
  fi
  echo "$jid"
}

submit_chain() {
  # Submits the build_mutation -> apply_layouts -> benchmark -> analyze
  # chain for the currently-configured MUTATION_TYPE. Each chain is
  # independent across mutation types because every output path embeds
  # ${TAG}.
  local export_arg="$1"
  local dep=""

  if [[ "$SKIP_MUTATION" -eq 0 ]]; then
    local jid
    jid=$(submit_with_dep "$dep" "$export_arg" "$SCRIPT_DIR/submit_build_mutation.sh")
    echo "  build_mutation:   $jid"
    dep="--dependency=afterok:$jid"
  fi

  if [[ "$SKIP_LAYOUTS" -eq 0 ]]; then
    local jid
    jid=$(submit_with_dep "$dep" "$export_arg" "$SCRIPT_DIR/submit_apply_layouts.sh")
    echo "  apply_layouts:    $jid"
    dep="--dependency=afterok:$jid"
  fi

  if [[ "$SKIP_BENCHMARK" -eq 0 ]]; then
    local jid
    jid=$(submit_with_dep "$dep" "$export_arg" "$SCRIPT_DIR/submit_interface_benchmark.sh")
    echo "  benchmark:        $jid"
    dep="--dependency=afterok:$jid"
  fi

  if [[ "$SKIP_ANALYSIS" -eq 0 ]]; then
    local jid
    jid=$(submit_with_dep "$dep" "$export_arg" "$SCRIPT_DIR/submit_analysis.sh")
    echo "  analysis:         $jid"
  fi
}

# One independent chain per mutation type. Derived paths (TAG, MUTATION_DIR,
# etc.) are recomputed from pipeline_config.sh for each iteration so the
# export args carry the per-mutation values into the child jobs.
for mut in $MUTATION_LIST; do
  export MUTATION_TYPE="$mut"
  # Recompute derived paths for the new MUTATION_TYPE.
  source "$SCRIPT_DIR/pipeline_config.sh"
  export_arg="$(build_export_arg)"
  echo "=== Submitting chain for MUTATION_TYPE=$mut (TAG=$TAG) ==="
  submit_chain "$export_arg"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Per-mutation outputs land in:"
for mut in $MUTATION_LIST; do
  tag="${WORKLOAD}_${mut}"
  echo "  [$mut]"
  echo "    mutations  : $SCRATCH_ROOT/mutation/$WORKLOAD/$SEMANTIC_CLASS/$GENERATION_CLASS/$mut/"
  echo "    layouts    : $SCRATCH_ROOT/prompt_organization/${tag}_<strategy>.jsonl"
  echo "    benchmarks : $SCRATCH_ROOT/benchmark_results/${tag}_<strategy>_cache_<mode>.jsonl"
  echo "    analysis   : $SCRATCH_ROOT/analysis/${tag}.{flat,merged}.csv  and  plots_${tag}/"
done
