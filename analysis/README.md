
# analysis

This module implements step 5 analysis.

## Goal

Analyze benchmark results to identify:
- where prefix caching breaks
- which mutation types preserve reuse
- which prompt-organization baselines recover performance
- how benefit depends on first divergence and shared-prefix ratio

## Files

- `analyze_prefix_cache.py` — load multiple benchmark JSONLs (all strategies and cache modes),
  flatten to one DataFrame, merge cache-on/off into per-case gain/speedup, and emit a
  summary with bootstrap CIs and an `unrelated_control`-based noise floor.
- `plot_prefix_cache_results.py` — plots:
  - `*_by_relation_and_layout.png` — grouped bar: mean gain per relation x layout
  - `*_box_partial_reuse_by_layout.png` — distribution box per layout
  - `*_vs_first_divergence_partial_reuse.png` — scatter vs divergence point
  - `wall_clock_on_vs_off_by_layout.png` — sanity diagonal
  - `*_by_mutation_type_partial_reuse.png` — mean gain per mutation_type
  - `cross_category/` subdirectory:
    - `*_heatmap_mutation_x_layout.png` — diverging heatmap, mean gain
    - `speedup_minus_1_heatmap_mutation_x_layout.png` — median speedup minus 1
    - `*_violin_partial_reuse_by_layout.png` — distribution shape per layout
    - `*_vs_severity_partial_reuse.png` — severity vs gain with per-strategy regression
    - `*_recovery_vs_original.png` — paired median delta vs the `original` baseline
    - `*_facet_by_workload.png` / `*_facet_by_semantic_class.png` — facets when >1 level
- `test_analysis.ipynb` — notebook scaffold for interactive analysis.

## Where to run

The analysis stage uses only `pandas`, `numpy`, and `matplotlib`. No GPU, no
LLM, no internet. Three valid runtimes:

1. SLURM (`submit_analysis.sh`) — fires automatically at the end of `run_pipeline.sh`.
2. Login node — `bash analyze_local.sh` after activating `.venv`.
3. MacBook — rsync the JSONLs first, then `bash analyze_local.sh` with
   `RESULTS_ROOT=outputs/benchmark_results` (or wherever you copied to). Use this
   loop for iterating on plots without re-running benchmarks.

## Direct python invocation

```bash
python -m analysis.analyze_prefix_cache \
  --input-paths \
    $SCRATCH/prefix_caching/benchmark_results/rag_chunk_reorder_original_cache_{off,on}.jsonl \
    $SCRATCH/prefix_caching/benchmark_results/rag_chunk_reorder_stable_first_cache_{off,on}.jsonl \
  --output-dir $SCRATCH/prefix_caching/analysis \
  --prefix rag_chunk_reorder \
  --metric wall_clock_gain_seconds

python -m analysis.plot_prefix_cache_results \
  --merged-csv $SCRATCH/prefix_caching/analysis/rag_chunk_reorder.merged.csv \
  --output-dir $SCRATCH/prefix_caching/analysis/plots_rag_chunk_reorder \
  --metric wall_clock_gain_seconds
```

The metric flag accepts `wall_clock_gain_seconds`, `wall_clock_speedup_ratio`,
`ttft_gain_seconds`, or `ttft_speedup_ratio`. `wall_clock_*` is the
end-to-end wall-clock time per request (prefill + decode); `ttft_*` is the
prefill-dominated time-to-first-token and is the metric most directly
affected by prefix caching. TTFT requires the async vLLM backend
(see `inference_benchmark/`).

## Reading the summary JSON

`<prefix>.summary.json` is the canonical text report. Top-level keys:

- `metric` — which column was summarized.
- `noise_floor` — bootstrap mean / median / 95% CI of the metric on
  `unrelated_control` cases. Any per-strategy gain that overlaps this band
  should be treated as measurement noise, not signal.
- `by_layout_strategy[strategy][relation]` — per-cell bootstrap CI.
- `partial_reuse_by_mutation_type[mutation]` — gain restricted to partial_reuse.
- `mutation_type_x_layout_strategy` — full cross-tab (this is the table that
  drives the heatmap).
- `recovery_vs_original[mutation][strategy]` — bootstrap CI of the per-case
  delta `gain[strategy] - gain[original]`. Positive means the layout strategy
  recovers performance over the naive baseline; CI lo > 0 is the strong
  significance bar.
- `partial_reuse_by_prefix_bin` — gain bucketed by shared prefix ratio.
