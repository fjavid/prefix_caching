
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
    - `*_vs_shared_prefix_partial_reuse.png` — **headline scaling plot**: gain vs
      `token_shared_prefix_ratio` with horizontal reference lines for the
      `exact_reuse` ceiling and the `unrelated_control` noise floor. The cache
      benefit saturates near the ceiling once shared prefix is ~1.0.
    - `*_heatmap_mutation_x_layout.png` — diverging heatmap, mean gain
    - `speedup_minus_1_heatmap_mutation_x_layout.png` — median speedup minus 1
    - `*_violin_partial_reuse_by_layout.png` — distribution shape per layout
    - `*_vs_severity_partial_reuse.png` — severity vs gain with per-strategy regression
    - `*_recovery_vs_original.png` — paired median delta vs the `original` baseline
    - `*_facet_by_workload.png` / `*_facet_by_semantic_class.png` — facets when >1 level
- `plot_report.py` — writeup figures, including the roofline validation that
  compares the measured per-token prefill cost against `2 * n_params / peak`.
  **Its defaults describe TinyLlama-1.1B on an A100 and will mislabel anything
  else.** Pass the run's actual configuration:

  ```bash
  python -m analysis.plot_report \
    --analysis-dir outputs/analysis --output-dir outputs/analysis/report \
    --n-params 8.0e9 --model-label Llama-3.1-8B-Instruct \
    --gpu-peak-tflops 990 --gpu-label H100
  ```

  `N_PARAMS` is carried per model in the `pipeline_config.sh` registry.
  `--model-label` and `--gpu-label` affect captions only, and are always rendered
  **together with the value they claim to describe** — `Llama-3.1-8B-Instruct
  (8.0e+09 params)` — so a label that disagrees with `--n-params` is visible in
  the figure rather than hidden. Omitting them shows the bare value.

  The roofline comparison is a per-**model-token** cost. When a result set has no
  engine token counts (`followup_prompt_model_tokens` absent or all-null), the
  prompt-length series falls back to whitespace-word counts; the theoretical line
  and the achieved-peak figure are then **withheld**, and every axis, legend,
  title, and stdout line says "word" instead of "token". Comparing a per-word
  slope against a per-token peak overstates efficiency by the word-to-token ratio
  (measured 1.42–2.62).
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

Cluster artifacts are namespaced by `MODEL_TAG`, so the paths below include it.
Reading `$SCRATCH/prefix_caching/benchmark_results/...` without the tag would
either find nothing or find a pre-namespacing result set produced by a different
model.

```bash
# MODEL_TAG must be set: unset, $ROOT collapses to the legacy pre-namespacing
# path warned about above, which may hold another model's results.
MODEL_TAG=Llama-3.1-8B-Instruct
ROOT=$SCRATCH/prefix_caching/$MODEL_TAG

python -m analysis.analyze_prefix_cache \
  --input-paths \
    $ROOT/benchmark_results/rag_chunk_reorder_original_cache_{off,on}.jsonl \
    $ROOT/benchmark_results/rag_chunk_reorder_stable_first_cache_{off,on}.jsonl \
  --output-dir $ROOT/analysis \
  --prefix rag_chunk_reorder \
  --metric ttft_gain_seconds

python -m analysis.plot_prefix_cache_results \
  --merged-csv $ROOT/analysis/rag_chunk_reorder.merged.csv \
  --output-dir $ROOT/analysis/plots_rag_chunk_reorder \
  --metric ttft_gain_seconds
```

The metric flag accepts `ttft_gain_seconds` (**default**), `ttft_speedup_ratio`,
`wall_clock_gain_seconds`, or `wall_clock_speedup_ratio`. `ttft_*` is the
prefill-dominated time-to-first-token and is the metric most directly
affected by prefix caching; `wall_clock_*` is the end-to-end request time
(prefill + decode) and is noisier because decode time fluctuates run-to-run.
Use TTFT for headline results and wall-clock only when you specifically
want the end-to-end view. TTFT requires the async vLLM backend
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
