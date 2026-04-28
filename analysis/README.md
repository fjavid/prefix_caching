
# analysis

This module implements step 5 analysis.

## Goal

Analyze benchmark results to identify:
- where prefix caching breaks
- which mutation types preserve reuse
- which prompt-organization baselines recover performance
- how benefit depends on first divergence and shared-prefix ratio

## Files

- `analyze_prefix_cache.py` — flatten benchmark JSONL, merge cache-on/off runs, summarize breakpoints
- `plot_prefix_cache_results.py` — generate a small set of baseline plots
- `test_analysis.ipynb` — notebook scaffold for interactive analysis
