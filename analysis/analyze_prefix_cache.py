
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_benchmark_jsonl(path: str) -> pd.DataFrame:
    """Load one benchmark JSONL into a flat DataFrame.

    Each row corresponds to a single (case, followup) measurement. Warmup
    requests are dropped at load time.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            case = row.get("case", {})
            follow = row.get("followup_metrics", {})
            base_metrics = row.get("base_metrics", {})
            # Drop any warmup-tagged entries that may have leaked through.
            if follow.get("phase") == "warmup" or case.get("relation") == "warmup":
                continue
            metadata = case.get("metadata", {}) or {}
            overlap = metadata.get("overlap_metrics", {}) or {}
            validation = metadata.get("validation", {}) or {}
            severity = metadata.get("severity", {}) or {}
            prompt_org = metadata.get("prompt_organization", {}) or {}

            rows.append({
                "case_id": case.get("case_id"),
                "workload": case.get("workload"),
                "semantic_class": case.get("semantic_class"),
                "mutation_type": case.get("mutation_type"),
                "relation": case.get("relation"),
                "cache_enabled": follow.get("cache_enabled"),
                "followup_ttft_seconds": follow.get("ttft_seconds"),
                "followup_latency_seconds": follow.get("latency_seconds"),
                "followup_tokens_per_second": follow.get("tokens_per_second"),
                "base_ttft_seconds": base_metrics.get("ttft_seconds"),
                "base_latency_seconds": base_metrics.get("latency_seconds"),
                "layout_strategy": prompt_org.get("strategy_name", "unknown"),
                "first_divergence_token": overlap.get("first_divergence_token"),
                "token_shared_prefix_ratio": overlap.get("token_shared_prefix_ratio"),
                "sequence_match_ratio": overlap.get("sequence_match_ratio"),
                "semantic_cosine_overlap": overlap.get("semantic_cosine"),
                "validation_is_valid": validation.get("is_valid"),
                "severity_combined_score": severity.get("combined_score") if isinstance(severity, dict) else None,
            })
    return pd.DataFrame(rows)


def merge_cache_on_off(df: pd.DataFrame) -> pd.DataFrame:
    cache_on = df[df["cache_enabled"] == True].copy()
    cache_off = df[df["cache_enabled"] == False].copy()

    cache_on = cache_on.rename(columns={
        "followup_ttft_seconds": "ttft_on",
        "followup_latency_seconds": "latency_on",
        "followup_tokens_per_second": "tps_on",
    })
    cache_off = cache_off.rename(columns={
        "followup_ttft_seconds": "ttft_off",
        "followup_latency_seconds": "latency_off",
        "followup_tokens_per_second": "tps_off",
    })

    # case_id alone is NOT unique once multiple layouts are stacked into the same
    # DataFrame: the same case_id appears once per (layout_strategy, cache_mode).
    # Joining on case_id only would do a cross product over layouts. We must join
    # on (case_id, layout_strategy) so cache_on and cache_off come from the same
    # layout run.
    key_cols = ["case_id", "layout_strategy"]
    meta_cols = [
        "workload",
        "semantic_class",
        "mutation_type",
        "relation",
        "first_divergence_token",
        "token_shared_prefix_ratio",
        "sequence_match_ratio",
        "semantic_cosine_overlap",
        "validation_is_valid",
        "severity_combined_score",
    ]

    dup_on = cache_on.duplicated(subset=key_cols).sum()
    dup_off = cache_off.duplicated(subset=key_cols).sum()
    if dup_on or dup_off:
        raise ValueError(
            f"Duplicate (case_id, layout_strategy) rows detected before merge: "
            f"cache_on={dup_on}, cache_off={dup_off}. Re-running the benchmark with "
            "a unique output path per (layout_strategy, cache_mode) should fix this."
        )

    merged = cache_on[key_cols + meta_cols + ["ttft_on", "latency_on", "tps_on"]].merge(
        cache_off[key_cols + ["ttft_off", "latency_off", "tps_off"]],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    merged["latency_gain_seconds"] = merged["latency_off"] - merged["latency_on"]
    merged["latency_speedup_ratio"] = merged["latency_off"] / merged["latency_on"]

    if merged["ttft_on"].notna().any() and merged["ttft_off"].notna().any():
        merged["ttft_gain_seconds"] = merged["ttft_off"] - merged["ttft_on"]
        merged["ttft_speedup_ratio"] = merged["ttft_off"] / merged["ttft_on"]
    return merged


def _bootstrap_ci(values: np.ndarray, num_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(num_boot, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_lo": float(np.quantile(boots, alpha / 2)),
        "ci_hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(len(values)),
    }


def _pivot_summary(
    merged: pd.DataFrame, metric: str, row: str, col: str,
    relation_filter: Optional[str] = "partial_reuse",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return {row_val: {col_val: bootstrap_ci_dict}} for the given metric."""
    df = merged if relation_filter is None else merged[merged["relation"] == relation_filter]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r_val, sub in df.groupby(row):
        out[str(r_val)] = {}
        for c_val, sub2 in sub.groupby(col):
            out[str(r_val)][str(c_val)] = _bootstrap_ci(sub2[metric].to_numpy())
    return out


def _recovery_vs_baseline(
    merged: pd.DataFrame, metric: str, baseline: str = "original",
    relation_filter: str = "partial_reuse",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each mutation_type, compute the per-case delta of `metric` between every
    non-baseline layout and the baseline layout, paired on prompt_id. Returns
    {mutation_type: {strategy: bootstrap_ci_dict}}.

    A positive delta means the strategy outperforms `baseline` on that case.
    """
    df = merged[merged["relation"] == relation_filter].copy()
    if df.empty or baseline not in set(df["layout_strategy"].unique()):
        return {}
    # case_id encodes the prompt + relation; strip the trailing strategy tag if present.
    # Pair on (mutation_type, prompt_id) where prompt_id is the case_id stem.
    df["prompt_id"] = df["case_id"].str.replace(r"::(partial|exact|control)$", "", regex=True)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mut, sub in df.groupby("mutation_type"):
        pivot = sub.pivot_table(index="prompt_id", columns="layout_strategy",
                                values=metric, aggfunc="mean")
        if baseline not in pivot.columns:
            continue
        deltas: Dict[str, Dict[str, float]] = {}
        for strat in pivot.columns:
            if strat == baseline:
                continue
            d = (pivot[strat] - pivot[baseline]).dropna().to_numpy()
            deltas[str(strat)] = _bootstrap_ci(d)
        out[str(mut)] = deltas
    return out


def summarize(merged: pd.DataFrame, metric: str = "latency_gain_seconds") -> Dict[str, Any]:
    if merged.empty:
        return {"note": "No rows after merging cache-on and cache-off results."}

    summary: Dict[str, Any] = {"metric": metric}

    # Noise floor: gain on unrelated_control cases (no prefix overlap, so any apparent
    # speedup is measurement noise / engine warmup that the warmup_iters didn't absorb).
    unrelated = merged[merged["relation"] == "unrelated_control"]
    summary["noise_floor"] = _bootstrap_ci(unrelated[metric].to_numpy())

    by_layout: Dict[str, Any] = {}
    for strat, sub in merged.groupby("layout_strategy"):
        per_relation: Dict[str, Any] = {}
        for rel, sub2 in sub.groupby("relation"):
            per_relation[rel] = _bootstrap_ci(sub2[metric].to_numpy())
        by_layout[strat] = per_relation
    summary["by_layout_strategy"] = by_layout

    by_mut: Dict[str, Any] = {}
    for mut, sub in merged.groupby("mutation_type"):
        by_mut[mut] = _bootstrap_ci(sub[sub["relation"] == "partial_reuse"][metric].to_numpy())
    summary["partial_reuse_by_mutation_type"] = by_mut

    # Cross-tabs: useful once there are multiple mutation_types / workloads / classes.
    summary["mutation_type_x_layout_strategy"] = _pivot_summary(
        merged, metric, row="mutation_type", col="layout_strategy")
    if merged["workload"].nunique() > 1:
        summary["workload_x_layout_strategy"] = _pivot_summary(
            merged, metric, row="workload", col="layout_strategy")
    if merged["semantic_class"].nunique() > 1:
        summary["semantic_class_x_layout_strategy"] = _pivot_summary(
            merged, metric, row="semantic_class", col="layout_strategy")

    summary["recovery_vs_original"] = _recovery_vs_baseline(merged, metric, baseline="original")

    if "token_shared_prefix_ratio" in merged.columns:
        tmp = merged[merged["relation"] == "partial_reuse"].copy()
        tmp["prefix_bin"] = pd.cut(
            tmp["token_shared_prefix_ratio"],
            bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
            labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        )
        by_bin: Dict[str, Any] = {}
        for b, sub in tmp.groupby("prefix_bin", observed=False):
            by_bin[str(b)] = _bootstrap_ci(sub[metric].to_numpy())
        summary["partial_reuse_by_prefix_bin"] = by_bin

    return summary


def save_outputs(df: pd.DataFrame, merged: pd.DataFrame, summary: Dict[str, Any],
                 output_dir: str, prefix: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{prefix}.flat.csv", index=False)
    merged.to_csv(out_dir / f"{prefix}.merged.csv", index=False)
    (out_dir / f"{prefix}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-paths", nargs="+", required=True,
                   help="Benchmark JSONL files (cache_on and cache_off; can include all strategies).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="analysis")
    p.add_argument("--metric", default="latency_gain_seconds",
                   choices=["latency_gain_seconds", "latency_speedup_ratio",
                            "ttft_gain_seconds", "ttft_speedup_ratio"])
    args = p.parse_args()

    dfs = [load_benchmark_jsonl(path) for path in args.input_paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    merged = merge_cache_on_off(df)
    summary = summarize(merged, metric=args.metric)
    save_outputs(df, merged, summary, args.output_dir, args.prefix)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
