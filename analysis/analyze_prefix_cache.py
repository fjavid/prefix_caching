
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def load_benchmark_jsonl(path: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            case = row.get("case", {})
            follow = row.get("followup_metrics", {})
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

    key_cols = [
        "case_id", "workload", "semantic_class", "mutation_type", "relation",
        "layout_strategy", # "first_divergence_token", "token_shared_prefix_ratio",
        # "sequence_match_ratio", "semantic_cosine_overlap", "validation_is_valid",
        # "severity_combined_score",
    ]

    meta_cols = [
        "first_divergence_token",
        "token_shared_prefix_ratio",
        "sequence_match_ratio",
        "semantic_cosine_overlap",
        "validation_is_valid",
        "severity_combined_score",
    ]

    merged = cache_on[key_cols + meta_cols + ["ttft_on", "latency_on", "tps_on"]].merge(
    cache_off[key_cols + ["ttft_off", "latency_off", "tps_off"]],
    on=key_cols,
    how="inner",
)
    # merged = cache_on[key_cols + ["ttft_on", "latency_on", "tps_on"]].merge(
    #     cache_off[key_cols + ["ttft_off", "latency_off", "tps_off"]],
    #     on=key_cols,
    #     how="inner",
    # )

    merged["latency_gain_seconds"] = merged["latency_off"] - merged["latency_on"]
    merged["latency_speedup_ratio"] = merged["latency_off"] / merged["latency_on"]

    if merged["ttft_on"].notna().any() and merged["ttft_off"].notna().any():
        merged["ttft_gain_seconds"] = merged["ttft_off"] - merged["ttft_on"]
        merged["ttft_speedup_ratio"] = merged["ttft_off"] / merged["ttft_on"]
    return merged


def summarize_breakpoints(merged: pd.DataFrame) -> Dict[str, Any]:
    if merged.empty:
        return {"note": "No rows after merging cache-on and cache-off results."}

    out: Dict[str, Any] = {}
    out["by_mutation_type"] = (
        merged.groupby("mutation_type")[["latency_gain_seconds", "latency_speedup_ratio"]]
        .mean(numeric_only=True)
        .sort_values("latency_gain_seconds", ascending=False)
        .round(6)
        .to_dict(orient="index")
    )
    out["by_workload"] = (
        merged.groupby("workload")[["latency_gain_seconds", "latency_speedup_ratio"]]
        .mean(numeric_only=True)
        .round(6)
        .to_dict(orient="index")
    )
    out["by_layout_strategy"] = (
        merged.groupby("layout_strategy")[["latency_gain_seconds", "latency_speedup_ratio"]]
        .mean(numeric_only=True)
        .round(6)
        .to_dict(orient="index")
    )

    tmp = merged.copy()
    tmp["prefix_bin"] = pd.cut(
        tmp["token_shared_prefix_ratio"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
    )
    out["by_prefix_bin"] = (
        tmp.groupby("prefix_bin")[["latency_gain_seconds", "latency_speedup_ratio"]]
        .mean(numeric_only=True)
        .round(6)
        .to_dict(orient="index")
    )
    return out


def save_outputs(df: pd.DataFrame, merged: pd.DataFrame, summary: Dict[str, Any], output_dir: str, prefix: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{prefix}.flat.csv", index=False)
    merged.to_csv(out_dir / f"{prefix}.merged.csv", index=False)
    (out_dir / f"{prefix}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-paths", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="analysis")
    args = p.parse_args()

    dfs = [load_benchmark_jsonl(path) for path in args.input_paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    merged = merge_cache_on_off(df)
    summary = summarize_breakpoints(merged)
    save_outputs(df, merged, summary, args.output_dir, args.prefix)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
