
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RELATION_ORDER = ["exact_reuse", "partial_reuse", "unrelated_control"]


def _agg(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    g = df.groupby(x)[y]
    return pd.DataFrame({
        "mean": g.mean(),
        "median": g.median(),
        "sem": g.sem(),
        "n": g.count(),
    }).reset_index()


def _grouped_bar(df: pd.DataFrame, x: str, hue: str, y: str, order: Iterable[str],
                 out_path: Path, title: str, y_label: str) -> None:
    if df.empty:
        return
    hue_levels = list(df[hue].dropna().unique())
    hue_levels.sort()
    x_levels = [v for v in order if v in set(df[x].unique())]
    width = 0.8 / max(len(hue_levels), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, h in enumerate(hue_levels):
        sub = df[df[hue] == h]
        means = []
        sems = []
        for xl in x_levels:
            s = sub[sub[x] == xl][y]
            means.append(float(s.mean()) if len(s) else 0.0)
            sems.append(float(s.sem()) if len(s) > 1 else 0.0)
        positions = np.arange(len(x_levels)) + j * width
        ax.bar(positions, means, width=width, label=str(h), yerr=sems, capsize=3)
    ax.set_xticks(np.arange(len(x_levels)) + width * (len(hue_levels) - 1) / 2)
    ax.set_xticklabels(x_levels)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(title=hue, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _box(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, y_label: str) -> None:
    if df.empty:
        return
    levels = sorted(df[x].dropna().unique())
    data = [df[df[x] == lvl][y].dropna().to_numpy() for lvl in levels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=levels, showmeans=True)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter(df: pd.DataFrame, x: str, y: str, hue: str, out_path: Path,
             title: str, x_label: str, y_label: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, sub in df.groupby(hue):
        ax.scatter(sub[x], sub[y], label=str(h), alpha=0.7, s=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(title=hue, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _heatmap(pivot: pd.DataFrame, out_path: Path, title: str, value_label: str,
             center: Optional[float] = 0.0) -> None:
    if pivot.empty:
        return
    arr = pivot.to_numpy(dtype=float)
    vmax = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)), 1e-9)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * pivot.shape[1] + 3),
                                    max(3, 0.5 * pivot.shape[0] + 2)))
    im = ax.imshow(arr, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax + (center or 0.0), vmax=vmax + (center or 0.0))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(v) < 0.5 * vmax else "white")
    fig.colorbar(im, ax=ax, label=value_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _violin(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str, y_label: str) -> None:
    if df.empty:
        return
    levels = sorted(df[x].dropna().unique())
    data = [df[df[x] == lvl][y].dropna().to_numpy() for lvl in levels]
    data = [d for d in data if len(d) > 0]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([str(lvl) for lvl, d in zip(levels, data) if len(d) > 0])
    ax.set_xlabel(x)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_with_trend(df: pd.DataFrame, x: str, y: str, hue: str, out_path: Path,
                        title: str, x_label: str, y_label: str) -> None:
    if df.empty:
        return
    sub_clean = df.dropna(subset=[x, y])
    if sub_clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, sub in sub_clean.groupby(hue):
        if len(sub) < 2:
            ax.scatter(sub[x], sub[y], label=str(h), alpha=0.7, s=20)
            continue
        ax.scatter(sub[x], sub[y], label=str(h), alpha=0.5, s=20)
        coeffs = np.polyfit(sub[x].to_numpy(), sub[y].to_numpy(), 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), lw=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(title=hue, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _faceted_grouped_bar(merged: pd.DataFrame, facet: str, x: str, hue: str, y: str,
                          order: Iterable[str], out_path: Path, suptitle: str,
                          y_label: str) -> None:
    facet_levels = sorted(merged[facet].dropna().unique())
    if len(facet_levels) < 2:
        return
    hue_levels = sorted(merged[hue].dropna().unique())
    x_levels = [v for v in order if v in set(merged[x].unique())]
    width = 0.8 / max(len(hue_levels), 1)
    fig, axes = plt.subplots(1, len(facet_levels),
                             figsize=(5 * len(facet_levels), 5), sharey=True)
    if len(facet_levels) == 1:
        axes = [axes]
    for ax, fl in zip(axes, facet_levels):
        sub = merged[merged[facet] == fl]
        for j, h in enumerate(hue_levels):
            s = sub[sub[hue] == h]
            means = [float(s[s[x] == xl][y].mean()) if len(s[s[x] == xl]) else 0.0 for xl in x_levels]
            sems = [float(s[s[x] == xl][y].sem()) if len(s[s[x] == xl]) > 1 else 0.0 for xl in x_levels]
            positions = np.arange(len(x_levels)) + j * width
            ax.bar(positions, means, width=width, label=str(h), yerr=sems, capsize=3)
        ax.set_xticks(np.arange(len(x_levels)) + width * (len(hue_levels) - 1) / 2)
        ax.set_xticklabels(x_levels, rotation=20, ha="right")
        ax.set_title(f"{facet}={fl}")
        ax.axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel(y_label)
    axes[-1].legend(title=hue, fontsize=9, loc="best")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_cross_category_plots(merged: pd.DataFrame, out_dir: Path, metric: str) -> None:
    """Plots that compare categorical slices against each other.

    Only emits a plot when the underlying slice has >1 level (otherwise the
    figure would be degenerate). All plots target the partial_reuse relation
    unless stated otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = merged[merged["relation"] == "partial_reuse"].copy()
    if pr.empty:
        return

    # 1) Heatmap mutation_type x layout_strategy.
    pivot = pr.pivot_table(index="mutation_type", columns="layout_strategy",
                           values=metric, aggfunc="mean")
    _heatmap(
        pivot, out_dir / f"{metric}_heatmap_mutation_x_layout.png",
        title=f"Mean {metric} (partial_reuse) - mutation_type x layout_strategy",
        value_label=metric,
    )

    # 2) Heatmap speedup_ratio (only meaningful if we have latency columns; metric != speedup).
    if {"latency_on", "latency_off"}.issubset(pr.columns) and metric != "latency_speedup_ratio":
        pr["_speedup"] = pr["latency_off"] / pr["latency_on"]
        pivot_sp = pr.pivot_table(index="mutation_type", columns="layout_strategy",
                                  values="_speedup", aggfunc="median")
        # Speedup is centered at 1 (no cache benefit), not 0.
        if not pivot_sp.empty:
            arr = pivot_sp.to_numpy(dtype=float) - 1.0  # plot delta from 1.0 so 0 means neutral.
            tmp = pd.DataFrame(arr, index=pivot_sp.index, columns=pivot_sp.columns)
            _heatmap(
                tmp, out_dir / "speedup_minus_1_heatmap_mutation_x_layout.png",
                title="Median speedup - 1.0 (partial_reuse) - mutation_type x layout_strategy",
                value_label="speedup - 1.0",
            )

    # 3) Distribution overlay (violin) of metric per layout, partial_reuse.
    _violin(
        pr, x="layout_strategy", y=metric,
        out_path=out_dir / f"{metric}_violin_partial_reuse_by_layout.png",
        title=f"{metric} distribution (partial_reuse) by layout strategy",
        y_label=metric,
    )

    # 4) Severity vs gain scatter with per-strategy trend.
    if "severity_combined_score" in pr.columns and pr["severity_combined_score"].notna().any():
        _scatter_with_trend(
            pr, x="severity_combined_score", y=metric, hue="layout_strategy",
            out_path=out_dir / f"{metric}_vs_severity_partial_reuse.png",
            title=f"{metric} vs mutation severity (partial_reuse)",
            x_label="severity_combined_score", y_label=metric,
        )

    # 5) Recovery delta vs original baseline.
    if "original" in set(pr["layout_strategy"].unique()):
        pr["_prompt_id"] = pr["case_id"].str.replace(r"::(partial|exact|control)$", "", regex=True)
        delta_rows: List[Dict[str, Any]] = []
        for mut, sub in pr.groupby("mutation_type"):
            piv = sub.pivot_table(index="_prompt_id", columns="layout_strategy",
                                  values=metric, aggfunc="mean")
            if "original" not in piv.columns:
                continue
            for strat in piv.columns:
                if strat == "original":
                    continue
                d = (piv[strat] - piv["original"]).dropna()
                if d.empty:
                    continue
                delta_rows.append({
                    "mutation_type": mut, "layout_strategy": strat,
                    "median_delta": float(d.median()),
                    "sem_delta": float(d.sem()) if len(d) > 1 else 0.0,
                })
        delta_df = pd.DataFrame(delta_rows)
        if not delta_df.empty:
            muts = sorted(delta_df["mutation_type"].unique())
            strats = sorted(delta_df["layout_strategy"].unique())
            width = 0.8 / max(len(strats), 1)
            fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(muts) + 2), 5))
            for j, s in enumerate(strats):
                sub = delta_df[delta_df["layout_strategy"] == s].set_index("mutation_type").reindex(muts)
                positions = np.arange(len(muts)) + j * width
                ax.bar(positions, sub["median_delta"].fillna(0.0), width=width,
                       yerr=sub["sem_delta"].fillna(0.0), capsize=3, label=s)
            ax.set_xticks(np.arange(len(muts)) + width * (len(strats) - 1) / 2)
            ax.set_xticklabels(muts, rotation=20, ha="right")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel(f"median({metric}) - median({metric} on original)")
            ax.set_title(f"Recovery of {metric} vs 'original' baseline (partial_reuse)")
            ax.legend(title="layout_strategy", fontsize=9)
            fig.tight_layout()
            fig.savefig(out_dir / f"{metric}_recovery_vs_original.png", dpi=150)
            plt.close(fig)

    # 6) Faceted grouped bars: one facet per workload (or per semantic_class).
    if merged["workload"].nunique() > 1:
        _faceted_grouped_bar(
            merged, facet="workload", x="relation", hue="layout_strategy", y=metric,
            order=RELATION_ORDER,
            out_path=out_dir / f"{metric}_facet_by_workload.png",
            suptitle=f"Mean {metric} by relation x layout, faceted by workload",
            y_label=metric,
        )
    if merged["semantic_class"].nunique() > 1:
        _faceted_grouped_bar(
            merged, facet="semantic_class", x="relation", hue="layout_strategy", y=metric,
            order=RELATION_ORDER,
            out_path=out_dir / f"{metric}_facet_by_semantic_class.png",
            suptitle=f"Mean {metric} by relation x layout, faceted by semantic_class",
            y_label=metric,
        )


def make_all_plots(merged: pd.DataFrame, out_dir: Path, metric: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _grouped_bar(
        merged, x="relation", hue="layout_strategy", y=metric,
        order=RELATION_ORDER,
        out_path=out_dir / f"{metric}_by_relation_and_layout.png",
        title=f"Mean {metric} by relation and layout strategy",
        y_label=metric,
    )

    pr = merged[merged["relation"] == "partial_reuse"]
    _box(
        pr, x="layout_strategy", y=metric,
        out_path=out_dir / f"{metric}_box_partial_reuse_by_layout.png",
        title=f"{metric} distribution (partial_reuse) by layout strategy",
        y_label=metric,
    )

    if "first_divergence_token" in merged.columns:
        _scatter(
            pr.dropna(subset=["first_divergence_token", metric]),
            x="first_divergence_token", y=metric, hue="layout_strategy",
            out_path=out_dir / f"{metric}_vs_first_divergence_partial_reuse.png",
            title=f"{metric} vs first divergence token (partial_reuse)",
            x_label="First divergence token", y_label=metric,
        )

    if {"latency_off", "latency_on"}.issubset(merged.columns):
        _scatter(
            merged.dropna(subset=["latency_off", "latency_on"]),
            x="latency_off", y="latency_on", hue="layout_strategy",
            out_path=out_dir / "latency_on_vs_off_by_layout.png",
            title="Cache-on latency vs cache-off latency",
            x_label="latency_off (s)", y_label="latency_on (s)",
        )

    if pr["mutation_type"].nunique() > 1:
        agg = _agg(pr, "mutation_type", metric).sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(agg["mutation_type"], agg["mean"], yerr=agg["sem"], capsize=3)
        ax.set_ylabel(f"Mean {metric}")
        ax.set_title(f"Mean {metric} by mutation_type (partial_reuse)")
        ax.axhline(0, color="black", lw=0.8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_by_mutation_type_partial_reuse.png", dpi=150)
        plt.close(fig)

    make_cross_category_plots(merged, out_dir / "cross_category", metric)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged-csv", required=True,
                   help="Combined merged CSV across all strategies.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metric", default="latency_gain_seconds",
                   choices=["latency_gain_seconds", "latency_speedup_ratio",
                            "ttft_gain_seconds", "ttft_speedup_ratio"])
    args = p.parse_args()
    merged = pd.read_csv(args.merged_csv)
    make_all_plots(merged, Path(args.output_dir), metric=args.metric)
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
