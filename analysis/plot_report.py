"""Cross-mutation report plots.

Reads every `outputs/analysis/rag_<mutation>.merged.csv` produced by
`analyze_prefix_cache.py` and emits five figures that consolidate the
findings across all mutations into one view:

1. `report_shared_prefix_by_mutation_layout.png`
   Grouped bar chart of mean(token_shared_prefix_ratio) per (mutation
   x layout). Tells the "what the layout intervention does to the prompt
   structure" story.

2. `report_ttft_gain_vs_shared_prefix.png`
   Scatter of TTFT-gain vs token_shared_prefix_ratio on partial_reuse
   rows, all four mutations on one axis. Marker shape encodes mutation,
   color encodes layout. Horizontal reference lines for the exact_reuse
   ceiling and the unrelated_control noise floor (averaged across the
   combined data). Tells the "gain rises monotonically with shared
   prefix and saturates near the cache-hit ceiling" story.

3. `report_ttft_off_vs_prompt_tokens.png`
   Validates that TTFT (cache off) scales linearly with prompt length.
   Total prompt length is reconstructed as
   `first_divergence_token / token_shared_prefix_ratio` (verified
   identical across the two layouts of the same case). A linear fit's
   slope gives the per-token prefill cost; we compare that to the
   theoretical "2 x params" forward-pass FLOPs / A100 BF16 peak.

4. `report_ttft_on_vs_shared_ratio.png`
   Companion to (3): TTFT (cache on) as a function of the FRACTION of
   the prompt that was cached, with points colored by total prompt
   length. The expected band is `intercept_on + slope_off * N * (1 - ratio)`
   evaluated at the p10 and p90 of `total_tokens`; the data should sit
   inside that band. This view answers the conceptual question "how does
   cache coverage shrink TTFT?" without misleadingly suggesting that a
   single 1D linear fit captures everything (TTFT_on actually depends
   on two variables: ratio AND total length).

5. `report_ttft_gain_vs_first_divergence.png`
   TTFT_gain (= TTFT_off - TTFT_on) vs first_divergence_token. Because
   the gain depends ONLY on the number of cached tokens (not on the
   remaining prompt length), this should be the tightest 1D
   relationship in the entire study: a straight line through the
   origin, slope ~= per-token prefill cost (~11.6 mu_s/token from
   plot 3). The intercept being near zero is itself an important
   sanity check.

Run from the repo root after the analysis stage has produced the merged
CSVs for every mutation:

    python -m analysis.plot_report \
        --analysis-dir outputs/analysis \
        --output-dir   outputs/analysis/report

The script is intentionally short and self-contained so it can be read
end-to-end and tweaked for the final writeup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Marker shape per mutation type; color per layout strategy. Stable order.
MUTATION_MARKERS = {
    "chunk_reorder": "o",
    "typo": "s",
    "formatting": "^",
    "synonym_substitution": "D",
}
LAYOUT_COLORS = {
    "original": "#d62728",      # red
    "stable_first": "#1f77b4",  # blue
}


def load_all(analysis_dir: Path) -> pd.DataFrame:
    """Load every rag_<mutation>.merged.csv and concatenate."""
    frames = []
    for csv in sorted(analysis_dir.glob("rag_*.merged.csv")):
        df = pd.read_csv(csv)
        frames.append(df)
        print(f"  loaded {csv.name}: {len(df):>5d} rows")
    if not frames:
        raise FileNotFoundError(
            f"No rag_*.merged.csv files found under {analysis_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def plot_shared_prefix(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: mean token_shared_prefix_ratio per (mutation x layout)."""
    pr = df[df["relation"] == "partial_reuse"].dropna(
        subset=["token_shared_prefix_ratio"]
    )
    if pr.empty:
        return
    pivot = pr.pivot_table(
        index="mutation_type", columns="layout_strategy",
        values="token_shared_prefix_ratio", aggfunc="mean",
    )
    # Preserve a deterministic mutation order for the report.
    desired_order = ["chunk_reorder", "typo", "formatting", "synonym_substitution"]
    pivot = pivot.reindex([m for m in desired_order if m in pivot.index])

    mutations = list(pivot.index)
    layouts = list(pivot.columns)
    width = 0.8 / max(len(layouts), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, layout in enumerate(layouts):
        means = pivot[layout].to_numpy()
        positions = np.arange(len(mutations)) + j * width
        ax.bar(positions, means, width=width,
               label=layout, color=LAYOUT_COLORS.get(layout, None),
               edgecolor="black", linewidth=0.5)
        for x, y in zip(positions, means):
            if not np.isnan(y):
                ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(mutations)) + width * (len(layouts) - 1) / 2)
    ax.set_xticklabels(mutations, rotation=10, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean token_shared_prefix_ratio (partial_reuse)")
    ax.set_title("Prompt layout effect on shared-prefix ratio")
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.legend(title="layout_strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gain_vs_shared_prefix(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: ttft_gain_seconds vs token_shared_prefix_ratio on
    partial_reuse rows. Marker = mutation, color = layout. Horizontal
    reference lines for exact_reuse ceiling and unrelated_control floor
    (combined across mutations).
    """
    pr = df[df["relation"] == "partial_reuse"].dropna(
        subset=["token_shared_prefix_ratio", "ttft_gain_seconds"]
    )
    if pr.empty:
        return

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for mut, marker in MUTATION_MARKERS.items():
        for layout, color in LAYOUT_COLORS.items():
            sub = pr[(pr["mutation_type"] == mut) & (pr["layout_strategy"] == layout)]
            if sub.empty:
                continue
            ax.scatter(
                sub["token_shared_prefix_ratio"],
                sub["ttft_gain_seconds"] * 1000.0,  # ms for readability
                marker=marker, color=color,
                s=22, alpha=0.40, edgecolors="none",
                label=f"{mut} / {layout}",
            )

    # Reference lines from the same combined DataFrame.
    exact = df.loc[df["relation"] == "exact_reuse", "ttft_gain_seconds"].dropna()
    ctrl = df.loc[df["relation"] == "unrelated_control", "ttft_gain_seconds"].dropna()
    if not exact.empty:
        ax.axhline(exact.mean() * 1000.0, color="black", lw=1.0, ls="--",
                   label=f"exact_reuse ceiling ({exact.mean()*1000:.2f} ms)")
    if not ctrl.empty:
        ax.axhline(ctrl.mean() * 1000.0, color="grey", lw=1.0, ls=":",
                   label=f"unrelated_control floor ({ctrl.mean()*1000:.2f} ms)")
    ax.axhline(0, color="black", lw=0.4)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("token_shared_prefix_ratio  (fraction of base prompt shared with mutated)")
    ax.set_ylabel("ttft_gain_seconds  (ms)")
    ax.set_title(
        "TTFT gain vs shared-prefix ratio for altered (volatile last) and original prompt layouts"
    )
    # Two-column legend so it fits without overlapping data.
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _estimate_total_tokens(df: pd.DataFrame) -> pd.Series:
    """Total prompt tokens per row, reconstructed from existing columns.

    For partial_reuse rows the overlap analyzer recorded:
        token_shared_prefix_ratio = first_divergence_token / total_tokens
    So total = first_divergence / ratio. Verified empirically that the
    estimate is identical across the two layouts of the same case
    (e.g. 516, 455, 302, 323, ... tokens reproduce to one token across
    layouts), so the formula is exact, not approximate.
    """
    fd = pd.to_numeric(df["first_divergence_token"], errors="coerce")
    tr = pd.to_numeric(df["token_shared_prefix_ratio"], errors="coerce")
    # Avoid division-by-zero noise from rows where the mutation is at position 0.
    return (fd / tr).where(tr > 0.0)


def plot_ttft_off_vs_tokens(df: pd.DataFrame, out_path: Path,
                            n_params: float = 1.1e9,
                            gpu_peak_tflops: float = 312.0,
                            gpu_utilization: float = 0.55) -> None:
    """Scatter of ttft_off vs total-prompt-tokens with a linear fit.

    The slope of the fit (seconds / token) is the per-token prefill cost.
    Compare against a theoretical estimate derived from the model's
    parameter count and the GPU's peak BF16 tensor-core throughput:

        predicted_per_token_seconds = (2 * n_params) / (peak * utilization)

    The 2x in the numerator is the standard "forward pass costs ~2 *
    params FLOPs per token" rule. We use n_params = 1.1e9 for TinyLlama
    and an A100-SXM4-40GB peak of 312 TFLOPS BF16. Utilization defaults
    to 0.55, a typical achieved fraction of peak for transformer prefill
    at this scale.
    """
    pr = df[df["relation"] == "partial_reuse"].copy()
    pr["total_tokens"] = _estimate_total_tokens(pr)
    pr["ttft_off_ms"] = pd.to_numeric(pr["ttft_off"], errors="coerce") * 1000.0
    pr = pr.dropna(subset=["total_tokens", "ttft_off_ms"])
    # Drop a tiny number of outliers (>3 sigma) so the fit isn't dragged.
    if pr.empty:
        return
    z = (pr["ttft_off_ms"] - pr["ttft_off_ms"].mean()) / pr["ttft_off_ms"].std(ddof=0)
    pr = pr[z.abs() < 3.5]

    # Linear regression: ttft_off = slope * total_tokens + intercept.
    x = pr["total_tokens"].to_numpy(dtype=float)
    y = pr["ttft_off_ms"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)  # slope in ms / token

    # Theoretical per-token cost in ms / token.
    theoretical_per_token_ms = (
        (2.0 * n_params) / (gpu_peak_tflops * 1e12 * gpu_utilization)
    ) * 1000.0

    # Per-mutation colors so the structure stays readable.
    mut_colors = {
        "chunk_reorder": "#2ca02c",          # green
        "typo": "#ff7f0e",                   # orange
        "formatting": "#9467bd",             # purple
        "synonym_substitution": "#17becf",   # cyan
    }

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for mut, color in mut_colors.items():
        sub = pr[pr["mutation_type"] == mut]
        if sub.empty:
            continue
        ax.scatter(sub["total_tokens"], sub["ttft_off_ms"],
                   color=color, s=14, alpha=0.30,
                   edgecolors="none", label=mut)
    xs = np.linspace(x.min(), x.max(), 64)
    ax.plot(xs, slope * xs + intercept, color="black", lw=2.0,
            label=(f"linear fit:  slope={slope*1000:.2f} mu_s/token, "
                   f"intercept={intercept:.2f} ms"))
    # Show the theoretical line through the same intercept so the comparison is fair.
    ax.plot(xs, theoretical_per_token_ms * xs + intercept,
            color="black", lw=1.4, ls="--",
            label=(f"theoretical:  {theoretical_per_token_ms*1000:.2f} mu_s/token "
                   f"(2*params / {gpu_peak_tflops:.0f} TFLOPS * {gpu_utilization:.2f})"))

    ax.set_xlabel("total prompt tokens (estimated from first_divergence / shared_prefix_ratio)")
    ax.set_ylabel("ttft_off  (ms)")
    ax.set_title(
        f"TTFT (cache off) vs prompt length — validates per-token prefill cost\n"
        f"measured {slope*1000:.2f} mu_s/token vs predicted "
        f"{theoretical_per_token_ms*1000:.2f} mu_s/token "
        f"(TinyLlama-1.1B on A100 @ {gpu_utilization*100:.0f}% of {gpu_peak_tflops:.0f} TFLOPS)"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Also report the numeric comparison on stdout for the writeup.
    print("\nLinear validation of per-token prefill cost:")
    print(f"  measured slope     : {slope*1000:6.3f} mu_s/token  (={slope:.6f} ms/tok)")
    print(f"  measured intercept : {intercept:6.3f} ms")
    print(f"  theoretical slope  : {theoretical_per_token_ms*1000:6.3f} mu_s/token "
          f"(at {gpu_utilization*100:.0f}% of {gpu_peak_tflops:.0f} TFLOPS, n_params={n_params:.1e})")
    if theoretical_per_token_ms > 0:
        print(f"  achieved fraction  : {theoretical_per_token_ms/slope*gpu_utilization*100:5.1f}% of A100 dense peak "
              "(derived from measured slope; >50% means near-peak utilization).")


def plot_ttft_on_vs_shared_ratio(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of ttft_on vs the fraction of the prompt that is cached.

    x-axis: token_shared_prefix_ratio (0 = nothing cached, 1 = entire
            prompt cached).
    y-axis: ttft_on (ms).
    color:  total_tokens of that request (continuous colormap), because
            the spread in ttft_on at a given ratio is driven by prompt
            length:
                ttft_on ~= intercept + per_token_cost * total_tokens * (1 - ratio)

    Two dashed reference curves are overlaid using the cache-off slope
    fit from plot 3, evaluated at the 10th and 90th percentiles of
    total_tokens. Points should lie within the band defined by those
    curves; that band visualizes the genuine prompt-length variation
    and explains why a single 1D fit underdescribes the data.
    """
    pr = df[df["relation"] == "partial_reuse"].copy()
    pr["ratio"] = pd.to_numeric(pr["token_shared_prefix_ratio"], errors="coerce")
    pr["ttft_on_ms"] = pd.to_numeric(pr["ttft_on"], errors="coerce") * 1000.0
    pr["total_tokens"] = _estimate_total_tokens(pr)
    pr = pr.dropna(subset=["ratio", "ttft_on_ms", "total_tokens"])
    if pr.empty:
        return
    # Trim tail outliers in ttft_on so the colormap range isn't distorted.
    z = (pr["ttft_on_ms"] - pr["ttft_on_ms"].mean()) / pr["ttft_on_ms"].std(ddof=0)
    pr = pr[z.abs() < 3.5]

    # Use the per-token cost from the cache-off regression. Mirror plot 3's
    # outlier trimming exactly so the two plots use consistent numbers.
    off = df[df["relation"] == "partial_reuse"].copy()
    off["total_tokens"] = _estimate_total_tokens(off)
    off["ttft_off_ms"] = pd.to_numeric(off["ttft_off"], errors="coerce") * 1000.0
    off = off.dropna(subset=["total_tokens", "ttft_off_ms"])
    z_off = (off["ttft_off_ms"] - off["ttft_off_ms"].mean()) / off["ttft_off_ms"].std(ddof=0)
    off = off[z_off.abs() < 3.5]
    slope_off, intercept_off = np.polyfit(
        off["total_tokens"].to_numpy(dtype=float),
        off["ttft_off_ms"].to_numpy(dtype=float),
        1,
    )

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sc = ax.scatter(
        pr["ratio"], pr["ttft_on_ms"],
        c=pr["total_tokens"], cmap="viridis",
        s=18, alpha=0.55, edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("total prompt tokens")

    # Expected curves at p10 and p90 of total_tokens to bracket the band.
    # ttft_on(ratio | N) ~= intercept_on_floor + slope_off * N * (1 - ratio)
    # We use the cache-on baseline intercept (TTFT_on at ratio=1, ~ 8 ms).
    rs = np.linspace(0.0, 1.0, 64)
    n_lo, n_hi = (pr["total_tokens"].quantile(0.10), pr["total_tokens"].quantile(0.90))
    intercept_on = pr.loc[pr["ratio"] > 0.95, "ttft_on_ms"].mean()
    if not np.isnan(intercept_on):
        ax.plot(rs, intercept_on + slope_off * n_lo * (1 - rs),
                color="black", lw=1.4, ls="--",
                label=f"predicted: total_tokens={n_lo:.0f}  (p10)")
        ax.plot(rs, intercept_on + slope_off * n_hi * (1 - rs),
                color="black", lw=1.4, ls=":",
                label=f"predicted: total_tokens={n_hi:.0f}  (p90)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("fraction of prompt cached  (token_shared_prefix_ratio)")
    ax.set_ylabel("ttft_on  (ms)")
    ax.set_title(
        "TTFT (cache on) vs fraction of prompt cached, colored by total prompt length\n"
        f"Two dashed curves: intercept + {slope_off*1000:.1f} mu_s/token * "
        f"total_tokens * (1 - ratio) at p10/p90 of total_tokens"
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\nTTFT_on vs cache fraction:")
    print(f"  expected: ttft_on(ratio) = intercept + slope_off * N * (1 - ratio)")
    print(f"  slope_off = {slope_off*1000:.3f} mu_s/token (from cache-off plot 3)")
    print(f"  intercept_on (at ratio ~= 1) = {intercept_on:.2f} ms")
    print("  Spread in y at fixed ratio is driven by prompt-length variation (p10..p90 band shown).")


def plot_ttft_gain_vs_first_divergence(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of TTFT_gain (ms) vs first_divergence_token (cached tokens).

    Of all the plots in this report, this should give the cleanest 1D
    linear relationship, because:

        TTFT_gain = TTFT_off - TTFT_on
                  ~= [intercept_off + slope * total_tokens]
                     - [intercept_on  + slope * (total_tokens - first_div)]
                  = (intercept_off - intercept_on) + slope * first_div

    The `total_tokens` term cancels: gain depends ONLY on first_div.
    So we expect a tight straight line through the origin (modulo a
    small intercept_off - intercept_on offset), slope equal to the
    per-token prefill cost from plot 3.
    """
    pr = df[df["relation"] == "partial_reuse"].copy()
    pr["first_div"] = pd.to_numeric(pr["first_divergence_token"], errors="coerce")
    pr["gain_ms"] = pd.to_numeric(pr["ttft_gain_seconds"], errors="coerce") * 1000.0
    pr = pr.dropna(subset=["first_div", "gain_ms"])
    if pr.empty:
        return
    z = (pr["gain_ms"] - pr["gain_ms"].mean()) / pr["gain_ms"].std(ddof=0)
    pr = pr[z.abs() < 3.5]

    x = pr["first_div"].to_numpy(dtype=float)
    y = pr["gain_ms"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    mut_colors = {
        "chunk_reorder": "#2ca02c",
        "typo": "#ff7f0e",
        "formatting": "#9467bd",
        "synonym_substitution": "#17becf",
    }
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for mut, color in mut_colors.items():
        sub = pr[pr["mutation_type"] == mut]
        if sub.empty:
            continue
        ax.scatter(sub["first_div"], sub["gain_ms"],
                   color=color, s=14, alpha=0.30,
                   edgecolors="none", label=mut)
    xs = np.linspace(x.min(), x.max(), 64)
    ax.plot(xs, slope * xs + intercept, color="black", lw=2.0,
            label=(f"linear fit:  slope={slope*1000:.2f} mu_s/cached_token, "
                   f"intercept={intercept:.2f} ms"))
    ax.axhline(0, color="black", lw=0.4)

    ax.set_xlabel("first_divergence_token  (= number of cached tokens)")
    ax.set_ylabel("ttft_gain_seconds  (ms)")
    ax.set_title("TTFT gain vs cached tokens")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\nTTFT_gain vs first_divergence_token:")
    print(f"  measured slope     : {slope*1000:6.3f} mu_s/cached_token")
    print(f"  measured intercept : {intercept:6.3f} ms")
    print(f"  Compare slope to plot-3 cache-off slope. Same number = clean cache substitution.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", default="outputs/analysis",
                   help="Directory containing rag_<mutation>.merged.csv files.")
    p.add_argument("--output-dir", default="outputs/analysis/report",
                   help="Where to write report figures.")
    p.add_argument("--n-params", type=float, default=1.1e9,
                   help="Model parameter count for the validation plot.")
    p.add_argument("--gpu-peak-tflops", type=float, default=312.0,
                   help="GPU peak BF16/FP16 tensor-core throughput in TFLOPS. "
                        "Default 312 is A100-SXM4-40GB dense peak.")
    p.add_argument("--gpu-utilization", type=float, default=0.55,
                   help="Achieved fraction of peak (0-1). 0.55 is a typical "
                        "transformer-prefill number; bump if you see your GPU "
                        "running hotter on similar workloads.")
    args = p.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(analysis_dir)
    print(f"Total combined rows: {len(df)}")
    print(f"Mutations present : {sorted(df['mutation_type'].dropna().unique())}")
    print(f"Layouts present   : {sorted(df['layout_strategy'].dropna().unique())}")

    plot_shared_prefix(
        df, out_dir / "report_shared_prefix_by_mutation_layout.png",
    )
    plot_gain_vs_shared_prefix(
        df, out_dir / "report_ttft_gain_vs_shared_prefix.png",
    )
    plot_ttft_off_vs_tokens(
        df, out_dir / "report_ttft_off_vs_prompt_tokens.png",
        n_params=args.n_params,
        gpu_peak_tflops=args.gpu_peak_tflops,
        gpu_utilization=args.gpu_utilization,
    )
    plot_ttft_on_vs_shared_ratio(
        df, out_dir / "report_ttft_on_vs_shared_ratio.png",
    )
    plot_ttft_gain_vs_first_divergence(
        df, out_dir / "report_ttft_gain_vs_first_divergence.png",
    )
    print(f"\nSaved 5 figures to {out_dir}")


if __name__ == "__main__":
    main()
