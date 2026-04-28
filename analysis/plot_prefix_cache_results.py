
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv)

    if "first_divergence_token" in df.columns and "latency_gain_seconds" in df.columns:
        plt.figure()
        plt.scatter(df["first_divergence_token"], df["latency_gain_seconds"])
        plt.xlabel("First divergence token")
        plt.ylabel("Latency gain (cache off - cache on)")
        plt.title("Latency gain vs first divergence token")
        plt.tight_layout()
        plt.savefig(out_dir / "latency_gain_vs_first_divergence.png")
        plt.close()

    if "mutation_type" in df.columns and "latency_gain_seconds" in df.columns:
        agg = df.groupby("mutation_type")["latency_gain_seconds"].mean().sort_values(ascending=False)
        plt.figure()
        agg.plot(kind="bar")
        plt.ylabel("Mean latency gain (s)")
        plt.title("Mean latency gain by mutation type")
        plt.tight_layout()
        plt.savefig(out_dir / "latency_gain_by_mutation_type.png")
        plt.close()

    if "layout_strategy" in df.columns and "latency_gain_seconds" in df.columns:
        agg = df.groupby("layout_strategy")["latency_gain_seconds"].mean().sort_values(ascending=False)
        plt.figure()
        agg.plot(kind="bar")
        plt.ylabel("Mean latency gain (s)")
        plt.title("Mean latency gain by layout strategy")
        plt.tight_layout()
        plt.savefig(out_dir / "latency_gain_by_layout_strategy.png")
        plt.close()


if __name__ == "__main__":
    main()
