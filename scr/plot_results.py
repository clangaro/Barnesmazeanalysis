from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import setup_logging


Metric = Tuple[str, str]  # (column, pretty label)
METRICS: List[Metric] = [
    ("poke_frequency", "Poke Frequency"),
    ("latency_first_poke_s", "Latency to First Poke (s)"),
    ("time_in_hole_s", "Time in Hole (s)"),
    ("time_in_boundary_s", "Time in Boundary Zone (s)"),
    ("percent_in_quadrant", "Time in Target Quadrant (%)"),
]


def plot_bar_sem_with_points(
    df_in: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    output_dir: Path,
    jitter: float = 0.1,
) -> None:
    """
    Bar means with SEM and overlay individual points.
    Saves the figure to `output_dir`.
    """
    df = df_in.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    stats = (
        df.groupby(group_col, dropna=False)[value_col]
          .agg(["mean", "count", "std"])
          .reset_index()
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    groups = stats[group_col].tolist()
    means = stats["mean"].to_numpy()
    sems = stats["sem"].to_numpy()

    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=sems, capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.set_xlabel(group_col)

    # overlay points
    for i, g in enumerate(groups):
        pts = df.loc[df[group_col] == g, value_col].dropna()
        xs = np.full(len(pts), x[i]) + np.random.uniform(-jitter, jitter, size=len(pts))
        ax.scatter(xs, pts, s=30, color="black", alpha=0.7)

    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    out = output_dir / f"{title.replace(' ', '_').lower()}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logging.info("Saved plot: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary metrics with SEM + points.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to barnes_summary_id_trial.csv")
    parser.add_argument("--results", type=Path, required=True, help="Folder to save plots")
    parser.add_argument("--group", default="trial", choices=["trial", "sex", "mouse_id"])
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    df = pd.read_csv(args.summary)
    for col, label in METRICS:
        plot_bar_sem_with_points(df, args.group, col, f"{label} by {args.group.title()}", args.results)


if __name__ == "__main__":
    main()
