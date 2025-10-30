from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import count_entries, ensure_numeric, save_csv, setup_logging


# Default analysis params
FPS_DEFAULT = 30
HOLE_RADIUS = 24
EDGE_MARGIN = 6
BOUNDARY_WIDTH = 10


def per_trial_metrics(g: pd.DataFrame, fps: int) -> Dict[str, float | int | str]:
    """
    Compute per (mouse_id, trial) metrics from a frame-level group.
    Assumes columns:
      - frame_idx, nose.x, nose.y, Hole.x, Hole.y, Centre.x, Centre.y
      - in_hole (bool), file, sex
    """
    # constants per video
    hx_series = g["Hole.x"].dropna() if "Hole.x" in g.columns else pd.Series(dtype=float)
    hy_series = g["Hole.y"].dropna() if "Hole.y" in g.columns else pd.Series(dtype=float)
    cx_series = g["Centre.x"].dropna() if "Centre.x" in g.columns else pd.Series(dtype=float)
    cy_series = g["Centre.y"].dropna() if "Centre.y" in g.columns else pd.Series(dtype=float)

    hx = float(hx_series.iloc[0]) if not hx_series.empty else np.nan
    hy = float(hy_series.iloc[0]) if not hy_series.empty else np.nan
    cx = float(cx_series.iloc[0]) if not cx_series.empty else np.nan
    cy = float(cy_series.iloc[0]) if not cy_series.empty else np.nan

    # distances
    if not np.isnan(hx):
        d_hole = np.sqrt((g["nose.x"] - hx) ** 2 + (g["nose.y"] - hy) ** 2)
    else:
        d_hole = pd.Series(np.nan, index=g.index)

    # boundary annulus
    in_boundary = (
        (d_hole.to_numpy() >= HOLE_RADIUS)
        & (d_hole.to_numpy() < HOLE_RADIUS + BOUNDARY_WIDTH)
    )

    # target quadrant: bottom-right of centre (example criterion)
    if not (np.isnan(cx) or np.isnan(cy)):
        in_quadrant = ((g["nose.x"].to_numpy() > cx) & (g["nose.y"].to_numpy() > cy)).astype(float)
        pct_in_quadrant = float(np.nanmean(in_quadrant) * 100.0)
    else:
        pct_in_quadrant = np.nan

    # poke frequency + latency
    in_hole = g["in_hole"].to_numpy() if "in_hole" in g.columns else np.zeros(len(g), dtype=bool)
    freq, latency = count_entries(in_hole, fps=fps, refractory_frames=10)

    # times (seconds)
    time_in_hole = float(np.nansum(in_hole) / fps)
    time_in_boundary = float(np.nansum(in_boundary) / fps)
    duration_s = float(len(g) / fps)

    sex = g["sex"].iloc[0] if "sex" in g.columns else None
    file = g["file"].iloc[0] if "file" in g.columns else None

    return {
        "sex": sex,
        "file": file,
        "duration_s": round(duration_s, 2),
        "poke_frequency": int(freq),
        "latency_first_poke_s": latency,
        "time_in_hole_s": round(time_in_hole, 3),
        "time_in_boundary_s": round(time_in_boundary, 3),
        "percent_in_quadrant": round(pct_in_quadrant, 2) if not np.isnan(pct_in_quadrant) else np.nan,
    }


def summarize(merged_imputed: Path, results_dir: Path, fps: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-trial table and roll-ups; save all CSVs."""
    df = pd.read_csv(merged_imputed)
    logging.info("Loaded %s rows from %s", f"{len(df):,}", merged_imputed)

    ensure_numeric(df, ["frame_idx", "nose.x", "nose.y", "Hole.x", "Hole.y", "Centre.x", "Centre.y"])
    df = df.sort_values(["mouse_id", "trial", "frame_idx"]).reset_index(drop=True)

    # per mouse×trial
    records = []
    for (mid, tr), g in df.groupby(["mouse_id", "trial"], dropna=False):
        g = g.copy()
        m = per_trial_metrics(g, fps=fps)
        m.update({"mouse_id": mid, "trial": tr})
        records.append(m)

    summary_id_trial = pd.DataFrame(records).sort_values(["mouse_id", "trial"]).reset_index(drop=True)
    logging.info("Per-mouse × trial summary (head):\n%s", summary_id_trial.head(10))

    # roll-ups
    def rollup(group_key: str) -> pd.DataFrame:
        agg = {
            "poke_frequency": ["mean", "sem", "count"],
            "latency_first_poke_s": ["mean", "sem", "count"],
            "time_in_hole_s": ["mean", "sem", "count"],
            "time_in_boundary_s": ["mean", "sem", "count"],
            "percent_in_quadrant": ["mean", "sem", "count"],
        }
        return summary_id_trial.groupby(group_key, dropna=True).agg(agg).reset_index()

    by_trial = rollup("trial")
    by_sex = rollup("sex")
    by_id = rollup("mouse_id")

    # save all
    save_csv(summary_id_trial, results_dir / "barnes_summary_id_trial.csv")
    save_csv(by_trial, results_dir / "barnes_summary_by_trial.csv")
    save_csv(by_sex, results_dir / "barnes_summary_by_sex.csv")
    save_csv(by_id, results_dir / "barnes_summary_by_id.csv")

    logging.info("Saved summaries to %s", results_dir)
    return summary_id_trial, by_trial, by_sex, by_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize imputed frame-level data.")
    parser.add_argument("--merged", type=Path, required=True, help="Path to merged_all_imputed.csv")
    parser.add_argument("--results", type=Path, required=True, help="Folder to save summary CSVs")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    summarize(args.merged, args.results, fps=args.fps)


if __name__ == "__main__":
    main()
