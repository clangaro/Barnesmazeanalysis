from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import broadcast_const, ensure_numeric, hysteresis_in, save_csv, setup_logging


# Defaults can be overridden via CLI
HOLE_RADIUS_DEFAULT = 24      # px, “in hole” if dist <= R
EDGE_MARGIN_DEFAULT = 6       # px, hysteresis margin
FPS_DEFAULT = 30


def interp_small_gaps(df: pd.DataFrame, cols: Tuple[str, str] = ("nose.x", "nose.y"), limit: int | None = None) -> pd.DataFrame:
    """
    Convert target columns to numeric, interpolate small NaN gaps (linear),
    and ffill/bfill remaining edges.
    """
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cols:
        out[c] = out[c].interpolate(limit=limit).ffill().bfill()
    return out


def impute_inside_hole(
    df: pd.DataFrame,
    hole_xy: Tuple[float, float],
    r_in: float,
    r_margin: float,
) -> pd.DataFrame:
    """
    Impute missing nose coordinates for runs bracketed by in-hole frames.
    - Compute distance to hole centre (hx, hy)
    - Hysteresis: enter when d <= r_in; exit when d >= r_in + r_margin
    - Identify runs of missing nose.x/nose.y
    - If run is bracketed by in_hole==True both sides, fill run with (hx, hy)
    """
    hx, hy = hole_xy
    out = df.copy()

    # distances and availability
    dist = np.sqrt((out["nose.x"] - hx) ** 2 + (out["nose.y"] - hy) ** 2)
    avail = out["nose.x"].notna() & out["nose.y"].notna()

    r_out = r_in + r_margin
    in_hole = hysteresis_in(dist.to_numpy(), r_in, r_out)

    # mark missing and impute by bracket logic
    miss = ~avail
    imputed = np.zeros(len(out), dtype=bool)

    if miss.any():
        run_id = (miss.ne(miss.shift(fill_value=False))).cumsum()
        for _, idx in out[miss].groupby(run_id[miss]).indices.items():
            start = min(idx)
            end = max(idx)
            prev_i = start - 1 if start > out.index.min() else None
            next_i = end + 1 if end < out.index.max() else None
            cond_prev = (prev_i is not None) and in_hole[prev_i]
            cond_next = (next_i is not None) and in_hole[next_i]
            if cond_prev and cond_next:
                out.loc[start:end, ["nose.x", "nose.y"]] = (hx, hy)
                in_hole[start:end] = True
                imputed[start:end] = True

    out["in_hole"] = in_hole
    out["imputed_hole"] = imputed
    return out


def process_folder(input_dir: Path, output_dir: Path, hole_radius: float, edge_margin: float) -> None:
    files = sorted(input_dir.glob("*_padded.csv"))
    logging.info("Found %d padded CSVs.", len(files))

    for fp in files:
        df = pd.read_csv(fp)
        if not {"frame_idx", "nose.x", "nose.y"}.issubset(df.columns):
            logging.warning("skip (missing cols): %s", fp.name)
            continue

        # broadcast per-video hole coords (first non-NaN)
        hole_x = broadcast_const(df, "Hole.x")
        hole_y = broadcast_const(df, "Hole.y")

        if hole_x.isna().all() or hole_y.isna().all():
            logging.warning("No hole coords in %s — please label/export.", fp.name)
            continue

        hx, hy = float(hole_x.iloc[0]), float(hole_y.iloc[0])

        # 1) small gap interpolation
        dfi = interp_small_gaps(df, cols=("nose.x", "nose.y"))
        # 2) impute inside-hole missing runs
        out = impute_inside_hole(dfi, (hx, hy), hole_radius, edge_margin)

        out_path = output_dir / fp.name.replace("_padded.csv", "_imputed.csv")
        save_csv(out, out_path)
        logging.info("imputed -> %s (added %d frames)", out_path.name, int(out["imputed_hole"].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Impute inside-hole missing runs with hysteresis.")
    parser.add_argument("--input", type=Path, required=True, help="Folder with *_padded.csv")
    parser.add_argument("--output", type=Path, required=True, help="Folder for *_imputed.csv")
    parser.add_argument("--hole-radius", type=float, default=HOLE_RADIUS_DEFAULT)
    parser.add_argument("--edge-margin", type=float, default=EDGE_MARGIN_DEFAULT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    process_folder(args.input, args.output, args.hole_radius, args.edge_margin)


if __name__ == "__main__":
    main()
