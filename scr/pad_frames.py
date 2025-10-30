from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .utils import save_csv, setup_logging


def h5_frames(h5_path: Path) -> int:
    """Return total number of frames from a SLEAP .h5 file."""
    with h5py.File(h5_path, "r") as f:
        if "tracks" in f:
            arr = np.squeeze(f["tracks"][:])
            return int(arr.shape[-1])  # last dim is frames
        if "frame_count" in f:
            return int(f["frame_count"][()])
    raise RuntimeError(f"Cannot infer total frames from {h5_path.name}")


def pad_csv_to_frames(csv_path: Path, h5_dir: Path, out_dir: Path) -> None:
    """
    Pad the CSV to full frame range [0..T-1] using the paired .h5 file in `h5_dir`.
    The paired .h5 is expected to share the same stem.
    """
    stem = csv_path.stem
    h5_guess = h5_dir / (stem.replace("_cfr.analysis", "") + ".h5")
    if not h5_guess.exists():
        logging.warning("SKIP (no H5): %s -> expected %s", csv_path.name, h5_guess.name)
        return

    try:
        total = h5_frames(h5_guess)
    except Exception as exc:
        logging.warning("SKIP (cannot read frames from %s): %s", h5_guess.name, exc)
        return

    df = pd.read_csv(csv_path)
    if "frame_idx" not in df.columns:
        logging.warning("SKIP (no frame_idx col): %s", csv_path.name)
        return

    full = pd.DataFrame({"frame_idx": np.arange(total, dtype=int)})
    dfp = full.merge(df, on="frame_idx", how="left")

    out = out_dir / (stem + "_padded.csv")
    save_csv(dfp, out)
    logging.info("Padded %s -> %s (rows %d -> %d, frames=%d)",
                 csv_path.name, out.name, len(df), len(dfp), total)


def process_folder(csv_dir: Path, h5_dir: Path, out_dir: Path) -> None:
    csv_files = sorted(csv_dir.glob("*.csv"))
    logging.info("Found %d CSVs to pad in: %s", len(csv_files), csv_dir)
    for fp in csv_files:
        if fp.name.endswith("_padded.csv"):
            continue
        pad_csv_to_frames(fp, h5_dir, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pad SLEAP CSVs to full frame count.")
    parser.add_argument("--csv", type=Path, required=True, help="Folder with CSVs (with meta).")
    parser.add_argument("--h5", type=Path, required=True, help="Folder with SLEAP .h5 files.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for *_padded.csv.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    process_folder(args.csv, args.h5, args.output)


if __name__ == "__main__":
    main()
