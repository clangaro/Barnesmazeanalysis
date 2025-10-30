from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.meta_extraction import process_folder as meta_process
from src.pad_frames import process_folder as pad_process
from src.impute_inside_hole import process_folder as impute_process
from src.summarize_trials import summarize
from src.utils import setup_logging


def merge_all_imputed(processed_dir: Path, results_dir: Path) -> Path:
    """Merge all *_imputed.csv in processed_dir into results_dir/merged_all_imputed.csv."""
    imputed = sorted(processed_dir.glob("*_imputed.csv"))
    logging.info("Found %d imputed CSVs.", len(imputed))
    dfs = [pd.read_csv(fp) for fp in imputed]
    merged = pd.concat(dfs, ignore_index=True)
    out = results_dir / "merged_all_imputed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    logging.info("Saved merged frame-level dataset to: %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Barnes Maze SLEAP pipeline.")
    parser.add_argument("--raw", type=Path, required=True, help="Folder with raw SLEAP files (.csv/.h5).")
    parser.add_argument("--processed", type=Path, required=True, help="Folder for processed CSVs.")
    parser.add_argument("--results", type=Path, required=True, help="Folder for results.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    raw = args.raw
    processed = args.processed
    results = args.results

    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    # 1) meta
    logging.info("=== Stage 1/5: metadata ===")
    meta_process(raw, processed, overwrite=True)

    # 2) pad to h5 frames
    logging.info("=== Stage 2/5: pad frames ===")
    pad_process(processed, raw, processed)

    # 3) impute inside hole
    logging.info("=== Stage 3/5: impute ===")
    impute_process(processed, processed, hole_radius=24, edge_margin=6)

    # 4) merge all imputed
    logging.info("=== Stage 4/5: merge ===")
    merged_path = merge_all_imputed(processed, results)

    # 5) summarize
    logging.info("=== Stage 5/5: summarize ===")
    summarize(merged_path, results, fps=30)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
