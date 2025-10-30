from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .utils import save_csv, setup_logging


# Sex rules (adjust to your cohort)
ID_FEMALE = range(101, 104)  # 101–103
ID_MALE = range(104, 107)    # 104–106

# Regex for SLEAP filename pattern
PAT = re.compile(
    r"""
    ^labels\.v\d+\.(?P<file_index>\d+)_   # labels.v001.<file_index>_
    (?P<mouse_id>\d+)\.                   # <mouse_id>.
    (?P<trial>\d+)                        # <trial>
    \.?_cfr\.analysis\.csv$               # optional extra dot + _cfr.analysis.csv
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_meta_from_name(basename: str) -> Optional[Tuple[int, int, int, str]]:
    """
    Parse file_index, mouse_id, trial, sex from a SLEAP filename.
    Returns None if the name does not match the expected pattern.
    """
    m = PAT.match(basename)
    if not m:
        return None
    file_index = int(m.group("file_index"))
    mouse_id = int(m.group("mouse_id"))
    trial = int(m.group("trial"))
    sex = (
        "Female" if mouse_id in ID_FEMALE
        else "Male" if mouse_id in ID_MALE
        else "Unknown"
    )
    return file_index, mouse_id, trial, sex


def process_folder(input_dir: Path, output_dir: Path, overwrite: bool = True) -> None:
    """
    Add metadata columns to CSVs in input_dir and write to output_dir.
    Metadata: file, file_index, mouse_id, trial, sex.
    """
    csvs = sorted(input_dir.glob("*.csv"))
    logging.info("Found %d CSV files in %s", len(csvs), input_dir)
    unmatched = []
    done = 0

    for fp in csvs:
        base = fp.name
        parsed = parse_meta_from_name(base)
        if parsed is None:
            unmatched.append(base)
            continue

        file_index, mouse_id, trial, sex = parsed
        df = pd.read_csv(fp)

        # Add/overwrite meta columns
        df["file"] = base
        df["file_index"] = file_index
        df["mouse_id"] = mouse_id
        df["trial"] = trial
        df["sex"] = sex

        out_fp = output_dir / base if overwrite else output_dir / (fp.stem + "_with_meta.csv")
        save_csv(df, out_fp)
        done += 1
        logging.info("meta added: %s -> %s", base, out_fp.name)

    logging.info("Completed: %d files updated.", done)
    if unmatched:
        logging.warning("These filenames did not match the expected pattern:")
        for name in unmatched:
            logging.warning("  - %s", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add metadata to SLEAP CSVs.")
    parser.add_argument("--input", type=Path, required=True, help="Folder with raw CSVs.")
    parser.add_argument("--output", type=Path, required=True, help="Folder to write CSVs with meta.")
    parser.add_argument("--no-overwrite", action="store_true", help="Write *_with_meta.csv instead of overwriting.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    process_folder(args.input, args.output, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
