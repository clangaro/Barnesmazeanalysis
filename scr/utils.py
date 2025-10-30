---

# src/utils.py

```python
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


LOG_FORMAT = "[%(levelname)s] %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once."""
    logging.basicConfig(level=level, format=LOG_FORMAT)


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Convert columns to numeric in-place (coerce errors to NaN)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def broadcast_const(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a Series filled with the first non-NaN value of column `col`.
    If missing, return a NaN series with same length as df.
    """
    if col not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index)
    v = df[col].dropna()
    if v.empty:
        return pd.Series([math.nan] * len(df), index=df.index)
    return pd.Series([float(v.iloc[0])] * len(df), index=df.index)


def hysteresis_in(dist_vec: np.ndarray, r_enter: float, r_exit: float) -> np.ndarray:
    """
    Boolean in/out from distances with hysteresis:
    - enter when d <= r_enter
    - exit  when d >= r_exit
    Keeps the last state when distance is NaN.
    """
    state = False
    out = np.zeros(dist_vec.shape[0], dtype=bool)
    for i, d in enumerate(dist_vec):
        if np.isnan(d):
            out[i] = state
            continue
        if not state and d <= r_enter:
            state = True
        elif state and d >= r_exit:
            state = False
        out[i] = state
    return out


def count_entries(in_bool: np.ndarray, fps: int, refractory_frames: int = 10) -> Tuple[int, float]:
    """
    Count rising edges with a refractory period to avoid double counting.
    Returns (count, latency_seconds or NaN if none).
    """
    if in_bool.size < 2:
        return 0, math.nan
    rising = np.where((~in_bool[:-1]) & (in_bool[1:]))[0] + 1
    if rising.size == 0:
        return 0, math.nan
    kept = [int(rising[0])]
    for r in rising[1:]:
        if r - kept[-1] >= refractory_frames:
            kept.append(int(r))
    latency = kept[0] / float(fps)
    return len(kept), latency


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV safely with UTF-8 encoding and create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
