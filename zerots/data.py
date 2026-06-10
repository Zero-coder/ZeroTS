from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass
class Window:
    context: np.ndarray
    target: np.ndarray
    start: int


def sliding_windows(
    values: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: Optional[int] = None,
) -> Iterator[Window]:
    arr = np.asarray(values, dtype=np.float32)
    stride = stride or prediction_length
    total = context_length + prediction_length
    for start in range(0, arr.shape[0] - total + 1, stride):
        yield Window(
            context=arr[start : start + context_length],
            target=arr[start + context_length : start + total],
            start=start,
        )


def load_csv_series(path: str, value_column: str, time_column: Optional[str] = None) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Install pandas or the 'data' extra to load CSV files") from exc

    frame = pd.read_csv(path)
    if time_column:
        frame = frame.sort_values(time_column)
    if value_column not in frame.columns:
        raise ValueError(f"Column not found: {value_column}")
    return frame[value_column].to_numpy(dtype=np.float32)


def stack_metric_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for key in keys:
        vals = [row[key] for row in rows]
        if isinstance(vals[0], (int, float, np.floating)):
            out[key] = float(np.mean(vals))
        else:
            out[key] = vals[0]
    return out
