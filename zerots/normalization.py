from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContextStats:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (self.std + self.eps)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * (self.std + self.eps) + self.mean


def fit_context_stats(values: np.ndarray, eps: float = 1e-6) -> ContextStats:
    """Fit normalization statistics from the observed context only."""

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        axis = 0
        keepdims = False
    elif arr.ndim == 2:
        axis = 0
        keepdims = False
    else:
        raise ValueError("values must be 1D or 2D")

    mean = arr.mean(axis=axis, keepdims=keepdims).astype(np.float32)
    std = arr.std(axis=axis, keepdims=keepdims).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return ContextStats(mean=mean, std=std, eps=eps)
