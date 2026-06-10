from __future__ import annotations

import numpy as np


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    return float(np.mean((pred - true) ** 2))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.abs(pred - true)))


def normalized_mae(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    scale = float(np.mean(np.abs(true - true.mean())))
    return mae(pred, true) / max(scale, eps)
