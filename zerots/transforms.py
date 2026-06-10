from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ReshapeInfo:
    pod: int
    total_length: int
    padded_length: int
    columns: int
    valid_mask: np.ndarray
    future_mask: np.ndarray


def ensure_2d_time_major(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected a 1D or 2D array")


def infer_period_from_context(values: np.ndarray, max_lag: int = 96) -> int:
    """Infer a period using context-only autocorrelation.

    The fallback is deliberately conservative. If no reliable lag is found, it
    returns the largest divisor-like lag bounded by the context length.
    """

    x = np.asarray(values, dtype=np.float32).reshape(-1)
    n = x.shape[0]
    if n < 4:
        return max(1, n)

    upper = max(2, min(max_lag, n // 2))
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 1e-8:
        return min(upper, n)

    best_lag = 1
    best_score = -np.inf
    for lag in range(2, upper + 1):
        a = centered[:-lag]
        b = centered[lag:]
        score = float(np.dot(a, b)) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8)
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag)


def build_masked_vector(
    context: np.ndarray,
    prediction_length: int,
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    context = np.asarray(context, dtype=np.float32).reshape(-1)
    future = np.full(prediction_length, fill_value, dtype=np.float32)
    vector = np.concatenate([context, future], axis=0)
    future_mask = np.zeros(vector.shape[0], dtype=bool)
    future_mask[context.shape[0] :] = True
    return vector, future_mask


def vector_to_period_image(
    vector: np.ndarray,
    future_mask: np.ndarray,
    pod: int,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, ReshapeInfo]:
    """Map a 1D sequence to a period-by-column image.

    Values are filled column-wise so each column represents one cycle of length
    `pod`. Padding cells are marked invalid and ignored during inverse mapping.
    """

    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    future_mask = np.asarray(future_mask, dtype=bool).reshape(-1)
    if vector.shape[0] != future_mask.shape[0]:
        raise ValueError("vector and future_mask must have the same length")
    if pod <= 0:
        raise ValueError("pod must be positive")

    total_length = vector.shape[0]
    columns = int(np.ceil(total_length / pod))
    padded_length = columns * pod
    pad = padded_length - total_length

    padded = np.pad(vector, (0, pad), constant_values=pad_value)
    padded_future = np.pad(future_mask, (0, pad), constant_values=False)
    valid_mask = np.zeros(padded_length, dtype=bool)
    valid_mask[:total_length] = True

    image = padded.reshape(columns, pod).T
    valid_image = valid_mask.reshape(columns, pod).T
    future_image = padded_future.reshape(columns, pod).T

    info = ReshapeInfo(
        pod=pod,
        total_length=total_length,
        padded_length=padded_length,
        columns=columns,
        valid_mask=valid_image,
        future_mask=future_image,
    )
    return image.astype(np.float32), info


def period_image_to_vector(image: np.ndarray, info: ReshapeInfo) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape[:2] != (info.pod, info.columns):
        raise ValueError("image shape does not match reshape metadata")
    vector = arr.T.reshape(-1)
    return vector[: info.total_length].astype(np.float32)


def temporal_gradient_augmentation(
    image: np.ndarray,
    window: int = 3,
    mode: str = "weighted_rgb",
) -> np.ndarray:
    """Create TGA channels from original, first, and second gradients.

    The default follows the manuscript equations: each component receives a
    local/global variance weight, the weighted signal is fused, and the result is
    copied into three image channels. `separate_channels` is useful for ablations.
    """

    x = np.asarray(image, dtype=np.float32)
    first = np.zeros_like(x)
    first[1:, :] = x[1:, :] - x[:-1, :]
    second = np.zeros_like(x)
    second[1:, :] = first[1:, :] - first[:-1, :]
    components = np.stack([x, first, second], axis=0).astype(np.float32)
    if mode == "separate_channels":
        return np.moveaxis(components, 0, -1).astype(np.float32)
    if mode != "weighted_rgb":
        raise ValueError("mode must be 'weighted_rgb' or 'separate_channels'")

    weights = np.stack(
        [_local_global_variance_weight(component, window=window) for component in components],
        axis=0,
    )
    fused = (weights * components).sum(axis=0) / (weights.sum(axis=0) + 1e-6)
    return np.repeat(fused[:, :, None], 3, axis=2).astype(np.float32)


def _local_global_variance_weight(component: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(component, dtype=np.float32)
    window = max(1, int(window))
    pad = window // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    local_var = np.empty_like(arr)
    for i in range(arr.shape[0]):
        region = padded[i : i + window, :]
        local_var[i, :] = region.var(axis=0)
    global_var = float(arr.var()) + 1e-6
    ratio = local_var / global_var
    mn = float(ratio.min())
    mx = float(ratio.max())
    if mx - mn < 1e-8:
        return np.ones_like(arr, dtype=np.float32)
    return ((ratio - mn) / (mx - mn)).astype(np.float32)


def resize_bilinear(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Numpy bilinear resize for HWC arrays."""

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError("image must be HxW or HxWxC")

    out_h, out_w = target_hw
    in_h, in_w, channels = arr.shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("target size must be positive")
    if in_h == out_h and in_w == out_w:
        return arr.copy()

    y = np.linspace(0, max(in_h - 1, 0), out_h)
    x = np.linspace(0, max(in_w - 1, 0), out_w)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    out = np.empty((out_h, out_w, channels), dtype=np.float32)
    for i in range(out_h):
        top = (1.0 - wx)[:, None] * arr[y0[i], x0, :] + wx[:, None] * arr[y0[i], x1, :]
        bottom = (1.0 - wx)[:, None] * arr[y1[i], x0, :] + wx[:, None] * arr[y1[i], x1, :]
        out[i, :, :] = (1.0 - wy[i]) * top + wy[i] * bottom
    return out.astype(np.float32)


def resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    resized = resize_bilinear(mask.astype(np.float32), target_hw)
    return resized[..., 0] >= 0.5
