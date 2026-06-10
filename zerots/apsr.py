from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PatchRegion:
    top: int
    left: int
    height: int
    width: int
    patch_size: int
    variation: float


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        base = arr[..., 0]
    elif arr.ndim == 2:
        base = arr
    else:
        raise ValueError("image must be HxW or HxWxC")
    gy = np.zeros_like(base)
    gx = np.zeros_like(base)
    gy[1:, :] = base[1:, :] - base[:-1, :]
    gx[:, 1:] = base[:, 1:] - base[:, :-1]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def local_variation_ratio(image: np.ndarray, window: int = 3) -> np.ndarray:
    mag = gradient_magnitude(image)
    pad = max(1, window // 2)
    padded = np.pad(mag, ((pad, pad), (pad, pad)), mode="edge")
    out = np.empty_like(mag)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            region = padded[i : i + 2 * pad + 1, j : j + 2 * pad + 1]
            out[i, j] = float(region.mean())
    mn = float(out.min())
    mx = float(out.max())
    if mx - mn < 1e-8:
        return np.zeros_like(out)
    return ((out - mn) / (mx - mn)).astype(np.float32)


def build_adaptive_regions(
    image: np.ndarray,
    min_patch_size: int,
    max_patch_size: int,
    overlap: float,
    variation_window: int = 3,
) -> List[PatchRegion]:
    if min_patch_size <= 0 or max_patch_size < min_patch_size:
        raise ValueError("invalid patch size range")
    ratio = local_variation_ratio(image, window=variation_window)
    h, w = ratio.shape
    regions: List[PatchRegion] = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            v = float(ratio[min(y, h - 1), min(x, w - 1)])
            size = int(round(max_patch_size - (max_patch_size - min_patch_size) * v))
            size = int(np.clip(size, min_patch_size, max_patch_size))
            height = min(size, h - y)
            width = min(size, w - x)
            regions.append(PatchRegion(y, x, height, width, size, v))
            step = max(1, int(round(size * (1.0 - overlap))))
            x += step
        step_y = max(1, int(round(max_patch_size * (1.0 - overlap))))
        y += step_y
    return regions


def adaptive_patch_smooth(
    image: np.ndarray,
    min_patch_size: int,
    max_patch_size: int,
    overlap: float,
    variation_window: int = 3,
) -> Tuple[np.ndarray, List[PatchRegion]]:
    """Apply a lightweight adaptive patch-resampling pass.

    High-variation regions keep more local detail. Low-variation regions are gently
    smoothed within larger regions, matching the intent of dynamic patch granularity.
    """

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("image must be HxWxC")

    regions = build_adaptive_regions(
        arr,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
        overlap=overlap,
        variation_window=variation_window,
    )
    acc = np.zeros_like(arr)
    weight = np.zeros(arr.shape[:2] + (1,), dtype=np.float32)
    for region in regions:
        patch = arr[
            region.top : region.top + region.height,
            region.left : region.left + region.width,
            :,
        ]
        mean_patch = patch.mean(axis=(0, 1), keepdims=True)
        detail_weight = np.clip(region.variation, 0.0, 1.0)
        mixed = detail_weight * patch + (1.0 - detail_weight) * mean_patch
        acc[
            region.top : region.top + region.height,
            region.left : region.left + region.width,
            :,
        ] += mixed
        weight[
            region.top : region.top + region.height,
            region.left : region.left + region.width,
            :,
        ] += 1.0
    return (acc / np.maximum(weight, 1.0)).astype(np.float32), regions


def patch_size_map(
    image: np.ndarray,
    min_patch_size: int,
    max_patch_size: int,
    variation_window: int = 3,
) -> np.ndarray:
    ratio = local_variation_ratio(image, window=variation_window)
    sizes = max_patch_size - (max_patch_size - min_patch_size) * ratio
    return np.rint(np.clip(sizes, min_patch_size, max_patch_size)).astype(np.int32)
