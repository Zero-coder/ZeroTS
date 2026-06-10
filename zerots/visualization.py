from __future__ import annotations

import numpy as np

from .apsr import patch_size_map


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        scaled = np.zeros_like(arr)
    else:
        scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def apsr_size_map(image: np.ndarray, min_patch_size: int, max_patch_size: int) -> np.ndarray:
    return patch_size_map(
        image,
        min_patch_size=min_patch_size,
        max_patch_size=max_patch_size,
    )
