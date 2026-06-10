from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ZeroTSConfig:
    """Configuration for the ZeroTS forecasting pipeline."""

    context_length: int
    prediction_length: int
    pod: Optional[int] = None
    image_size: int = 224
    patch_size: int = 16
    min_patch_size: int = 4
    max_patch_size: int = 16
    tga_window: int = 3
    tga_mode: Literal["weighted_rgb", "separate_channels"] = "weighted_rgb"
    apsr_overlap: float = 0.1
    interpolation: Literal["bilinear"] = "bilinear"
    eps: float = 1e-6
    multivariate_mode: Literal["independent"] = "independent"
    normalize_per_variable: bool = True
    infer_pod_max_lag: int = 96
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.min_patch_size <= 0:
            raise ValueError("min_patch_size must be positive")
        if self.max_patch_size < self.min_patch_size:
            raise ValueError("max_patch_size must be >= min_patch_size")
        if not 0.0 <= self.apsr_overlap < 1.0:
            raise ValueError("apsr_overlap must be in [0, 1)")
        if self.pod is not None and self.pod <= 0:
            raise ValueError("pod must be positive when provided")
