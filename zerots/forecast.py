from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .apsr import PatchRegion, adaptive_patch_smooth
from .backbones import InterpolationBackbone, ReconstructionBackbone
from .config import ZeroTSConfig
from .normalization import fit_context_stats
from .transforms import (
    build_masked_vector,
    ensure_2d_time_major,
    infer_period_from_context,
    period_image_to_vector,
    resize_bilinear,
    resize_mask,
    temporal_gradient_augmentation,
    vector_to_period_image,
)


@dataclass
class ForecastTrace:
    pod: int
    normalized_image: np.ndarray
    augmented_image: np.ndarray
    processed_image: np.ndarray
    resized_image: np.ndarray
    resized_mask: np.ndarray
    reconstructed_image: np.ndarray
    adaptive_regions: list[PatchRegion]


class ZeroTSForecaster:
    """ZeroTS forecasting pipeline.

    Multivariate inputs are handled independently by default. This deliberately
    keeps the leakage-free univariate pipeline explicit and reproducible.
    """

    def __init__(
        self,
        config: ZeroTSConfig,
        backbone: Optional[ReconstructionBackbone] = None,
    ) -> None:
        self.config = config
        self.backbone = backbone or InterpolationBackbone()

    def forecast(self, context: np.ndarray, return_trace: bool = False):
        arr = ensure_2d_time_major(context)
        if arr.shape[0] != self.config.context_length:
            raise ValueError(
                f"Expected context length {self.config.context_length}, got {arr.shape[0]}"
            )
        if self.config.multivariate_mode != "independent":
            raise NotImplementedError("Only independent multivariate mode is implemented")

        forecasts = []
        traces = []
        for idx in range(arr.shape[1]):
            pred, trace = self._forecast_univariate(arr[:, idx])
            forecasts.append(pred)
            traces.append(trace)
        result = np.stack(forecasts, axis=1)
        if context.ndim == 1:
            result = result[:, 0]
        if return_trace:
            return result.astype(np.float32), traces
        return result.astype(np.float32)

    def _forecast_univariate(self, context: np.ndarray) -> Tuple[np.ndarray, ForecastTrace]:
        cfg = self.config
        stats = fit_context_stats(context, eps=cfg.eps)
        normalized_context = stats.normalize(context.astype(np.float32))

        pod = cfg.pod or infer_period_from_context(
            normalized_context,
            max_lag=cfg.infer_pod_max_lag,
        )
        masked_vector, future_mask = build_masked_vector(
            normalized_context,
            prediction_length=cfg.prediction_length,
            fill_value=0.0,
        )
        normalized_image, info = vector_to_period_image(masked_vector, future_mask, pod=pod)
        augmented = temporal_gradient_augmentation(
            normalized_image,
            window=cfg.tga_window,
            mode=cfg.tga_mode,
        )
        processed, regions = adaptive_patch_smooth(
            augmented,
            min_patch_size=cfg.min_patch_size,
            max_patch_size=cfg.max_patch_size,
            overlap=cfg.apsr_overlap,
            variation_window=cfg.tga_window,
        )

        resized = resize_bilinear(processed, (cfg.image_size, cfg.image_size))
        resized_mask = resize_mask(info.future_mask, (cfg.image_size, cfg.image_size))
        reconstructed_resized = self.backbone.reconstruct(resized, resized_mask)

        reconstructed_period = resize_bilinear(reconstructed_resized, normalized_image.shape)
        reconstructed_scalar = reconstructed_period[..., 0]
        reconstructed_vector = period_image_to_vector(reconstructed_scalar, info)
        normalized_future = reconstructed_vector[-cfg.prediction_length :]
        forecast = stats.denormalize(normalized_future)

        trace = ForecastTrace(
            pod=pod,
            normalized_image=normalized_image,
            augmented_image=augmented,
            processed_image=processed,
            resized_image=resized,
            resized_mask=resized_mask,
            reconstructed_image=reconstructed_resized,
            adaptive_regions=regions,
        )
        return forecast.astype(np.float32), trace
