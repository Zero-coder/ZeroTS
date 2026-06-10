from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class ReconstructionBackbone(Protocol):
    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return a reconstructed HxWxC image."""


@dataclass
class InterpolationBackbone:
    """A deterministic fallback backend for pipeline validation.

    It fills masked regions by extending the nearest visible column and applying a
    small moving average. This is not a replacement for a pretrained image model;
    it exists so preprocessing, masking, and inverse mapping can be tested without
    external checkpoints.
    """

    smoothing_window: int = 3

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32).copy()
        m = np.asarray(mask, dtype=bool)
        if arr.ndim != 3:
            raise ValueError("image must be HxWxC")
        if m.shape != arr.shape[:2]:
            raise ValueError("mask shape must match image spatial shape")

        visible_cols = np.where(~m.all(axis=0))[0]
        if visible_cols.size == 0:
            return arr
        last_visible = int(visible_cols[-1])
        for col in range(arr.shape[1]):
            if m[:, col].any():
                source = min(last_visible, col)
                arr[m[:, col], col, :] = arr[m[:, col], source, :]
        return self._smooth(arr, m)

    def _smooth(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.smoothing_window <= 1:
            return image
        pad = self.smoothing_window // 2
        padded = np.pad(image, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        out = image.copy()
        for col in range(image.shape[1]):
            window = padded[:, col : col + self.smoothing_window, :]
            out[:, col, :] = window.mean(axis=1)
        return np.where(mask[..., None], out, image).astype(np.float32)


@dataclass
class CallableBackbone:
    """Adapter around a user-supplied reconstruction callable."""

    fn: object

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = self.fn(image, mask)
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape != image.shape:
            raise ValueError("backbone output must have the same shape as input image")
        return arr


@dataclass
class HFMaskedAutoencoderBackbone:
    """Optional adapter for HuggingFace ViTMAE-style models."""

    model_name: str = "facebook/vit-mae-base"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, ViTMAEForPreTraining
        except ImportError as exc:
            raise ImportError(
                "Install the 'hf' extra to use HFMaskedAutoencoderBackbone"
            ) from exc

        self._torch = torch
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = ViTMAEForPreTraining.from_pretrained(self.model_name)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        torch = self._torch
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("HF backend expects HxWx3 image")

        scaled = _to_uint_image(arr)
        inputs = self.processor(images=scaled, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        bool_masked_pos = _spatial_mask_to_patch_mask(mask, self.model.config.patch_size)
        bool_masked_pos = torch.tensor(bool_masked_pos[None, :], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        reconstructed = self.model.unpatchify(outputs.logits).detach().cpu()
        reconstructed = reconstructed[0].permute(1, 2, 0).numpy().astype(np.float32)
        reconstructed = _from_model_scale(reconstructed)
        if reconstructed.shape[:2] != arr.shape[:2]:
            from .transforms import resize_bilinear

            reconstructed = resize_bilinear(reconstructed, arr.shape[:2])
        return np.where(mask[..., None], reconstructed, arr).astype(np.float32)


def _to_uint_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < 1e-8:
        scaled = np.zeros_like(arr)
    else:
        scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _from_model_scale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 2.0:
        arr = arr / 255.0
    return arr * 2.0 - 1.0


def _spatial_mask_to_patch_mask(mask: np.ndarray, patch_size: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    ph = h // patch_size
    pw = w // patch_size
    if ph == 0 or pw == 0:
        raise ValueError("mask is smaller than one patch")
    cropped = m[: ph * patch_size, : pw * patch_size]
    patch_mask = cropped.reshape(ph, patch_size, pw, patch_size).mean(axis=(1, 3)) > 0.5
    return patch_mask.reshape(-1)
