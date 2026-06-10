# ZeroTS

ZeroTS is a zero-shot time-series forecasting codebase that reformulates forecasting
as an image-reconstruction style pipeline. The implementation focuses on leakage-free
preprocessing, temporal gradient augmentation, adaptive patch selection and
resampling, explicit hyperparameters, and reproducible evaluation scripts.

## What is included

- `zerots.config.ZeroTSConfig`: all core hyperparameters in one dataclass.
- `zerots.forecast.ZeroTSForecaster`: end-to-end forecasting pipeline.
- Context-only normalization: statistics are computed from the observed history
  only, never from the future prediction window.
- TGA: original, first-order gradient, and second-order gradient channels.
- APSR: adaptive patch-size map and gradient-aware smoothing/resampling.
- Pluggable reconstruction backend:
  - `InterpolationBackbone` for pipeline tests and sanity checks.
  - `HFMaskedAutoencoderBackbone` for HuggingFace ViTMAE-style checkpoints.
  - `CallableBackbone` for custom pretrained image-reconstruction models.
- CSV evaluation and synthetic demo scripts.

## Install

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e ".[hf,data,plot,dev]"
```

## Quick demo

```bash
python scripts/run_synthetic_demo.py --context-length 336 --prediction-length 96
```

This runs the full preprocessing, masking, reconstruction, and inverse-mapping
pipeline with the deterministic interpolation backend. It is intended as a sanity
check. For paper-scale experiments, use a pretrained image-reconstruction backend.

## CSV evaluation

```bash
python scripts/evaluate_csv.py \
  --csv path/to/series.csv \
  --value-column value \
  --context-length 336 \
  --prediction-length 96 \
  --stride 96
```

If your CSV has a timestamp column:

```bash
python scripts/evaluate_csv.py \
  --csv path/to/series.csv \
  --time-column date \
  --value-column value \
  --context-length 336 \
  --prediction-length 96
```

## Using a pretrained reconstruction backend

```python
from zerots import ZeroTSConfig, ZeroTSForecaster
from zerots.backbones import HFMaskedAutoencoderBackbone

cfg = ZeroTSConfig(context_length=336, prediction_length=96, pod=24)
backbone = HFMaskedAutoencoderBackbone(model_name="facebook/vit-mae-base")
model = ZeroTSForecaster(cfg, backbone=backbone)
forecast = model.forecast(context)
```

You can also pass any callable with the signature:

```python
def reconstruct(image, mask):
    return reconstructed_image
```

and wrap it with `CallableBackbone`.

## Reproducibility notes

- Normalization is fitted on the historical context only.
- The default multivariate mode forecasts each variable independently.
- `pod` can be fixed from the dataset frequency or inferred from context-only
  autocorrelation. For revised experiments, prefer fixed dataset-frequency values
  and report them in the hyperparameter table.
- If the sequence length is not divisible by `pod`, the implementation pads only
  after the available context plus future placeholders. Padding cells are tracked
  and removed during inverse mapping.
- The deterministic backend is not the method used for final accuracy claims. It
  exists to verify preprocessing and reconstruction mechanics without external
  checkpoints.

## Suggested hyperparameter table fields

Report the following fields in the manuscript or appendix:

| Field | Code argument |
| --- | --- |
| Context length | `context_length` |
| Prediction length | `prediction_length` |
| Period/rows | `pod` |
| Image resolution | `image_size` |
| MAE patch size | `patch_size` |
| Minimum adaptive patch size | `min_patch_size` |
| Maximum adaptive patch size | `max_patch_size` |
| TGA/local variation window | `tga_window` |
| TGA mode | `tga_mode` |
| APSR overlap ratio | `apsr_overlap` |
| Interpolation | `interpolation` |
| Multivariate handling | `multivariate_mode` |
| Reconstruction checkpoint | backend-specific |
