# Reproducibility Protocol

This document summarizes the implementation choices that should be mirrored in the
paper, appendix, and response letter.

## Leakage-free preprocessing

For each forecasting window, the input is split into:

- historical context: length `L`
- prediction horizon: length `O`

Normalization statistics are fitted only on the historical context. The future
region is initialized with zeros after context normalization and is marked as masked.
The future ground truth is never used for normalization, period selection,
resampling, reconstruction, or hyperparameter selection.

ZeroTS is evaluated as deterministic zero-shot inference. The vision backbone is
frozen, no task-specific training is performed, and no random initialization is
introduced in the main forecasting path. Random seeds are only relevant for
stochastic auxiliary analyses, such as t-SNE visualizations, random-encoder
controls, or trainable probing experiments.

## Period-to-image construction

The 1D vector of length `L + O` is mapped to a `pod x columns` image by filling
columns with consecutive cycles of length `pod`. If `L + O` is not divisible by
`pod`, zeros are padded at the end. Padding cells are tracked and removed during
the inverse mapping.

Recommended `pod` values should be fixed from known dataset frequency whenever
possible. Context-only autocorrelation is provided as a fallback for exploratory
usage, but final benchmark experiments should report the fixed value used for each
dataset.

Recommended LTSF presets:

| Dataset | Sampling interval | `pod` |
| --- | --- | --- |
| ETTh1 | 1 hour | 24 |
| ETTh2 | 1 hour | 24 |
| ETTm1 | 15 minutes | 96 |
| ETTm2 | 15 minutes | 96 |
| Electricity | 1 hour | 24 |
| Traffic | 1 hour | 24 |
| Weather | 10 minutes | 144 |
| Exchange | 1 day | 7 |
| ILI | 1 week | 52 |

## Multivariate handling

The default implementation forecasts each variable independently. This avoids
mixing cross-variable information into the univariate image reconstruction pipeline.
If this setting is used in experiments, the paper should state it clearly and discuss
the lack of explicit cross-variable dependency modeling as a limitation.

## TGA

TGA computes:

- original normalized signal
- first-order temporal gradient
- second-order temporal gradient

The default mode computes local/global variance weights for each component, fuses
the weighted signal, and repeats it into three image channels. The
`separate_channels` mode keeps the three components in separate channels and can be
used for ablations.

## APSR

APSR estimates local temporal variation from gradient magnitude. High-variation
regions receive smaller effective patch sizes, while stable regions receive larger
patch sizes. The implementation exposes the patch-size map so that the manuscript
can include a figure showing narrow patches over rapid fluctuations and wider
patches over stable trends.

Default revised settings:

- minimum patch size `s_min = 4`
- maximum patch size `s_max = 16`
- overlap ratio `gamma = 0.1`
- local variation window `T_w = 3`
- resizing interpolation: bilinear

## Reconstruction backend

The fallback interpolation backend is provided only for pipeline checks. Accuracy
experiments should use a pretrained image-reconstruction backend and should report:

- checkpoint name and source
- input resolution
- patch size
- mask construction
- device
- inference batch size

The default MAE-style backend adapter in this repository supports HuggingFace
ViTMAE checkpoints such as `facebook/vit-mae-base`.

## Inference algorithm

For each univariate series:

1. Fit mean and standard deviation on the historical context only.
2. Normalize the context using those statistics.
3. Append zero placeholders for the prediction horizon.
4. Construct a binary future mask over the placeholder region.
5. Reshape the length `L + O` vector into a `pod x columns` image.
6. Pad the final column if needed and record a validity mask.
7. Apply TGA to add temporal-gradient information.
8. Apply APSR to adapt local patch granularity.
9. Resize the representation and mask to the vision backbone resolution.
10. Reconstruct the masked future region with the image-reconstruction backend.
11. Map the reconstructed image back to a 1D vector.
12. Remove padding, keep the final `O` values, and de-normalize.

## Statistical reporting

For deterministic zero-shot inference, seed-based standard deviations are not
meaningful for the main ZeroTS results. For paper reporting, paired comparisons
across matched datasets or dataset-horizon pairs are recommended. The revised
manuscript uses paired Wilcoxon signed-rank tests for key comparisons.

## Hyperparameters to report

| Field | Code argument |
| --- | --- |
| Context length | `context_length` |
| Prediction length | `prediction_length` |
| Period/rows | `pod` |
| Image resolution | `image_size` |
| Patch size | `patch_size` |
| Minimum adaptive patch size | `min_patch_size` |
| Maximum adaptive patch size | `max_patch_size` |
| TGA window | `tga_window` |
| TGA mode | `tga_mode` |
| APSR overlap | `apsr_overlap` |
| Interpolation | `interpolation` |
| Multivariate mode | `multivariate_mode` |
| Reconstruction checkpoint | backend-specific |

Default values used by `configs/default.json`:

| Field | Value |
| --- | --- |
| Context length | 336 |
| Prediction length | 96 |
| Period/rows | 24 |
| Image resolution | 224 |
| Patch size | 16 |
| Minimum adaptive patch size | 4 |
| Maximum adaptive patch size | 16 |
| TGA window | 3 |
| TGA mode | weighted_rgb |
| APSR overlap | 0.1 |
| Interpolation | bilinear |
| Multivariate mode | independent |
| Random seed | 2026 |
