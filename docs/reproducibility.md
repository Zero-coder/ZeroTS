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

## Period-to-image construction

The 1D vector of length `L + O` is mapped to a `pod x columns` image by filling
columns with consecutive cycles of length `pod`. If `L + O` is not divisible by
`pod`, zeros are padded at the end. Padding cells are tracked and removed during
the inverse mapping.

Recommended `pod` values should be fixed from known dataset frequency whenever
possible. Context-only autocorrelation is provided as a fallback for exploratory
usage, but final benchmark experiments should report the fixed value used for each
dataset.

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

## Reconstruction backend

The fallback interpolation backend is provided only for pipeline checks. Accuracy
experiments should use a pretrained image-reconstruction backend and should report:

- checkpoint name and source
- input resolution
- patch size
- mask construction
- device
- inference batch size

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
