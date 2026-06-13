# Baseline Reproducibility Notes

This document records the baseline protocol that should be mirrored in the
revised manuscript and response letter.

## Evaluation groups

ZeroTS is evaluated as a deterministic zero-shot method. Baselines should be
reported by protocol instead of mixing them into a single setting:

| Group | Use of target training data | Examples |
| --- | --- | --- |
| Zero-shot / cross-domain | none during test evaluation | ZeroTS, MOIRAI |
| Few-shot | 15% downstream training data | TimeLLM, GPT4TS when fine-tuned |
| Full-shot reference | full target training split | DLinear, PatchTST, TimesNet, FECAM, FEDformer, Autoformer |

## LTSF protocol

- Prediction lengths: `96, 192, 336, 720`.
- Dataset splits: standard long-term forecasting train/validation/test splits.
- Hyperparameters are selected on the validation split and fixed for test
  evaluation.
- Test labels are not used for preprocessing, period selection, normalization, or
  hyperparameter selection.

## MOIRAI

- Use official pretrained MOIRAI checkpoints when reproducing results.
- Report the exact model scale used: Small, Base, Large, or an average over
  scales.
- If `MOIRAI-Avg` is reported, state that it averages the selected MOIRAI scales
  under the same dataset and horizon protocol.

## Prompt-based or LLM-based baselines

For LLMTime, GPT4TS, TimeLLM, or related methods, report:

- public codebase or official implementation used
- model checkpoint or language-model backbone
- prompt template or input serialization
- context length and prediction length
- whether the method is zero-shot or fine-tuned
- if few-shot, the percentage of target training data used and validation rule

## Supervised references

For DLinear, PatchTST, TimesNet, FECAM, FEDformer, and Autoformer, report:

- codebase used
- train/validation/test split
- context length and prediction length
- training epochs or early-stopping rule
- validation metric for model selection
- whether published numbers or reproduced numbers are used

## Reporting recommendation

When presenting results, label each baseline with its evaluation protocol. Use
few-shot and full-shot methods as references rather than as direct zero-shot
comparisons unless they use no target training data.
