from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zerots import ZeroTSConfig, ZeroTSForecaster
from zerots.visualization import apsr_size_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=336)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--pod", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("apsr_patch_size_map.csv"))
    args = parser.parse_args()

    t = np.arange(args.context_length, dtype=np.float32)
    context = np.sin(2.0 * np.pi * t / 24.0)
    context[120:160] += 0.7 * np.sin(2.0 * np.pi * t[120:160] / 4.0)

    cfg = ZeroTSConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        pod=args.pod,
    )
    model = ZeroTSForecaster(cfg)
    _, traces = model.forecast(context, return_trace=True)
    size_map = apsr_size_map(
        traces[0].augmented_image,
        min_patch_size=cfg.min_patch_size,
        max_patch_size=cfg.max_patch_size,
    )
    np.savetxt(args.output, size_map, delimiter=",", fmt="%d")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
