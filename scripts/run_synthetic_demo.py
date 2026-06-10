from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zerots import ZeroTSConfig, ZeroTSForecaster
from zerots.metrics import mae, mse


def build_series(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    trend = 0.002 * t
    seasonal = np.sin(2.0 * np.pi * t / 24.0) + 0.4 * np.sin(2.0 * np.pi * t / 96.0)
    noise = 0.05 * rng.standard_normal(length)
    return (trend + seasonal + noise).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, default=336)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--pod", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("synthetic_demo_output.csv"))
    args = parser.parse_args()

    total = args.context_length + args.prediction_length
    series = build_series(total, args.seed)
    context = series[: args.context_length]
    target = series[args.context_length :]

    cfg = ZeroTSConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        pod=args.pod,
        random_seed=args.seed,
    )
    model = ZeroTSForecaster(cfg)
    prediction = model.forecast(context)

    rows = ["step,target,prediction"]
    for i, (true, pred) in enumerate(zip(target, prediction)):
        rows.append(f"{i},{float(true):.8f},{float(pred):.8f}")
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"mse={mse(prediction, target):.6f}")
    print(f"mae={mae(prediction, target):.6f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
