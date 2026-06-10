from __future__ import annotations

import argparse
import json
from pathlib import Path

from zerots import ZeroTSConfig, ZeroTSForecaster
from zerots.data import load_csv_series, sliding_windows
from zerots.metrics import mae, mse, normalized_mae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--context-length", type=int, default=336)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--pod", type=int, default=None)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("evaluation_summary.json"))
    args = parser.parse_args()

    series = load_csv_series(args.csv, value_column=args.value_column, time_column=args.time_column)
    cfg = ZeroTSConfig(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        pod=args.pod,
    )
    model = ZeroTSForecaster(cfg)

    rows = []
    for idx, window in enumerate(
        sliding_windows(series, args.context_length, args.prediction_length, stride=args.stride)
    ):
        if args.limit_windows is not None and idx >= args.limit_windows:
            break
        pred = model.forecast(window.context)
        rows.append(
            {
                "start": window.start,
                "mse": mse(pred, window.target),
                "mae": mae(pred, window.target),
                "normalized_mae": normalized_mae(pred, window.target),
            }
        )

    if not rows:
        raise RuntimeError("No evaluation windows were produced")

    summary = {
        "num_windows": len(rows),
        "mse": sum(row["mse"] for row in rows) / len(rows),
        "mae": sum(row["mae"] for row in rows) / len(rows),
        "normalized_mae": sum(row["normalized_mae"] for row in rows) / len(rows),
        "config": cfg.__dict__,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
