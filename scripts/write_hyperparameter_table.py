from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    config_path = Path("configs/default.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = ["| Hyperparameter | Value |", "| --- | --- |"]
    for key, value in config.items():
        rows.append(f"| `{key}` | `{value}` |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
