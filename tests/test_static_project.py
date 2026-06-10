from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_files_do_not_use_disallowed_project_name() -> None:
    forbidden = ("vision" + "ts", "vision" + " ts")
    for path in (ROOT / "zerots").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(term in text for term in forbidden), path
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(term in text for term in forbidden), path


def test_required_entrypoints_exist() -> None:
    required = [
        ROOT / "zerots" / "forecast.py",
        ROOT / "zerots" / "transforms.py",
        ROOT / "zerots" / "apsr.py",
        ROOT / "scripts" / "evaluate_csv.py",
        ROOT / "scripts" / "run_synthetic_demo.py",
    ]
    for path in required:
        assert path.exists(), path
