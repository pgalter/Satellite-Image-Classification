"""Completion checks and atomic metadata writes for resumable experiments."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import config as C


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def completed_run(name: str, expected_config: dict | None = None,
                  save_model: bool = False) -> dict | None:
    path = C.RUNS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if expected_config is not None and result.get("config") != expected_config:
        raise ValueError(f"{name}: saved configuration differs; use a new --tag / "
                         "output directory, or --force to replace it")
    if not (C.PREDS_DIR / f"{name}.npz").exists():
        return None
    if save_model and not (C.RESULTS_DIR / "checkpoints" / f"{name}.pt").exists():
        return None
    return result
