"""SIFT threshold choices belong to one train/validation/test split only."""
from __future__ import annotations

import json

from . import config as C


def tuning_path(train_frac: float, seed: int):
    return C.TABLES_DIR / f"bovw_tuning-tf{int(round(train_frac * 100)):02d}-s{seed}.json"


def selected_threshold(train_frac: float, seed: int) -> float:
    path = tuning_path(train_frac, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split-specific tuning: {path}. Run scripts/02a_tune_bovw.py "
            f"--fracs {train_frac:g} --seeds {seed} first. Do not reuse the 80/20 seed-0 "
            "choice on other splits: their test images can overlap its tuning data.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("train_frac") != train_frac or result.get("seed") != seed:
        raise ValueError(f"Tuning provenance does not match the requested split: {path}")
    return float(result["selected_contrast_threshold"])
