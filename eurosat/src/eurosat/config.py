"""Paths, dataset constants and train/validation/test split settings.

Paper being reproduced:
    P. Helber, B. Bischke, A. Dengel, D. Borth,
    "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
    Land Cover Classification", arXiv:1709.00029v2 (IEEE JSTARS 2019).
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded archive
IMG_DIR = DATA_DIR / "EuroSAT_RGB"    # extracted JPEGs (one folder per class)
CACHE_DIR = DATA_DIR / "cache"        # decoded arrays / cached features
SPLIT_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"       # one JSON per experiment
PREDS_DIR = RESULTS_DIR / "preds"     # test-set predictions (npz) per experiment
TABLES_DIR = RESULTS_DIR / "tables"
LOG_DIR = ROOT / "logs"

# ------------------------------------------------------------------------- dataset
# Official record published by the paper's authors (DOI 10.5281/zenodo.7711810, MIT licence).
ZENODO_RECORD = "https://zenodo.org/records/7711810"
ZENODO_URL = "https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content"
ARCHIVE_NAME = "EuroSAT_RGB.zip"
ARCHIVE_MD5 = "f46e308c4d50d4bf32fedad2d3d62f3b"   # as listed on the Zenodo record
ARCHIVE_SIZE = 94_658_721                          # bytes

IMG_SIZE = 64
N_IMAGES = 27_000

# Folder names in the archive, alphabetical = label ids 0..9 (same order as Fig. 8 of the paper).
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
EXPECTED_COUNTS = {
    "AnnualCrop": 3000, "Forest": 3000, "HerbaceousVegetation": 3000, "Highway": 2500,
    "Industrial": 2500, "Pasture": 2000, "PermanentCrop": 2500, "Residential": 3000,
    "River": 2500, "SeaLake": 3000,
}
N_CLASSES = len(CLASSES)

# ImageNet statistics (inputs of ImageNet-pretrained networks), on a 0..1 scale.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------------------ protocol
TRAIN_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # the nine splits of Table II
VAL_FRAC_OF_TRAIN = 0.10   # share of each training portion held out for model selection
DEFAULT_SEED = 0


def split_name(train_frac: float) -> str:
    """0.8 -> '80/20' (the paper's naming)."""
    p = int(round(train_frac * 100))
    return f"{p}/{100 - p}"


def ensure_dirs() -> None:
    for d in (RAW_DIR, CACHE_DIR, SPLIT_DIR, RUNS_DIR, PREDS_DIR, TABLES_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
