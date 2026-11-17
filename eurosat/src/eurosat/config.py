"""Central configuration: paths, dataset constants and the paper's reference numbers.

Paper being reproduced:
    P. Helber, B. Bischke, A. Dengel, D. Borth,
    "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and
    Land Cover Classification", arXiv:1709.00029v2 (IEEE JSTARS 2019).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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
FIG_DIR = ROOT / "figures"
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
CLASS_SHORT = [
    "An. Crop", "Forest", "Herbaceous", "Highway", "Industrial",
    "Pasture", "Per. Crop", "Residential", "River", "Sea & Lake",
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


# ----------------------------------------------------------- paper reference values
SPLIT_NAMES = [split_name(f) for f in TRAIN_FRACS]

# Table II: overall accuracy (%) on EuroSAT RGB for the nine train/test splits.
# ResNet-50 / GoogLeNet rows of Table II are *randomly initialised* networks (Sec. IV-A).
PAPER_TABLE2 = {
    "BoVW (SVM, SIFT, k=10)":  [54.54, 56.13, 56.77, 57.06, 57.22, 57.47, 57.71, 58.55, 58.44],
    "BoVW (SVM, SIFT, k=100)": [63.07, 64.80, 65.50, 66.16, 66.25, 66.34, 66.50, 67.22, 66.18],
    "BoVW (SVM, SIFT, k=500)": [65.62, 67.26, 68.01, 68.52, 68.61, 68.74, 69.07, 70.05, 69.54],
    "CNN (two layers)":        [75.88, 79.84, 81.29, 83.04, 84.48, 85.77, 87.24, 87.96, 88.66],
    "ResNet-50":               [75.06, 88.53, 93.75, 94.01, 94.45, 95.26, 95.32, 96.43, 96.37],
    "GoogLeNet":               [77.37, 90.97, 90.57, 91.62, 94.96, 95.54, 95.70, 96.02, 96.17],
}

# Table III (EuroSAT column): ImageNet-pretrained + fine-tuned, 80/20 split.
PAPER_TABLE3 = {"ResNet-50": 98.57, "GoogLeNet": 98.18}

# Fig. 8: row-normalised confusion matrix of the fine-tuned ResNet-50 (rows = true class).
PAPER_FIG8 = np.array([
    [0.98, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00],   # Annual crop
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # Forest
    [0.00, 0.00, 0.99, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],   # Herbaceous vegetation
    [0.00, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.00, 0.02, 0.00],   # Highway
    [0.00, 0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.01, 0.00, 0.00],   # Industrial
    [0.00, 0.00, 0.02, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00, 0.00],   # Pasture
    [0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.98, 0.00, 0.00, 0.00],   # Permanent crop
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],   # Residential
    [0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.98, 0.01],   # River
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],   # Sea & lake
])


def ensure_dirs() -> None:
    for d in (RAW_DIR, CACHE_DIR, SPLIT_DIR, RUNS_DIR, PREDS_DIR, TABLES_DIR, FIG_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
