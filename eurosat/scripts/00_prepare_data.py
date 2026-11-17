"""Step 0 - download, verify, extract and index EuroSAT RGB; export the train/val/test splits.

    python scripts/00_prepare_data.py            # idempotent: skips what is already done
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    archive = data.download_archive(force=args.force)
    data.extract_archive(archive, force=args.force)
    ds = data.load_dataset()
    counts = np.bincount(ds.y, minlength=C.N_CLASSES)
    print(f"[data] {len(ds)} images, array {ds.X.shape} {ds.X.dtype}")
    for cls, n in zip(C.CLASSES, counts):
        print(f"[data]   {cls:22s} {n}")
    data.export_splits(ds, args.seeds)
    s = data.make_split(ds.y, 0.8, seed=args.seeds[0])
    print("[data] 80/20 split sizes:", {k: len(v) for k, v in s.items()})


if __name__ == "__main__":
    main()
