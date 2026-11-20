"""Step 2a - choose the SIFT detector's contrast threshold on *validation* data.

The paper only says "BoVW ... using SIFT features and a trained SVM".  With OpenCV's default
detector, 28% of the 64x64 patches (about 90% of Forest and Sea & Lake) contain no keypoint at
all, so their bag-of-words histogram is empty and the classifier cannot tell them apart.  The
detector threshold is therefore treated as a hyper-parameter: for each candidate we build the
vocabulary (k = 100 and 500) on the training images of the 80/20 split and measure the accuracy
of the SVM on the validation images.  The test set is never touched here.

    python scripts/02a_tune_bovw.py

Writes results/tables/bovw_tuning-tf<share>-s<seed>.csv and .json.
Each requested split is tuned separately; the default is 80/20, seed 0.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat.classical import build_features, fit_select, preprocess  # noqa: E402
from eurosat.tuning import tuning_path  # noqa: E402
from eurosat.run_io import write_json  # noqa: E402

THRESHOLDS = (0.04, 0.02, 0.01, 0.005, 0.0025)
KS = (100, 500)


def _extract(ct: float) -> tuple[float, float, float]:
    ds = data.load_dataset(verbose=False)
    _, counts = features.sift_keypoint_descriptors(ds.X, ct)
    return ct, float(counts.mean()), float((counts == 0).mean())


def _evaluate(ct: float, k: int, train_frac: float, seed: int) -> dict:
    t0 = time.time()
    ds = data.load_dataset(verbose=False, mmap=True)
    split = data.make_split(ds.y, train_frac, seed)
    feat, kind, info = build_features(f"sift-kp@{ct:g}-k{k}", ds, split, seed)
    feat = preprocess(feat, kind, split["train"])
    _, best_c, val_by_c = fit_select("svm", feat[split["train"]], ds.y[split["train"]],
                                     feat[split["val"]], ds.y[split["val"]])
    return {"contrast_threshold": ct, "k": k, "train_frac": train_frac, "seed": seed,
            "keypoints_per_image": info["mean_descriptors_per_image"],
            "images_without_keypoint": info["images_without_descriptor"],
            "best_C": best_c, "val_acc": val_by_c[str(best_c)],
            **{f"val_acc_C{c}": v for c, v in val_by_c.items()}, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fracs", nargs="+", default=["0.8"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    if any(s < 0 for s in args.seeds) or args.jobs == 0:
        ap.error("seeds must be nonnegative and --jobs cannot be zero")
    C.ensure_dirs()
    t0 = time.time()
    for ct, mean_kp, empty in Parallel(n_jobs=min(args.jobs, len(THRESHOLDS)))(
            delayed(_extract)(ct) for ct in THRESHOLDS):
        print(f"[tune] contrast {ct:<7g} {mean_kp:5.1f} keypoints/image, "
              f"{100 * empty:4.1f}% images without keypoint", flush=True)
    fracs = C.TRAIN_FRACS if args.fracs == ["all"] else [float(f) for f in args.fracs]
    if any(f not in C.TRAIN_FRACS for f in fracs):
        ap.error("--fracs must be all or values from 0.1 to 0.9 in steps of 0.1")
    for seed in args.seeds:
        for frac in fracs:
            rows = []
            jobs = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
                delayed(_evaluate)(ct, k, frac, seed) for k in KS for ct in THRESHOLDS)
            for row in jobs:
                rows.append(row)
                print(f"[tune] split={frac:g} seed={seed} ({len(rows)}/10) "
                      f"contrast={row['contrast_threshold']:g} k={row['k']} "
                      f"val={row['val_acc']:.4f}", flush=True)
            df = pd.DataFrame(rows).sort_values(["k", "contrast_threshold"], ascending=[True, False])
            path = tuning_path(frac, seed)
            df.to_csv(path.with_suffix(".csv"), index=False)
            # A deterministic tie break keeps the larger (stricter) threshold.
            best = df[df.k == 500].sort_values(
                ["val_acc", "contrast_threshold"], ascending=[False, False]).iloc[0]
            sel = {"selected_contrast_threshold": float(best.contrast_threshold),
                   "train_frac": frac, "seed": seed,
                   "criterion": "highest validation accuracy at k=500 on this split only",
                   "val_acc": float(best.val_acc), "candidates": list(THRESHOLDS)}
            write_json(path, sel)
            print(f"[tune] selected contrast threshold {sel['selected_contrast_threshold']:g} "
                  f"for split={frac:g} seed={seed} (val {100 * sel['val_acc']:.2f}%) "
                  f"[{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
