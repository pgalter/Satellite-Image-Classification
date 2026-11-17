"""Step 2a - choose the SIFT detector's contrast threshold on *validation* data.

The paper only says "BoVW ... using SIFT features and a trained SVM".  With OpenCV's default
detector, 28% of the 64x64 patches (about 90% of Forest and Sea & Lake) contain no keypoint at
all, so their bag-of-words histogram is empty and the classifier cannot tell them apart.  The
detector threshold is therefore treated as a hyper-parameter: for each candidate we build the
vocabulary (k = 100 and 500) on the training images of the 80/20 split and measure the accuracy
of the SVM on the validation images.  The test set is never touched here.

    python scripts/02a_tune_bovw.py

Writes results/tables/bovw_tuning.csv and results/tables/bovw_tuning.json (selected value).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat.classical import build_features, fit_select, preprocess  # noqa: E402

THRESHOLDS = (0.04, 0.02, 0.01, 0.005, 0.0025)
KS = (100, 500)
TRAIN_FRAC, SEED = 0.8, 0


def _extract(ct: float) -> tuple[float, float, float]:
    ds = data.load_dataset(verbose=False)
    _, counts = features.sift_keypoint_descriptors(ds.X, ct)
    return ct, float(counts.mean()), float((counts == 0).mean())


def _evaluate(ct: float, k: int) -> dict:
    t0 = time.time()
    ds = data.load_dataset(verbose=False, mmap=True)
    split = data.make_split(ds.y, TRAIN_FRAC, SEED)
    feat, kind, info = build_features(f"sift-kp@{ct:g}-k{k}", ds, split, SEED)
    feat = preprocess(feat, kind, split["train"])
    _, best_c, val_by_c = fit_select("svm", feat[split["train"]], ds.y[split["train"]],
                                     feat[split["val"]], ds.y[split["val"]])
    return {"contrast_threshold": ct, "k": k,
            "keypoints_per_image": info["mean_descriptors_per_image"],
            "images_without_keypoint": info["images_without_descriptor"],
            "best_C": best_c, "val_acc": val_by_c[str(best_c)],
            **{f"val_acc_C{c}": v for c, v in val_by_c.items()}, "seconds": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    C.ensure_dirs()
    t0 = time.time()
    for ct, mean_kp, empty in Parallel(n_jobs=min(args.jobs, len(THRESHOLDS)))(
            delayed(_extract)(ct) for ct in THRESHOLDS):
        print(f"[tune] contrast {ct:<7g} {mean_kp:5.1f} keypoints/image, "
              f"{100 * empty:4.1f}% images without keypoint", flush=True)
    rows = Parallel(n_jobs=args.jobs)(delayed(_evaluate)(ct, k) for k in KS for ct in THRESHOLDS)
    df = pd.DataFrame(rows).sort_values(["k", "contrast_threshold"], ascending=[True, False])
    df.to_csv(C.TABLES_DIR / "bovw_tuning.csv", index=False)
    print(df[["contrast_threshold", "k", "keypoints_per_image", "images_without_keypoint",
              "best_C", "val_acc"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # selection rule: best validation accuracy at k = 500 (the paper's strongest BoVW setting)
    best = df[df.k == 500].sort_values("val_acc", ascending=False).iloc[0]
    sel = {"selected_contrast_threshold": float(best.contrast_threshold), "criterion":
           "highest validation accuracy at k=500, 80/20 split, seed 0",
           "val_acc": float(best.val_acc), "candidates": list(THRESHOLDS)}
    (C.TABLES_DIR / "bovw_tuning.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(f"[tune] selected contrast threshold {sel['selected_contrast_threshold']:g} "
          f"(val {100 * sel['val_acc']:.2f}%)  [{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
