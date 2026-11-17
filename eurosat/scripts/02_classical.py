"""Step 2 - classical baselines: SIFT bag-of-visual-words + SVM (paper, Table II) and the
simple colour / pixel representations, on all nine train/test splits.

    python scripts/02a_tune_bovw.py                      # first: detector threshold (validation)
    python scripts/02_classical.py                       # everything, seed 0
    python scripts/02_classical.py --features color-hist --fracs 0.8
    python scripts/02_classical.py --paper-only --seeds 1 2

Finished runs are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from joblib import Parallel, delayed  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat.classical import SIFT_DEFAULT_CT, feature_list, run_classical, run_name  # noqa: E402


def selected_contrast_threshold() -> float:
    """The detector threshold chosen on validation data by scripts/02a_tune_bovw.py."""
    f = C.TABLES_DIR / "bovw_tuning.json"
    if not f.exists():
        print("[classical] no tuning result found, using the OpenCV default threshold", flush=True)
        return SIFT_DEFAULT_CT
    return float(json.loads(f.read_text())["selected_contrast_threshold"])


def _task(feature: str, clf: str, frac: float, seed: int, force: bool) -> str:
    r = run_classical(feature, clf, frac, seed, force=force)
    return (f"{r['name']:44s} dim {r['config']['dim']:4d}  C={r['best_C']:<6g} "
            f"val {100 * r['val']['accuracy']:5.2f}  test {100 * r['test']['accuracy']:5.2f}  "
            f"({r['train_seconds']:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--features", nargs="+")
    ap.add_argument("--paper-only", action="store_true", help="only the three BoVW rows of Table II")
    ap.add_argument("--fracs", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ct = selected_contrast_threshold()
    all_feats = feature_list(ct)
    feats = [f for f in all_feats if f.startswith("sift-kp@")] if args.paper_only else (
        args.features or list(all_feats))
    fracs = C.TRAIN_FRACS if args.fracs == ["all"] else [float(f) for f in args.fracs]

    # split-independent caches are built once here, the workers only read them
    ds = data.load_dataset()
    t0 = time.time()
    for thr in sorted({float(f.split("@")[1].split("-")[0]) for f in feats if "@" in f} | {SIFT_DEFAULT_CT}):
        desc, counts = features.sift_keypoint_descriptors(ds.X, thr)
        print(f"[classical] sparse SIFT (contrast {thr:g}): {len(desc)} descriptors, "
              f"{counts.mean():.1f} per image, {100 * (counts == 0).mean():.1f}% of images without "
              f"keypoint", flush=True)
    if any("dense" in f for f in feats):
        dense = features.sift_dense_descriptors(ds.X)
        print(f"[classical] dense SIFT: {dense.shape[1]} descriptors per image", flush=True)
    for fn in (features.color_moments, features.color_histogram, features.tiny_pixels):
        fn(ds.X, cache=True)
    print(f"[classical] descriptor caches ready ({time.time() - t0:.0f}s)", flush=True)

    tasks = [(f, "svm", fr, s) for s in args.seeds for fr in fracs for f in feats]
    if "pixels16" in feats:   # the simplest baseline: a linear classifier on raw pixels
        tasks += [("pixels16", "logreg", fr, s) for s in args.seeds for fr in fracs]
    if not args.features and not args.paper_only and 0.8 in fracs and ct != SIFT_DEFAULT_CT:
        # ablation: the same pipeline with OpenCV's default detector (80/20 split only)
        tasks += [(f"sift-kp@{SIFT_DEFAULT_CT:g}-k500", "svm", 0.8, s) for s in args.seeds]
    todo = [t for t in tasks if args.force or not (C.RUNS_DIR / f"{run_name(*t)}.json").exists()]
    print(f"[classical] {len(todo)} runs to do ({len(tasks) - len(todo)} already finished)", flush=True)
    todo.sort(key=lambda t: -t[2])        # big splits first: better load balancing
    lines = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_task)(*t, args.force) for t in todo)
    for i, line in enumerate(lines, 1):
        print(f"[classical] ({i}/{len(todo)}) {line}   t={time.time() - t0:.0f}s", flush=True)
    print(f"[classical] finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
