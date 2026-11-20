"""Run SIFT + BoVW + SVM using the threshold tuned for each split.

    python scripts/02_classical.py --ks 500 --fracs 0.8 --seeds 0 --jobs 2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import time
from joblib import Parallel, delayed

from eurosat import config as C, data, features
from eurosat.classical import run_classical, run_name
from eurosat.tuning import selected_threshold
from eurosat.run_io import completed_run


def _task(feature, clf, frac, seed, force):
    r = run_classical(feature, clf, frac, seed, force=force)
    return (f"{r['name']} C={r['best_C']:g} "
            f"val={100*r['val']['accuracy']:.2f}% test={100*r['test']['accuracy']:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", type=int, nargs="+", choices=[10, 100, 500], default=[10, 100, 500])
    ap.add_argument("--fracs", nargs="+", default=["0.8"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fracs = C.TRAIN_FRACS if args.fracs == ["all"] else [float(f) for f in args.fracs]
    if any(f not in C.TRAIN_FRACS for f in fracs):
        ap.error("--fracs must be all or values from 0.1 to 0.9 in steps of 0.1")
    if any(s < 0 for s in args.seeds) or args.jobs == 0:
        ap.error("seeds must be nonnegative and --jobs cannot be zero")
    C.ensure_dirs()
    tasks = []
    for s in args.seeds:
        for fr in fracs:
            ct = selected_threshold(fr, s)
            tasks += [(f"sift-kp@{ct:g}-k{k}", "svm", fr, s) for k in args.ks]
    todo = [t for t in tasks if args.force or completed_run(run_name(*t)) is None]
    print(f"[classical] {len(todo)} runs to do ({len(tasks)-len(todo)} already finished)", flush=True)
    if not todo:
        return
    ds = data.load_dataset()
    for ct in sorted({float(t[0].split("@")[1].split("-")[0]) for t in todo}):
        features.sift_keypoint_descriptors(ds.X, ct)
    del ds
    t0 = time.time()
    lines = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(_task)(*t, args.force) for t in todo)
    for i, line in enumerate(lines, 1):
        print(f"[classical] ({i}/{len(todo)}) {line}", flush=True)
    print(f"[classical] finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
