"""Step 3 - train and evaluate the CNNs (shallow CNN, ResNet-50, GoogLeNet).

    python scripts/03_cnn.py --queue core            # Table II + Table III + Fig. 8 runs
    python scripts/03_cnn.py --queue seeds           # the same runs with seeds 1 and 2
    python scripts/03_cnn.py --preset finetune --arch resnet50 --fracs 0.8 --seeds 0
    python scripts/03_cnn.py --smoke                 # two-minute end-to-end check

Finished runs are skipped, so an interrupted queue can simply be started again.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data  # noqa: E402
from eurosat.experiments import PRESETS, make_cfg, queue  # noqa: E402
from eurosat.train import GPUData, run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--queue", choices=["core", "seeds", "all"])
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--arch", nargs="+", default=[None])
    ap.add_argument("--fracs", nargs="+", default=["0.8"], help="train fractions or 'all'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--res", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-augment", action="store_true", help="train without flips / rotations")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", default="A", help="name of this worker when several share a queue")
    ap.add_argument("--max-res", type=int, help="only take runs with an input size up to this")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required for the CNN experiments")
    device = torch.device("cuda")
    gd = GPUData(data.load_dataset(), device)

    if args.smoke:
        tiny = dict(epochs=1, min_steps=0, head_min_steps=0, tag="smoke")
        jobs = [(make_cfg("shallow", None, 0.1, 0, **tiny), False),
                (make_cfg("scratch", "googlenet", 0.1, 0, **tiny), False),
                (make_cfg("scratch", "resnet50", 0.1, 0, **tiny), False),
                (make_cfg("finetune", "googlenet", 0.1, 0, head_epochs=1, **tiny), False),
                (make_cfg("finetune", "resnet50", 0.1, 0, head_epochs=1, **tiny), False)]
        args.force = True
    elif args.queue:
        jobs = queue(args.queue)
    elif args.preset:
        fracs = C.TRAIN_FRACS if args.fracs == ["all"] else [float(f) for f in args.fracs]
        over = {k: v for k, v in (("res", args.res), ("epochs", args.epochs)) if v is not None}
        if args.no_augment:
            over.update(augment=False, tag="noaug")
        if args.tag:
            over["tag"] = args.tag
        jobs = [(make_cfg(args.preset, a, f, s, **over), args.save_model)
                for a in args.arch for f in fracs for s in args.seeds]
    else:
        ap.error("give --queue, --preset or --smoke")

    if args.max_res:
        jobs = [(c, s) for c, s in jobs if c.res <= args.max_res]
    # Several workers may share one queue: a run is claimed with an exclusive lock file.  Locks
    # that this worker left behind in an earlier (crashed) incarnation are released first.
    for stale in C.RUNS_DIR.glob("*.lock"):
        if stale.read_text() == args.worker:
            stale.unlink(missing_ok=True)

    print(f"[queue] worker {args.worker}: {len(jobs)} runs on {torch.cuda.get_device_name(0)}",
          flush=True)
    t0 = time.time()
    skipped = []
    for i, (cfg, save) in enumerate(jobs, 1):
        if (C.RUNS_DIR / f"{cfg.name}.json").exists() and not args.force:
            continue
        marker = C.RUNS_DIR / f"{cfg.name}.failed"
        n_fail = int(marker.read_text()) if marker.exists() else 0
        if n_fail >= 2 and not args.force:
            skipped.append(cfg.name)
            continue
        lock = C.RUNS_DIR / f"{cfg.name}.lock"
        try:
            with open(lock, "x") as fh:
                fh.write(args.worker)
        except FileExistsError:
            print(f"[queue] ({i}/{len(jobs)}) {cfg.name} is being run by another worker", flush=True)
            continue
        print(f"[queue] ({i}/{len(jobs)}) {cfg.name}   elapsed {(time.time() - t0) / 60:.1f} min",
              flush=True)
        try:
            run_experiment(cfg, gd, force=args.force, save_model=save)
            marker.unlink(missing_ok=True)
        except Exception:
            # A CUDA error can leave the context unusable: stop here with a non-zero exit code so
            # that the caller restarts the queue in a fresh process (finished runs are skipped).
            traceback.print_exc()
            marker.write_text(str(n_fail + 1))
            print(f"[queue] {cfg.name} failed ({n_fail + 1}x) - exiting for a clean restart", flush=True)
            raise SystemExit(1)
        finally:
            lock.unlink(missing_ok=True)
    print(f"[queue] worker {args.worker} finished in {(time.time() - t0) / 60:.1f} min; "
          f"given up after two failures: {skipped or 'none'}", flush=True)
    if args.smoke:
        for cfg, _ in jobs:
            for p in (C.RUNS_DIR / f"{cfg.name}.json", C.PREDS_DIR / f"{cfg.name}.npz"):
                p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
