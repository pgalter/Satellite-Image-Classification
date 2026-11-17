"""Sanity checks of the data pipeline (fast, no GPU needed).

    python scripts/99_check_pipeline.py

Verifies the properties the evaluation protocol relies on: splits are class-wise, disjoint,
complete, nested and deterministic; the exported split files match; the augmentation is a
label-preserving symmetry; feature builders behave as documented.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat.train import d4_augment  # noqa: E402


def check_splits(ds: data.EuroSAT, seeds=(0, 1, 2)) -> None:
    n_class = np.bincount(ds.y, minlength=C.N_CLASSES)
    for seed in seeds:
        prev = None
        for f in C.TRAIN_FRACS:
            s = data.make_split(ds.y, f, seed)
            tr, va, te = (set(s[k].tolist()) for k in ("train", "val", "test"))
            assert not (tr & va) and not (tr & te) and not (va & te), "parts overlap"
            assert len(tr) + len(va) + len(te) == len(ds), "split is not complete"
            for c in range(C.N_CLASSES):                      # class-wise, exact proportions
                n_dev = int(round(f * n_class[c]))
                assert (ds.y[s["test"]] == c).sum() == n_class[c] - n_dev
                assert (ds.y[s["val"]] == c).sum() == n_dev // 10
                assert (ds.y[s["train"]] == c).sum() == n_dev - n_dev // 10
            again = data.make_split(ds.y, f, seed)             # deterministic
            assert all(np.array_equal(s[k], again[k]) for k in s)
            if prev is not None:                               # nested across training shares
                assert prev[0] <= tr and prev[1] <= va and te <= prev[2], "splits are not nested"
            prev = (tr, va, te)
        other = data.make_split(ds.y, 0.8, seed + 100)
        assert not np.array_equal(other["test"], data.make_split(ds.y, 0.8, seed)["test"])
        f_json = C.SPLIT_DIR / f"seed{seed}.json"
        if f_json.exists():                                    # exported files = recomputed splits
            blob = json.loads(f_json.read_text(encoding="utf-8"))["splits"]
            for f in C.TRAIN_FRACS:
                s = data.make_split(ds.y, f, seed)
                assert all(blob[C.split_name(f)][k] == s[k].tolist() for k in s)
    print(f"splits ok: {len(seeds)} seeds x {len(C.TRAIN_FRACS)} ratios - disjoint, complete, "
          f"class-wise, nested, deterministic, equal to splits/*.json")


def check_dataset(ds: data.EuroSAT) -> None:
    assert ds.X.shape == (C.N_IMAGES, C.IMG_SIZE, C.IMG_SIZE, 3) and ds.X.dtype == np.uint8
    counts = np.bincount(ds.y, minlength=C.N_CLASSES)
    assert counts.tolist() == [C.EXPECTED_COUNTS[c] for c in C.CLASSES]
    assert all(p.startswith(C.CLASSES[y] + "/") for p, y in zip(ds.paths[::997], ds.y[::997]))
    print(f"dataset ok: {ds.X.shape} {ds.X.dtype}, class counts {counts.tolist()}")


def check_augmentation() -> None:
    gen = torch.Generator().manual_seed(0)
    img = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    seen = set()
    for _ in range(400):
        out = d4_augment(img.clone(), gen)
        assert torch.equal(out.flatten().sort().values, img.flatten().sort().values)   # same pixels
        assert torch.equal(out.sum(dim=(2, 3)), img.sum(dim=(2, 3)))                   # per channel
        seen.add(tuple(out.flatten().tolist()))
    assert len(seen) == 8, f"expected the 8 symmetries of the square, got {len(seen)}"
    print("augmentation ok: exactly the 8 symmetries of the square, pixel values untouched")


def check_features(ds: data.EuroSAT) -> None:
    x = np.asarray(ds.X[:64])
    h = features.color_histogram(x)
    assert h.shape == (64, 48) and np.allclose(h.reshape(64, 3, 16).sum(2), 1.0, atol=1e-5)
    m = features.color_moments(x)
    assert m.shape == (64, 6) and np.allclose(m[:, :3], x.reshape(64, -1, 3).mean(1) / 255, atol=1e-5)
    p = features.tiny_pixels(x)
    assert p.shape == (64, 768) and 0.0 <= p.min() and p.max() <= 1.0
    words = np.array([0, 0, 1, 3, 3, 3])
    image_id = np.array([0, 0, 0, 2, 2, 2])
    hist = features.bovw_histograms(words, image_id, n_images=3, k=4)
    assert np.allclose(hist[0], [2 / 3, 1 / 3, 0, 0]) and np.allclose(hist[2], [0, 0, 0, 1])
    assert np.all(hist[1] == 0), "an image without descriptors must give the zero histogram"
    centers = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    assert features.assign_words(np.array([[1, 1], [9, 8]], dtype=np.uint8), centers).tolist() == [0, 1]
    print("features ok: colour histogram / moments / thumbnails / word assignment / BoVW histogram")


def main() -> None:
    ds = data.load_dataset(verbose=False, mmap=True)
    check_dataset(ds)
    check_splits(ds)
    check_augmentation()
    check_features(ds)
    print("all checks passed")


if __name__ == "__main__":
    main()
