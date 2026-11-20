"""Dataset access: download + checksum, safe extraction, indexing, decoding and splits.

Everything downstream works on one in-memory array ``X`` (N, 64, 64, 3, uint8, RGB) with labels
``y`` (N,), in a fixed, documented order (class folder alphabetical, then image number).  Splits
are index arrays into that order and are also exported with file names to ``splits/``.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C

_MEMBER_RE = re.compile(r"^(?P<cls>[A-Za-z]+)_(?P<num>\d+)\.jpg$")


# ----------------------------------------------------------------- download / extract
def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_archive(force: bool = False) -> Path:
    """Download EuroSAT_RGB.zip from the authors' Zenodo record and verify size + MD5."""
    C.ensure_dirs()
    dst = C.RAW_DIR / C.ARCHIVE_NAME
    if dst.exists() and not force:
        if dst.stat().st_size == C.ARCHIVE_SIZE and md5sum(dst) == C.ARCHIVE_MD5:
            print(f"[data] archive already present and verified: {dst}")
            return dst
        print("[data] existing archive failed verification, downloading again")
    tmp = dst.with_suffix(".part")
    print(f"[data] downloading {C.ZENODO_URL}")
    req = urllib.request.Request(C.ZENODO_URL, headers={"User-Agent": "python-urllib"})
    done = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if done % (10 << 20) < (1 << 20):
                print(f"[data]   {done / 1e6:6.1f} / {C.ARCHIVE_SIZE / 1e6:.1f} MB", flush=True)
    size, digest = tmp.stat().st_size, md5sum(tmp)
    if size != C.ARCHIVE_SIZE or digest != C.ARCHIVE_MD5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download corrupted: size={size} md5={digest} "
                           f"(expected {C.ARCHIVE_SIZE} / {C.ARCHIVE_MD5})")
    tmp.replace(dst)
    print(f"[data] saved {dst} ({size} bytes, md5 {digest} OK)")
    return dst


def extract_archive(archive: Path | None = None, force: bool = False) -> Path:
    """Extract the JPEGs into ``data/EuroSAT_RGB/<Class>/<Class>_<n>.jpg``.

    Archive paths are never trusted: only members whose *basename* matches ``<Class>_<n>.jpg``
    for one of the ten known classes are written, each to a path we build ourselves.
    """
    archive = archive or (C.RAW_DIR / C.ARCHIVE_NAME)
    if C.IMG_DIR.exists():
        if not force and _count_images(C.IMG_DIR) == C.N_IMAGES:
            print(f"[data] images already extracted: {C.IMG_DIR}")
            return C.IMG_DIR
        shutil.rmtree(C.IMG_DIR)
    C.IMG_DIR.mkdir(parents=True)
    for cls in C.CLASSES:
        (C.IMG_DIR / cls).mkdir()
    n = 0
    skipped: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            m = _MEMBER_RE.match(name)
            if m is None or m["cls"] not in C.CLASSES:
                skipped.append(info.filename)
                continue
            with zf.open(info) as src, open(C.IMG_DIR / m["cls"] / name, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    print(f"[data] extracted {n} images to {C.IMG_DIR}; skipped {len(skipped)} other members"
          + (f" (e.g. {skipped[:3]})" if skipped else ""))
    if n != C.N_IMAGES:
        raise RuntimeError(f"expected {C.N_IMAGES} images, extracted {n}")
    # A restored/re-extracted archive invalidates both decoded pixels and derived features.
    # Otherwise --force could silently retain arrays from modified/corrupt JPEGs.
    for pattern in ("eurosat_rgb_*.npy", "eurosat_rgb_index.json", "sift_keypoint_*.npz",
                    "sift_dense.npy", "color_moments.npy", "color_hist16.npy", "pixels16.npy"):
        for cached in C.CACHE_DIR.glob(pattern):
            cached.unlink()
    return C.IMG_DIR


def _count_images(img_dir: Path) -> int:
    return sum(1 for cls in C.CLASSES if (img_dir / cls).is_dir()
               for p in (img_dir / cls).iterdir() if p.suffix.lower() == ".jpg")


# ------------------------------------------------------------------- index / decoding
def build_index(img_dir: Path | None = None) -> tuple[list[str], np.ndarray]:
    """Relative paths and integer labels in canonical order (class, then image number)."""
    img_dir = img_dir or C.IMG_DIR
    paths: list[str] = []
    labels: list[int] = []
    for ci, cls in enumerate(C.CLASSES):
        files = []
        for p in (img_dir / cls).iterdir():
            m = _MEMBER_RE.match(p.name)
            if m is None or m["cls"] != cls:
                raise RuntimeError(f"unexpected file in dataset folder: {p}")
            files.append((int(m["num"]), p.name))
        files.sort()
        if len(files) != C.EXPECTED_COUNTS[cls]:
            raise RuntimeError(f"{cls}: found {len(files)} images, expected {C.EXPECTED_COUNTS[cls]}")
        paths += [f"{cls}/{name}" for _, name in files]
        labels += [ci] * len(files)
    return paths, np.asarray(labels, dtype=np.int64)


@dataclass
class EuroSAT:
    X: np.ndarray          # (N, 64, 64, 3) uint8, RGB
    y: np.ndarray          # (N,) int64, index into config.CLASSES
    paths: list[str]       # relative path of every image, same order

    def __len__(self) -> int:
        return len(self.y)


def load_dataset(use_cache: bool = True, verbose: bool = True, mmap: bool = False) -> EuroSAT:
    """Decode all JPEGs once (PIL -> RGB uint8) and cache the array for fast reuse.

    ``mmap=True`` memory-maps the cached array read-only instead of loading it, so that parallel
    worker processes share a single copy.
    """
    C.ensure_dirs()
    cache = C.CACHE_DIR / "eurosat_rgb_uint8.npy"
    cache_index = C.CACHE_DIR / "eurosat_rgb_index.json"
    paths, y = build_index()
    if use_cache and cache.exists() and cache_index.exists():
        if json.loads(cache_index.read_text(encoding="utf-8")) == paths:
            return EuroSAT(np.load(cache, mmap_mode="r" if mmap else None), y, paths)
        if verbose:
            print("[data] cache does not match the image folder, rebuilding")
    X = np.empty((len(paths), C.IMG_SIZE, C.IMG_SIZE, 3), dtype=np.uint8)
    for i, rel in enumerate(paths):
        with Image.open(C.IMG_DIR / rel) as im:
            if im.size != (C.IMG_SIZE, C.IMG_SIZE):
                raise RuntimeError(f"{rel}: unexpected size {im.size}")
            X[i] = np.asarray(im.convert("RGB"))
        if verbose and (i + 1) % 5000 == 0:
            print(f"[data]   decoded {i + 1}/{len(paths)}", flush=True)
    np.save(cache, X)
    cache_index.write_text(json.dumps(paths), encoding="utf-8")
    if verbose:
        print(f"[data] cached decoded images: {cache}")
    return EuroSAT(X, y, paths)


# ----------------------------------------------------------------------------- splits
def make_split(y: np.ndarray, train_frac: float, seed: int = C.DEFAULT_SEED,
               val_frac: float = C.VAL_FRAC_OF_TRAIN) -> dict[str, np.ndarray]:
    """Class-wise (stratified) train / validation / test split.

    Following the paper, ``train_frac`` of *every class* forms the training portion and the rest
    is the test set ("we ensured that the split is applied class-wise").  The paper has no
    validation set; we additionally hold out ``val_frac`` of each class's training portion for
    model selection, so the test set is never used for any decision.

    Each class is shuffled once with a generator seeded by (seed, class id).  The training
    portion is the first ``train_frac * n_c`` images of that order and every ``1/val_frac``-th
    of them is a validation image.  Consequently the splits are *nested*: the 10% training
    portion is contained in the 20% one, and so on, and the 90/10 test set is contained in every
    other test set.  This removes one source of noise from the learning curves of Table II.
    """
    y = np.asarray(y)
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must be in (0,1)")
    if y.ndim != 1 or not len(y) or seed < 0:
        raise ValueError("labels must be a nonempty vector and seed must be nonnegative")
    every = int(round(1.0 / val_frac))
    if every < 2 or not np.isclose(val_frac * every, 1.0):
        raise ValueError("nested validation selection requires val_frac = 1 / n, with integer n >= 2")
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng = np.random.default_rng([seed, int(c)])
        perm = idx[rng.permutation(len(idx))]
        n_dev = int(round(train_frac * len(idx)))
        dev, test = perm[:n_dev], perm[n_dev:]
        is_val = (np.arange(n_dev) % every) == (every - 1)
        tr.append(dev[~is_val])
        va.append(dev[is_val])
        te.append(test)
    out = {"train": np.sort(np.concatenate(tr)), "val": np.sort(np.concatenate(va)),
           "test": np.sort(np.concatenate(te))}
    assert len(set(out["train"]) | set(out["val"]) | set(out["test"])) == len(y)
    assert len(out["train"]) + len(out["val"]) + len(out["test"]) == len(y)
    return out


def export_splits(ds: EuroSAT, seeds: list[int]) -> None:
    """Write every split to ``splits/seed<k>.json`` (indices) for documentation / external reuse."""
    C.ensure_dirs()
    (C.SPLIT_DIR / "index.json").write_text(json.dumps(
        {"description": "position -> relative image path; split files hold positions",
         "classes": C.CLASSES, "paths": ds.paths}), encoding="utf-8")
    for seed in seeds:
        blob = {"seed": seed, "val_frac_of_train": C.VAL_FRAC_OF_TRAIN, "splits": {}}
        for f in C.TRAIN_FRACS:
            s = make_split(ds.y, f, seed)
            blob["splits"][C.split_name(f)] = {k: v.tolist() for k, v in s.items()}
        (C.SPLIT_DIR / f"seed{seed}.json").write_text(json.dumps(blob), encoding="utf-8")
        print(f"[data] wrote {C.SPLIT_DIR / f'seed{seed}.json'}")


def split_summary(y: np.ndarray, split: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Per-class image counts of each part of a split."""
    return {k: np.bincount(y[v], minlength=C.N_CLASSES).tolist() for k, v in split.items()}
