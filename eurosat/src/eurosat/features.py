"""Hand-crafted image representations: SIFT bag-of-visual-words and simple colour descriptors.

Split-independent quantities (raw SIFT descriptors, colour statistics) are computed once for all
27,000 images and cached.  Anything that is *learned* (the visual vocabulary, scalers) is fitted
on the training images of the split at hand - see ``classical.py``.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from . import config as C


def to_gray(X: np.ndarray) -> np.ndarray:
    """(N,64,64,3) RGB uint8 -> (N,64,64) uint8 luminance (ITU-R 601, OpenCV)."""
    return np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in X])


# ------------------------------------------------------------------------------- SIFT
def sift_keypoint_descriptors(X: np.ndarray, contrast_threshold: float = 0.04,
                              cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Classic sparse SIFT: DoG keypoint detector + 128-d descriptor (OpenCV).

    ``contrast_threshold`` is the detector's threshold on the DoG response (OpenCV default 0.04);
    lowering it keeps weaker keypoints.  All other settings are the OpenCV defaults.

    Returns ``desc`` (M,128) uint8 - all descriptors of all images concatenated - and ``counts``
    (N,) - the number of keypoints of every image (0 for many homogeneous patches).
    """
    f = C.CACHE_DIR / f"sift_keypoint_ct{contrast_threshold:g}.npz"
    if cache and f.exists():
        z = np.load(f)
        if len(z["counts"]) == len(X):
            return z["desc"], z["counts"]
    sift = cv2.SIFT_create(contrastThreshold=contrast_threshold)
    descs, counts = [np.zeros((0, 128), dtype=np.uint8)], np.zeros(len(X), dtype=np.int32)
    for i, g in enumerate(to_gray(X)):
        _, d = sift.detectAndCompute(g, None)
        if d is not None:
            descs.append(np.rint(d).astype(np.uint8))
            counts[i] = len(d)
    desc = np.concatenate(descs)
    if cache:
        C.ensure_dirs()
        np.savez(f, desc=desc, counts=counts)
    return desc, counts


# Dense SIFT layout: (spatial bin width in px, grid step in px).  A descriptor has 4x4 spatial
# bins, i.e. it covers 16x16 px (first entry) or 32x32 px (second entry).
DENSE_LAYOUT = ((4, 6), (8, 8))


def _dense_grid() -> list[cv2.KeyPoint]:
    kps = []
    for bin_px, step in DENSE_LAYOUT:
        half = 2 * bin_px                               # half of the descriptor support
        centres = np.arange(half, C.IMG_SIZE - half + 1, step, dtype=float)
        size = 2.0 * bin_px / 3.0                       # OpenCV: bin width = 3 * (size / 2)
        kps += [cv2.KeyPoint(float(x), float(y), size, 0.0) for y in centres for x in centres]
    return kps


def sift_dense_descriptors(X: np.ndarray, cache: bool = True, mmap: bool = False) -> np.ndarray:
    """SIFT descriptors on a fixed regular grid (no detector): (N, G, 128) uint8, upright.

    ``mmap=True`` memory-maps the cached array (shared between parallel worker processes).
    """
    f = C.CACHE_DIR / "sift_dense.npy"
    if cache and f.exists():
        d = np.load(f, mmap_mode="r" if mmap else None)
        if len(d) == len(X):
            return d
    sift = cv2.SIFT_create()
    grid = _dense_grid()
    out = np.empty((len(X), len(grid), 128), dtype=np.uint8)
    for i, g in enumerate(to_gray(X)):
        kps, d = sift.compute(g, grid)
        if d is None or len(d) != len(grid):
            raise RuntimeError("dense SIFT lost grid points")
        out[i] = np.rint(d).astype(np.uint8)
    if cache:
        C.ensure_dirs()
        np.save(f, out)
    return out


# ------------------------------------------------------------------ bag of visual words
def fit_codebook(desc: np.ndarray, k: int, seed: int, max_samples: int = 200_000) -> np.ndarray:
    """k-means visual vocabulary from (a random sample of) *training* descriptors."""
    rng = np.random.default_rng(seed)
    if len(desc) > max_samples:
        desc = desc[rng.choice(len(desc), max_samples, replace=False)]
    km = MiniBatchKMeans(n_clusters=k, batch_size=8192, n_init=3, max_iter=100,
                         random_state=seed)
    km.fit(desc.astype(np.float32))
    return km.cluster_centers_.astype(np.float32)


def assign_words(desc: np.ndarray, centers: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Index of the nearest visual word (Euclidean) for every descriptor."""
    c2 = (centers ** 2).sum(1)
    out = np.empty(len(desc), dtype=np.int32)
    for s in range(0, len(desc), chunk):
        d = desc[s:s + chunk].astype(np.float32)
        out[s:s + chunk] = np.argmin(c2[None, :] - 2.0 * d @ centers.T, axis=1)
    return out


def bovw_histograms(words: np.ndarray, image_id: np.ndarray, n_images: int, k: int) -> np.ndarray:
    """L1-normalised word histograms (N,k).  An image without descriptors gets the zero vector."""
    flat = np.bincount(image_id.astype(np.int64) * k + words, minlength=n_images * k)
    h = flat.reshape(n_images, k).astype(np.float32)
    s = h.sum(1, keepdims=True)
    return np.divide(h, s, out=h, where=s > 0)


# ------------------------------------------------------------------------------ colour
def _per_image(fn, X: np.ndarray, cache_name: str | None, chunk: int = 1000) -> np.ndarray:
    """Apply ``fn`` to chunks of images (bounded memory); optionally keep the result on disk."""
    f = C.CACHE_DIR / f"{cache_name}.npy" if cache_name else None
    if f is not None and f.exists():
        a = np.load(f)
        if len(a) == len(X):
            return a
    a = np.concatenate([fn(np.asarray(X[i:i + chunk])) for i in range(0, len(X), chunk)])
    if f is not None:
        C.ensure_dirs()
        np.save(f, a)
    return a


def _color_moments(X: np.ndarray) -> np.ndarray:
    x = X.reshape(len(X), -1, 3).astype(np.float32) / 255.0
    return np.concatenate([x.mean(1), x.std(1)], axis=1)


def _color_histogram(X: np.ndarray, bins: int = 16) -> np.ndarray:
    n, npix = len(X), X.shape[1] * X.shape[2]
    q = (X.reshape(n, npix, 3).astype(np.int64) * bins) // 256
    offset = (np.arange(n, dtype=np.int64) * bins)[:, None]
    out = np.empty((n, 3, bins), dtype=np.float32)
    for ch in range(3):
        out[:, ch, :] = np.bincount((q[:, :, ch] + offset).ravel(),
                                    minlength=n * bins).reshape(n, bins)
    return (out / npix).reshape(n, 3 * bins)


def _tiny_pixels(X: np.ndarray, side: int = 16) -> np.ndarray:
    f = X.shape[1] // side
    x = X.reshape(len(X), side, f, side, f, 3).astype(np.float32).mean(axis=(2, 4))
    return x.reshape(len(X), -1) / 255.0


def color_moments(X: np.ndarray, cache: bool = False) -> np.ndarray:
    """Mean and standard deviation of each RGB channel: (N,6), 0..1 scale.

    ``cache=True`` (here and below) stores / reuses the result for the *full dataset* on disk.
    """
    return _per_image(_color_moments, X, "color_moments" if cache else None)


def color_histogram(X: np.ndarray, cache: bool = False) -> np.ndarray:
    """Per-channel 16-bin intensity histograms, each L1-normalised: (N,48)."""
    return _per_image(_color_histogram, X, "color_hist16" if cache else None)


def tiny_pixels(X: np.ndarray, cache: bool = False) -> np.ndarray:
    """Raw pixels after area-averaging to 16 x 16: (N,768), 0..1 scale."""
    return _per_image(_tiny_pixels, X, "pixels16" if cache else None)


def gradient_energy(X: np.ndarray) -> np.ndarray:
    """Mean Sobel gradient magnitude of the grey image - a one-number 'texture' summary (N,)."""
    out = np.empty(len(X), dtype=np.float32)
    for i, g in enumerate(to_gray(X)):
        g = g.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        out[i] = np.sqrt(gx * gx + gy * gy).mean()
    return out
