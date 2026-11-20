"""SIFT descriptors, training-only visual vocabularies and BoVW histograms.

Split-independent quantities (raw SIFT descriptors) are computed once for all
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


# ------------------------------------------------------------------ bag of visual words
def fit_codebook(desc: np.ndarray, k: int, seed: int, max_samples: int = 200_000) -> np.ndarray:
    """k-means visual vocabulary from (a random sample of) *training* descriptors."""
    if k < 1 or len(desc) < k:
        raise ValueError(f"need at least {k} training descriptors, got {len(desc)}")
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
