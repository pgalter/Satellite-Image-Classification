"""Classical pipelines: hand-crafted representation -> standard classifier.

Protocol of one run - the same as for the CNNs:
    1. class-wise train / val / test split for (train_frac, seed);
    2. everything that is learned (visual vocabulary, feature scaling, classifier) is fitted on
       *train* only;
    3. the regularisation constant C is chosen on *val*;
    4. the chosen model is evaluated once on *test*.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from . import config as C
from . import data as D
from . import features as F
from .metrics import summarize

# Representations.  "sift-kp@<t>-k<K>": sparse SIFT with detector contrast threshold t and a
# vocabulary of K visual words - the paper's baseline.  t = 0.04 is the OpenCV default.
SIFT_DEFAULT_CT = 0.04


def feature_list(ct: float) -> tuple[str, ...]:
    """All classical representations, given the contrast threshold selected on validation data."""
    return (
        "pixels16",                      # 16x16x3 raw pixels
        "color-moments",                 # mean + std per channel (6-d)
        "color-hist",                    # 3 x 16-bin intensity histograms (48-d)
        f"sift-kp@{ct:g}-k10", f"sift-kp@{ct:g}-k100", f"sift-kp@{ct:g}-k500",   # paper, Table II
        "sift-dense-k500",               # same vocabulary size, descriptors on a regular grid
        "sift-dense-k500+color-hist",    # texture + colour
    )


SVM_C_GRID = (1.0, 10.0, 100.0, 1000.0)
LOGREG_C_GRID = (0.01, 0.1, 1.0)


def run_name(feature: str, clf: str, train_frac: float, seed: int) -> str:
    return f"{feature}-{clf}-tf{int(round(train_frac * 100)):02d}-s{seed}"


def build_features(feature: str, ds: D.EuroSAT, split: dict, seed: int) -> tuple[np.ndarray, str, dict]:
    """Feature matrix for *all* images (N,d), its type ('hist' | 'real') and some bookkeeping.

    The visual vocabulary is learned from descriptors of the training images only.
    """
    n = len(ds)
    if feature == "pixels16":
        return F.tiny_pixels(ds.X, cache=True), "real", {}
    if feature == "color-moments":
        return F.color_moments(ds.X, cache=True), "real", {}
    if feature == "color-hist":
        return F.color_histogram(ds.X, cache=True) / 3.0, "hist", {}
    if not feature.startswith("sift-"):
        raise ValueError(f"unknown feature {feature!r}")

    base, *extra = feature.split("+")
    _, detector, kk = base.split("-")
    k = int(kk[1:])
    if detector.startswith("kp@"):
        desc, counts = F.sift_keypoint_descriptors(ds.X, float(detector[3:]))
        image_id = np.repeat(np.arange(n), counts)
        in_train = np.zeros(n, dtype=bool)
        in_train[split["train"]] = True
        train_desc = desc[in_train[image_id]]
        n_train_desc = len(train_desc)
    elif detector == "dense":
        dense = F.sift_dense_descriptors(ds.X, mmap=True)
        g = dense.shape[1]
        counts = np.full(n, g, dtype=np.int32)
        image_id = np.repeat(np.arange(n), g)
        # vocabulary sample: random (training image, grid position) pairs
        n_train_desc = len(split["train"]) * g
        pick = np.sort(np.random.default_rng([seed, 17]).choice(
            n_train_desc, min(200_000, n_train_desc), replace=False))
        train_desc = np.asarray(dense[split["train"][pick // g], pick % g])
        desc = dense.reshape(-1, 128)
    else:
        raise ValueError(f"unknown SIFT variant {detector!r}")
    centers = F.fit_codebook(train_desc, k, seed)
    hist = F.bovw_histograms(F.assign_words(desc, centers), image_id, n, k)
    info = {"k": k, "n_train_descriptors": int(n_train_desc),
            "mean_descriptors_per_image": float(counts.mean()),
            "images_without_descriptor": float((counts == 0).mean())}
    if extra == ["color-hist"]:
        hist = np.concatenate([hist, F.color_histogram(ds.X, cache=True) / 3.0], axis=1)
    elif extra:
        raise ValueError(f"unknown feature combination {feature!r}")
    return hist, "hist", info


def preprocess(feat: np.ndarray, kind: str, train_idx: np.ndarray) -> np.ndarray:
    """Histograms: element-wise square root (Hellinger map; every L1-normalised block gets unit
    L2 norm).  Real-valued features: standardisation with training mean / std."""
    if kind == "hist":
        return np.sqrt(feat, dtype=np.float32)
    mu = feat[train_idx].mean(0)
    sd = feat[train_idx].std(0) + 1e-8
    return ((feat - mu) / sd).astype(np.float32)


def fit_select(clf: str, ftr: np.ndarray, ytr: np.ndarray, fva: np.ndarray, yva: np.ndarray):
    """Fit one model per value of C on train, return the one with the best validation accuracy.

    C is increased step by step; the search stops once two successive values failed to improve
    on the best validation accuracy (large C is very slow on noisy features and never wins there).
    """
    grid = SVM_C_GRID if clf == "svm" else LOGREG_C_GRID
    val_by_c: dict[str, float] = {}
    best = None
    since_best = 0
    for c in grid:
        if since_best >= 2:
            break
        if clf == "svm":
            model = SVC(C=c, kernel="rbf", gamma="scale", cache_size=1500)
        elif clf == "logreg":
            model = LogisticRegression(C=c, max_iter=3000)
        else:
            raise ValueError(f"unknown classifier {clf!r}")
        model.fit(ftr, ytr)
        acc = float((model.predict(fva) == yva).mean())
        val_by_c[str(c)] = acc
        if best is None or acc > best[0]:      # ties -> the smaller C (stronger regularisation)
            best = (acc, c, model)
            since_best = 0
        else:
            since_best += 1
    return best[2], best[1], val_by_c


def run_classical(feature: str, clf: str, train_frac: float, seed: int = 0,
                  force: bool = False) -> dict:
    """One classical experiment; the result is written to ``results/runs/<name>.json``."""
    C.ensure_dirs()
    name = run_name(feature, clf, train_frac, seed)
    out_json = C.RUNS_DIR / f"{name}.json"
    if out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))
    t0 = time.time()
    ds = D.load_dataset(verbose=False, mmap=True)      # workers share one read-only copy
    split = D.make_split(ds.y, train_frac, seed)
    feat, kind, info = build_features(feature, ds, split, seed)
    feat = preprocess(feat, kind, split["train"])
    t_feat = time.time() - t0
    ytr, yva, yte = ds.y[split["train"]], ds.y[split["val"]], ds.y[split["test"]]
    ftr, fva, fte = feat[split["train"]], feat[split["val"]], feat[split["test"]]

    model, best_c, val_by_c = fit_select(clf, ftr, ytr, fva, yva)
    val_pred = model.predict(fva)
    test_pred = model.predict(fte)
    result = {
        "name": name, "kind": "classical",
        "config": {"feature": feature, "classifier": clf, "train_frac": train_frac, "seed": seed,
                   "C_grid": list(SVM_C_GRID if clf == "svm" else LOGREG_C_GRID), "kernel": "rbf" if clf == "svm" else None,
                   "dim": int(feat.shape[1])},
        "split": C.split_name(train_frac),
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "val_by_C": val_by_c, "best_C": best_c,
        "val": summarize(yva, val_pred), "test": summarize(yte, test_pred),
        "feature_info": info, "feature_seconds": t_feat, "train_seconds": time.time() - t0,
    }
    np.savez_compressed(C.PREDS_DIR / f"{name}.npz", idx=split["test"], y_true=yte,
                        y_pred=test_pred)
    out_json.write_text(json.dumps(result), encoding="utf-8")
    return result
