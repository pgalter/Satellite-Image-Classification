"""Step 1 - exploratory data analysis: figures in figures/eda_*.png + results/tables/eda_summary.json.

Descriptive statistics (counts, file properties, duplicates) use all 27,000 images.  Everything
that could guide modelling decisions (class profiles, nearest-neighbour confusions, the 2-D
embedding) uses only the *training* images of the 80/20 split, so the test set stays unseen.

    python scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat import plotting as P  # noqa: E402

SEED = 0


def file_properties(ds: data.EuroSAT, rng: np.random.Generator, n_sample: int = 1000) -> dict:
    """File sizes from the directory listing (all files); container format / colour mode from a
    random sample.  The pixel size of *every* image was already checked when decoding."""
    nbytes = np.asarray([e.stat().st_size for cls in C.CLASSES
                         for e in os.scandir(C.IMG_DIR / cls) if e.name.lower().endswith(".jpg")])
    fmt, mode, size = Counter(), Counter(), Counter()
    for i in rng.choice(len(ds), n_sample, replace=False):
        with Image.open(C.IMG_DIR / ds.paths[i]) as im:
            fmt[im.format] += 1
            mode[im.mode] += 1
            size[f"{im.size[0]}x{im.size[1]}"] += 1
    return {"n_files": int(len(nbytes)), "sampled_for_format": n_sample,
            "format": dict(fmt), "mode": dict(mode), "size": dict(size),
            "decoded_shape_all_images": list(ds.X.shape),
            "file_bytes": {"min": int(nbytes.min()), "median": float(np.median(nbytes)),
                           "max": int(nbytes.max()), "total_MB": float(nbytes.sum() / 1e6)}}


def duplicates(ds: data.EuroSAT, split: dict) -> dict:
    groups = defaultdict(list)
    for i, x in enumerate(ds.X):
        groups[hashlib.sha1(x.tobytes()).hexdigest()].append(i)
    dup = [g for g in groups.values() if len(g) > 1]
    part = np.empty(len(ds), dtype=object)
    for k, v in split.items():
        part[v] = k
    cross_class = [g for g in dup if len({int(ds.y[i]) for i in g}) > 1]
    leak = [g for g in dup if "test" in {part[i] for i in g} and len({part[i] for i in g}) > 1]
    return {"groups": len(dup), "images_in_groups": int(sum(len(g) for g in dup)),
            "groups_spanning_classes": len(cross_class),
            "groups_spanning_test_and_trainval_80_20": len(leak),
            "examples": [[ds.paths[i] for i in g] for g in dup[:10]]}


def fig_class_counts(ds: data.EuroSAT) -> None:
    counts = np.bincount(ds.y, minlength=C.N_CLASSES)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ypos = np.arange(C.N_CLASSES)
    ax.barh(ypos, counts, height=0.5, color=P.ACCENT)
    ax.set_yticks(ypos, C.CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 3400)
    ax.set_xlabel("images")
    ax.grid(axis="y", visible=False)
    for y, c in zip(ypos, counts):
        ax.text(c + 40, y, f"{c:,}", va="center", fontsize=8.5, color=P.INK2)
    ax.set_title("Images per class (27,000 in total; largest / smallest = 1.5)")
    P.save(fig, C.FIG_DIR / "eda_class_counts.png")


def fig_samples(ds: data.EuroSAT, train: np.ndarray, rng: np.random.Generator, n: int = 10) -> None:
    imgs = []
    for c in range(C.N_CLASSES):
        idx = train[ds.y[train] == c]
        imgs += [ds.X[i] for i in rng.choice(idx, n, replace=False)]
    fig = P.image_grid(imgs, n_cols=n, row_labels=C.CLASSES, cell=0.82, label_width=1.55,
                       suptitle="Random training patches (64 x 64 px, 10 m per pixel), one row per class")
    P.save(fig, C.FIG_DIR / "eda_samples.png")


def fig_profiles(ds: data.EuroSAT, train: np.ndarray, gray: np.ndarray, grad: np.ndarray) -> dict:
    """Per-class distributions of three one-number image descriptors + the mean colour."""
    y = ds.y[train]
    bright = gray[train].reshape(len(train), -1).mean(1)
    contrast = gray[train].reshape(len(train), -1).std(1)
    edge = grad[train]
    mean_rgb = np.stack([ds.X[train][y == c].reshape(-1, 3).mean(0) for c in range(C.N_CLASSES)])
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.6), gridspec_kw={"width_ratios": [0.34, 1, 1, 1]},
                             sharey=True)
    ypos = np.arange(C.N_CLASSES)
    ax = axes[0]
    ax.imshow(mean_rgb[:, None, :] / 255.0, aspect="auto", extent=(0, 1, C.N_CLASSES - 0.5, -0.5))
    ax.set_yticks(ypos, C.CLASSES)
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("mean colour", fontsize=9, fontweight="normal", color=P.INK2, loc="center")
    panels = [(bright, "brightness (mean grey level)"), (contrast, "contrast (std of grey level)"),
              (edge, "edge strength (mean Sobel magnitude)")]
    stats = {}
    for ax, (v, title) in zip(axes[1:], panels):
        groups = [v[y == c] for c in range(C.N_CLASSES)]
        ax.boxplot(groups, positions=ypos, vert=False, widths=0.5, whis=(5, 95), showfliers=False,
                   manage_ticks=False, patch_artist=True, boxprops=dict(facecolor="#cde2fb", edgecolor=P.ACCENT, linewidth=1.2),
                   whiskerprops=dict(color=P.ACCENT, linewidth=1.2), capprops=dict(color=P.ACCENT, linewidth=1.2),
                   medianprops=dict(color=P.INK, linewidth=1.4))
        ax.set_title(title, fontsize=9, fontweight="normal", color=P.INK2)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
        stats[title] = {C.CLASSES[c]: float(np.median(g)) for c, g in enumerate(groups)}
    axes[0].set_ylim(C.N_CLASSES - 0.5, -0.5)
    axes[0].set_yticks(ypos, C.CLASSES)
    fig.suptitle("Class profiles on three one-number descriptors (box = quartiles, whiskers = 5-95%)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.subplots_adjust(wspace=0.16, top=0.82)
    P.save(fig, C.FIG_DIR / "eda_class_profiles.png")
    stats["mean_rgb"] = {C.CLASSES[c]: [float(v) for v in mean_rgb[c]] for c in range(C.N_CLASSES)}
    return stats


def nn_confusion(feat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out 1-nearest-neighbour confusion (counts): rows = class of the image,
    columns = class of its nearest other image."""
    nn = NearestNeighbors(n_neighbors=2).fit(feat)
    _, ind = nn.kneighbors(feat)
    self_first = ind[:, 0] == np.arange(len(feat))
    neighbour = np.where(self_first, ind[:, 1], ind[:, 0])
    cm = np.zeros((C.N_CLASSES, C.N_CLASSES), dtype=np.int64)
    np.add.at(cm, (y, y[neighbour]), 1)
    return cm


def fig_nn_confusion(cm: np.ndarray, name: str, title: str) -> dict:
    pct = 100.0 * cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    P.heatmap(ax, pct, C.CLASS_SHORT, C.CLASS_SHORT, fmt="{:.0f}", vmax=100, hide_below=0.5)
    ax.set_xlabel("class of the nearest neighbour")
    ax.set_ylabel("class of the image")
    ax.set_title(title)
    P.save(fig, C.FIG_DIR / name)
    off = pct.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j] + off[j, i]) / 2, C.CLASSES[i], C.CLASSES[j])
                   for i in range(C.N_CLASSES) for j in range(i + 1, C.N_CLASSES))[::-1][:8]
    return {"loo_1nn_accuracy": float(np.trace(cm) / cm.sum()),
            "per_class_recall": {C.CLASSES[i]: float(pct[i, i] / 100) for i in range(C.N_CLASSES)},
            "most_confused_pairs_mean_pct": [[a, b, float(v)] for v, a, b in pairs]}


def fig_tsne(feat: np.ndarray, y: np.ndarray, rng: np.random.Generator, per_class: int = 500) -> None:
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                          for c in range(C.N_CLASSES)])
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
               random_state=SEED).fit_transform(feat[idx])
    fig, axes = plt.subplots(2, 5, figsize=(10.2, 4.5), sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        m = y[idx] == c
        ax.scatter(emb[~m, 0], emb[~m, 1], s=2, color="#dddcd5", linewidths=0, rasterized=True)
        ax.scatter(emb[m, 0], emb[m, 1], s=3.5, color=P.ACCENT, linewidths=0, rasterized=True)
        ax.set_title(C.CLASSES[c], fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("t-SNE of colour histograms (5,000 training images): each panel highlights one class",
                 x=0.01, ha="left", fontsize=10.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.18, top=0.86)
    P.save(fig, C.FIG_DIR / "eda_tsne_colour.png")


def fig_unusual(ds: data.EuroSAT, gray: np.ndarray, n: int = 10) -> dict:
    """Images at the extremes: flattest, brightest and the strongest colour casts."""
    flat = gray.reshape(len(ds), -1)
    contrast, bright = flat.std(1), flat.mean(1)
    m = ds.X.reshape(len(ds), -1, 3).mean(1)
    blue_cast = m[:, 2] - m[:, :2].mean(1)
    red_cast = m[:, 0] - m[:, 1:].mean(1)
    rows = [("lowest contrast", np.argsort(contrast)[:n]),
            ("brightest", np.argsort(-bright)[:n]),
            ("strongest blue cast", np.argsort(-blue_cast)[:n]),
            ("strongest red cast", np.argsort(-red_cast)[:n])]
    imgs, titles = [], []
    for _, idx in rows:
        imgs += [ds.X[i] for i in idx]
        titles += [C.CLASS_SHORT[ds.y[i]] for i in idx]
    fig = P.image_grid(imgs, n_cols=n, titles=titles, row_labels=[r[0] for r in rows], cell=0.82,
                       label_width=1.45, suptitle="Unusual patches (label above each image)")
    P.save(fig, C.FIG_DIR / "eda_unusual.png")
    low = contrast < 2.0
    return {"near_constant_images_std_lt_2": {C.CLASSES[c]: int((low & (ds.y == c)).sum())
                                              for c in range(C.N_CLASSES)},
            "brightness_mean_range": [float(bright.min()), float(bright.max())]}


def near_duplicates(ds: data.EuroSAT, split: dict, n_show: int = 10) -> dict:
    """How close is every test image of the 80/20 split to its most similar train/val image?

    Candidates are found on 16x16 thumbnails; the reported distance is the RMSE over all
    64 x 64 x 3 values (grey levels, 0..255).  Patches cut from the same scene next to each other
    would show up here as near-identical pairs across the split.
    """
    dev = np.concatenate([split["train"], split["val"]])
    te = split["test"]
    thumb = features.tiny_pixels(ds.X)
    _, ind = NearestNeighbors(n_neighbors=1).fit(thumb[dev]).kneighbors(thumb[te])
    mate = dev[ind[:, 0]]
    diff = ds.X[te].astype(np.float32) - ds.X[mate].astype(np.float32)
    rmse = np.sqrt((diff ** 2).reshape(len(te), -1).mean(1))
    yt = ds.y[te]
    out = {"n_test": int(len(te)), "median_rmse": float(np.median(rmse)),
           "nearest_has_same_label": float((ds.y[mate] == yt).mean()),
           "test_images_with_rmse_below": {str(t): int((rmse < t).sum()) for t in (2, 4, 8)},
           "rmse_below_4_by_class": {C.CLASSES[c]: int(((rmse < 4) & (yt == c)).sum())
                                     for c in range(C.N_CLASSES)},
           "median_rmse_by_class": {C.CLASSES[c]: float(np.median(rmse[yt == c]))
                                    for c in range(C.N_CLASSES)}}
    # show the closest pairs outside the two homogeneous classes, where "similar" is trivial
    textured = np.flatnonzero(~np.isin(yt, [C.CLASSES.index("SeaLake"), C.CLASSES.index("Forest")]))
    order = textured[np.argsort(rmse[textured])][:n_show]
    imgs = [ds.X[te[i]] for i in order] + [ds.X[mate[i]] for i in order]
    titles = [f"{C.CLASS_SHORT[yt[i]]}" for i in order] + [f"RMSE {rmse[i]:.1f}" for i in order]
    fig = P.image_grid(imgs, n_cols=n_show, titles=titles,
                       row_labels=["test image", "closest train/val\nimage"], cell=0.82,
                       label_width=1.45,
                       suptitle="Closest train/test pairs outside Forest and Sea & Lake (80/20 split)")
    P.save(fig, C.FIG_DIR / "eda_closest_train_test.png")
    out["closest_pairs_shown"] = [[ds.paths[te[i]], ds.paths[mate[i]], float(rmse[i])] for i in order]
    return out


def main() -> None:
    C.ensure_dirs()
    P.apply_style()
    rng = np.random.default_rng(SEED)
    ds = data.load_dataset()
    split = data.make_split(ds.y, 0.8, SEED)
    train = split["train"]
    summary: dict = {"n_images": len(ds), "classes": C.CLASSES,
                     "counts": np.bincount(ds.y, minlength=C.N_CLASSES).tolist()}

    t0 = time.time()
    summary["files"] = file_properties(ds, rng)
    print(f"[eda] file properties done ({time.time() - t0:.0f}s)", flush=True)
    summary["exact_duplicates"] = duplicates(ds, split)
    print(f"[eda] duplicate check done ({time.time() - t0:.0f}s)", flush=True)
    s1, s2, n_px = np.zeros(3), np.zeros(3), 0          # exact per-channel moments, in chunks
    for i in range(0, len(train), 2000):
        x = ds.X[train[i:i + 2000]].reshape(-1, 3).astype(np.float64) / 255.0
        s1, s2, n_px = s1 + x.sum(0), s2 + (x * x).sum(0), n_px + len(x)
    summary["train_channel_mean"] = (s1 / n_px).tolist()
    summary["train_channel_std"] = np.sqrt(s2 / n_px - (s1 / n_px) ** 2).tolist()
    summary["value_range"] = [int(ds.X.min()), int(ds.X.max())]

    gray = features.to_gray(ds.X).astype(np.float32)
    grad = features.gradient_energy(ds.X)

    fig_class_counts(ds)
    fig_samples(ds, train, rng)
    summary["class_profiles_median"] = fig_profiles(ds, train, gray, grad)

    hist = np.sqrt(features.color_histogram(ds.X) / 3.0)          # Hellinger-mapped colour histogram
    summary["nn_colour"] = fig_nn_confusion(
        nn_confusion(hist[train], ds.y[train]), "eda_nn_confusion_colour.png",
        "Whose nearest neighbour is it? Colour histograms only (%)")
    fig_tsne(hist[train], ds.y[train], rng)
    summary["unusual"] = fig_unusual(ds, gray)
    summary["nearest_train_image_of_test_images"] = near_duplicates(ds, split)

    (C.TABLES_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("class_profiles_median",)},
                     indent=1)[:3500])


if __name__ == "__main__":
    main()
