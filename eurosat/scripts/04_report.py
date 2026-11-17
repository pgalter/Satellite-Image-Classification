"""Step 4 - turn the saved runs into the tables and figures of the report.

    python scripts/04_report.py

Reads results/runs/*.json and results/preds/*.npz; writes results/tables/*.md|csv and
figures/res_*.png.  Missing runs are skipped, so it can be called while experiments are running.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402
import re  # noqa: E402

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from eurosat import config as C  # noqa: E402
from eurosat import data, features  # noqa: E402
from eurosat import plotting as P  # noqa: E402
from eurosat.classical import SIFT_DEFAULT_CT  # noqa: E402
from eurosat.metrics import binomial_ci  # noqa: E402

FT_R50 = "resnet50-imagenet-r224"
FT_GLN = "googlenet-imagenet-r224"


# ------------------------------------------------------------------------- loading
def selected_ct() -> float:
    f = C.TABLES_DIR / "bovw_tuning.json"
    return json.loads(f.read_text())["selected_contrast_threshold"] if f.exists() else SIFT_DEFAULT_CT


def load_runs(ct: float) -> tuple[pd.DataFrame, dict]:
    """The runs of the paper's experiments (Table II, Table III); other runs in results/runs
    are ignored."""
    keep = set(labels(ct))
    rows, raw = [], {}
    for f in sorted(C.RUNS_DIR.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        cfg = r["config"]
        if r["kind"] == "cnn":
            method = f"{cfg['arch']}-{cfg['init']}-r{cfg['res']}" + (f"-{cfg['tag']}" if cfg["tag"] else "")
        else:
            method = f"{cfg['feature']}-{cfg['classifier']}"
        if method not in keep:
            continue
        raw[r["name"]] = r
        rows.append({
            "name": r["name"], "kind": r["kind"], "method": method,
            "train_frac": cfg["train_frac"], "split": r["split"], "seed": cfg["seed"],
            "test_acc": 100 * r["test"]["accuracy"], "val_acc": 100 * r["val"]["accuracy"],
            "macro_f1": 100 * r["test"]["macro_f1"], "n_train": r["n_train"], "n_test": r["n_test"],
            "minutes": r["train_seconds"] / 60,
            "head_only_test": 100 * r["head_only"]["test"]["accuracy"] if r.get("head_only") else np.nan,
        })
    return pd.DataFrame(rows), raw


def labels(ct: float) -> dict[str, str]:
    """Readable names of the paper's methods (keys = '<feature>-<clf>' or '<arch>-<init>-r<res>')."""
    return {
        f"sift-kp@{ct:g}-k10-svm": "BoVW (SIFT, k=10) + SVM",
        f"sift-kp@{ct:g}-k100-svm": "BoVW (SIFT, k=100) + SVM",
        f"sift-kp@{ct:g}-k500-svm": "BoVW (SIFT, k=500) + SVM",
        "shallow-scratch-r64": "CNN, two conv layers (64 px)",
        "googlenet-scratch-r128": "GoogLeNet, from scratch (128 px)",
        "resnet50-scratch-r128": "ResNet-50, from scratch (128 px)",
        FT_GLN: "GoogLeNet, ImageNet, fine-tuned (224 px)",
        FT_R50: "ResNet-50, ImageNet, fine-tuned (224 px)",
    }


def table2_methods(ct: float) -> dict[str, str]:
    """Row of the paper's Table II -> our method key."""
    return {
        "BoVW (SVM, SIFT, k=10)": f"sift-kp@{ct:g}-k10-svm",
        "BoVW (SVM, SIFT, k=100)": f"sift-kp@{ct:g}-k100-svm",
        "BoVW (SVM, SIFT, k=500)": f"sift-kp@{ct:g}-k500-svm",
        "CNN (two layers)": "shallow-scratch-r64",
        "ResNet-50": "resnet50-scratch-r128",
        "GoogLeNet": "googlenet-scratch-r128",
    }


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = [":---" if a == "l" else "---:" for a in align]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def by_split(df: pd.DataFrame, method: str, seed: int | None = 0, col: str = "test_acc") -> list[float]:
    """Values of ``col`` for the nine splits (NaN where the run is missing)."""
    d = df[df.method == method]
    if seed is not None:
        d = d[d.seed == seed]
    s = d.groupby("train_frac")[col].mean()
    return [float(s.get(f, np.nan)) for f in C.TRAIN_FRACS]


def fmt(v: float, nd: int = 2) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# ------------------------------------------------------------------------- Table II
def make_table2(df: pd.DataFrame, ct: float) -> None:
    rows, csv_rows = [], []
    for paper_name, method in table2_methods(ct).items():
        paper = C.PAPER_TABLE2[paper_name]
        ours = by_split(df, method, seed=0)
        rows.append([f"**{paper_name}**", "paper"] + [fmt(v) for v in paper])
        rows.append(["", "ours"] + [fmt(v) for v in ours])
        rows.append(["", "ours − paper"] + [("–" if np.isnan(o) else f"{o - p:+.2f}")
                                             for o, p in zip(ours, paper)])
        d = df[df.method == method]
        if d.seed.nunique() > 1:
            g = d.groupby("train_frac").test_acc
            mean, std, n = g.mean(), g.std(ddof=1), g.count()
            rows.append(["", f"ours, mean ± std ({int(n.max())} seeds)"] +
                        [(f"{mean.get(f, np.nan):.2f} ± {std.get(f, np.nan):.2f}"
                          if n.get(f, 0) > 1 else "–") for f in C.TRAIN_FRACS])
        csv_rows.append({"method": paper_name, "source": "paper", **dict(zip(C.SPLIT_NAMES, paper))})
        csv_rows.append({"method": paper_name, "source": "ours", **dict(zip(C.SPLIT_NAMES, ours))})
    (C.TABLES_DIR / "table2.md").write_text(
        md_table(["Method", ""] + C.SPLIT_NAMES, rows, "ll" + "r" * 9), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(C.TABLES_DIR / "table2.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), sharex=True, sharey=True)
    x = [int(round(100 * f)) for f in C.TRAIN_FRACS]
    for ax, (paper_name, method) in zip(axes.ravel(), table2_methods(ct).items()):
        ours = np.array(by_split(df, method, seed=0))
        paper = np.array(C.PAPER_TABLE2[paper_name])
        ax.plot(x, paper, color=P.NEUTRAL, marker="o", label="paper (Table II)")
        ok = ~np.isnan(ours)
        ax.plot(np.array(x)[ok], ours[ok], color=P.ACCENT, marker="o", label="this reproduction")
        ax.set_title(paper_name, fontsize=9.5)
        ax.set_xticks(x)
        ax.set_ylim(38, 100)
        if ok.any():
            up = ours[ok][-1] >= paper[-1]          # the higher end point gets its label above
            ax.annotate(f"{ours[ok][-1]:.1f}", (np.array(x)[ok][-1], ours[ok][-1]),
                        textcoords="offset points", xytext=(0, 7 if up else -13), ha="center",
                        fontsize=8, color=P.INK)
            ax.annotate(f"{paper[-1]:.1f}", (x[-1], paper[-1]), textcoords="offset points",
                        xytext=(0, -13 if up else 7), ha="center", fontsize=8, color=P.INK2)
    for ax in axes[-1]:
        ax.set_xlabel("share of images used for training (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)")
    handles, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="upper right", ncol=2, bbox_to_anchor=(0.985, 1.0))
    fig.suptitle("Table II: accuracy vs. training share", x=0.01, ha="left")
    fig.subplots_adjust(top=0.88, wspace=0.08, hspace=0.25)
    P.save(fig, C.FIG_DIR / "res_table2_curves.png")


# ------------------------------------------------------------------------ Table III
def mcnemar(a: str, b: str) -> dict | None:
    fa, fb = C.PREDS_DIR / f"{a}.npz", C.PREDS_DIR / f"{b}.npz"
    if not (fa.exists() and fb.exists()):
        return None
    za, zb = np.load(fa), np.load(fb)
    assert np.array_equal(za["idx"], zb["idx"])
    ca, cb = za["y_pred"] == za["y_true"], zb["y_pred"] == zb["y_true"]
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": float(p)}


def make_table3(df: pd.DataFrame, raw: dict) -> None:
    rows = []
    for paper_name, method in (("ResNet-50", FT_R50), ("GoogLeNet", FT_GLN)):
        d = df[(df.method == method) & (df.train_frac == 0.8)].sort_values("seed")
        if d.empty:
            continue
        s0 = d[d.seed == 0].iloc[0]
        lo, hi = binomial_ci(s0.test_acc / 100, int(s0.n_test))
        seeds = (f"{d.test_acc.mean():.2f} ± {d.test_acc.std(ddof=1):.2f} (n={len(d)})"
                 if len(d) > 1 else "–")
        rows.append([paper_name, fmt(C.PAPER_TABLE3[paper_name]), fmt(s0.test_acc),
                     f"[{100 * lo:.2f}, {100 * hi:.2f}]", seeds, fmt(s0.head_only_test),
                     fmt(s0.val_acc)])
    out = md_table(["Network (ImageNet-pretrained, 80/20)", "paper", "ours (seed 0)",
                    "95% CI (seed 0)", "ours, mean ± std over seeds", "last layer only", "val acc"], rows)
    m = mcnemar(f"{FT_R50}-tf80-s0", f"{FT_GLN}-tf80-s0")
    if m:
        out += (f"\nMcNemar test, ResNet-50 vs GoogLeNet on the same 5,400 test images (seed 0): "
                f"{m['only_first_correct']} images only ResNet-50 gets right, "
                f"{m['only_second_correct']} only GoogLeNet; exact p = {m['p_value']:.3f}.\n")
        (C.TABLES_DIR / "mcnemar.json").write_text(json.dumps(m, indent=2))
    (C.TABLES_DIR / "table3.md").write_text(out, encoding="utf-8")


# ------------------------------------------------- the paper's approaches side by side
def make_ladder(df: pd.DataFrame, ct: float) -> None:
    lab = labels(ct)
    d = df[(df.train_frac == 0.8) & (df.seed == 0) & df.method.isin(lab)].copy()
    if d.empty:
        return
    d["label"] = d.method.map(lab)
    extra = d[d.method == FT_R50]
    if not extra.empty and not np.isnan(extra.iloc[0].head_only_test):
        r = extra.iloc[0].copy()
        r["label"], r["test_acc"], r["method"] = "ResNet-50, ImageNet, last layer only (224 px)", r.head_only_test, "linear-probe"
        r["val_acc"], r["macro_f1"], r["minutes"] = np.nan, np.nan, np.nan
        d = pd.concat([d, r.to_frame().T], ignore_index=True)
    d["test_acc"] = d.test_acc.astype(float)
    d = d.sort_values("test_acc")
    paper80 = {m: C.PAPER_TABLE2[p][7] for p, m in table2_methods(ct).items()}
    paper80.update({FT_R50: C.PAPER_TABLE3["ResNet-50"], FT_GLN: C.PAPER_TABLE3["GoogLeNet"]})

    rows = [[r.label, fmt(float(r.test_acc)), fmt(float(r.val_acc)), fmt(float(r.macro_f1)),
             fmt(paper80.get(r.method, np.nan)), fmt(float(r.minutes), 1)]
            for r in d.sort_values("test_acc", ascending=False).itertuples()]
    (C.TABLES_DIR / "ladder_80_20.md").write_text(md_table(
        ["Method (80/20 split, seed 0)", "test acc", "val acc", "macro-F1",
         "paper", "minutes"], rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.6, 0.33 * len(d) + 1.1))
    ypos = np.arange(len(d))
    ax.barh(ypos, d.test_acc, height=0.5, color=P.ACCENT)
    first = True
    for y, r in zip(ypos, d.itertuples()):
        ax.text(r.test_acc + 0.8, y, f"{r.test_acc:.1f}", va="center", fontsize=8.5, color=P.INK)
        if r.method in paper80:
            ax.plot([paper80[r.method]] * 2, [y - 0.36, y + 0.36], color=P.INK, linewidth=1.6,
                    solid_capstyle="butt", label="paper" if first else None)
            first = False
    ax.set_yticks(ypos, d.label)
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("test accuracy (%), 80/20 split")
    ax.grid(axis="y", visible=False)
    if not first:
        ax.legend(loc="lower right")
    ax.set_title("The paper's approaches on the same 5,400 test images (80/20)")
    P.save(fig, C.FIG_DIR / "res_ladder_80_20.png")


def make_tuning_table(ct: float) -> None:
    f = C.TABLES_DIR / "bovw_tuning.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    rows = []
    for thr in sorted(t.contrast_threshold.unique(), reverse=True):
        d = t[t.contrast_threshold == thr].set_index("k")
        name = f"{thr:g}" + (" (OpenCV default)" if thr == SIFT_DEFAULT_CT else "") + (
            " **(selected)**" if thr == ct else "")
        rows.append([name, fmt(float(d.keypoints_per_image.iloc[0]), 1),
                     f"{100 * float(d.images_without_keypoint.iloc[0]):.1f}%",
                     fmt(100 * float(d.loc[100, "val_acc"])), fmt(100 * float(d.loc[500, "val_acc"]))])
    (C.TABLES_DIR / "bovw_tuning.md").write_text(md_table(
        ["Detector contrast threshold", "keypoints per image", "images without keypoint",
         "validation acc., k = 100", "validation acc., k = 500"], rows), encoding="utf-8")


def make_keypoint_figures(ds: data.EuroSAT, ct: float) -> None:
    split = data.make_split(ds.y, 0.8, 0)
    rng = np.random.default_rng(3)
    picks = [int(rng.choice(split["train"][ds.y[split["train"]] == c])) for c in range(C.N_CLASSES)]
    imgs = []
    for thr in (SIFT_DEFAULT_CT, ct):
        sift = cv2.SIFT_create(contrastThreshold=thr)
        for i in picks:
            big = cv2.resize(ds.X[i], (256, 256), interpolation=cv2.INTER_NEAREST)
            for kp in sift.detect(cv2.cvtColor(ds.X[i], cv2.COLOR_RGB2GRAY), None):
                centre = (int(round(kp.pt[0] * 4 + 2)), int(round(kp.pt[1] * 4 + 2)))
                cv2.circle(big, centre, max(3, int(round(kp.size * 2))), (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(big, centre, max(3, int(round(kp.size * 2))), (11, 11, 11), 1, cv2.LINE_AA)
            imgs.append(big)
    fig = P.image_grid(imgs, n_cols=C.N_CLASSES, titles=C.CLASS_SHORT + [""] * C.N_CLASSES,
                       row_labels=[f"default detector\n(threshold {SIFT_DEFAULT_CT:g})",
                                   f"tuned detector\n(threshold {ct:g})"],
                       cell=0.95, label_width=1.25, title_size=7.5,
                       suptitle="SIFT keypoints (circle = keypoint scale) on one training patch per class")
    P.save(fig, C.FIG_DIR / "res_sift_keypoints.png")

    f = C.TABLES_DIR / "bovw_tuning.csv"
    if not f.exists():
        return
    thresholds = sorted(pd.read_csv(f).contrast_threshold.unique(), reverse=True)
    m = np.zeros((len(thresholds), C.N_CLASSES))
    for a, thr in enumerate(thresholds):
        _, counts = features.sift_keypoint_descriptors(ds.X, float(thr))
        m[a] = [100 * (counts[ds.y == c] == 0).mean() for c in range(C.N_CLASSES)]
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    P.heatmap(ax, m, [f"{t:g}" for t in thresholds], C.CLASS_SHORT, fmt="{:.0f}", vmax=100, hide_below=0.5)
    ax.set_ylabel("detector contrast\nthreshold")
    ax.set_title("Images without a single SIFT keypoint (%)")
    P.save(fig, C.FIG_DIR / "res_sift_empty_images.png")


# --------------------------------------------------------------------- error analysis
def make_confusions(df: pd.DataFrame, raw: dict, ct: float) -> None:
    r = raw.get(f"{FT_R50}-tf80-s0")
    if r is not None:
        cm = np.array(r["test"]["confusion_matrix"])
        pct = 100.0 * cm / cm.sum(1, keepdims=True)
        err = cm.astype(float).copy()
        np.fill_diagonal(err, np.nan)
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
        P.heatmap(axes[0], 100 * C.PAPER_FIG8, C.CLASS_SHORT, C.CLASS_SHORT, fmt="{:.0f}", vmax=100, hide_below=0.5)
        axes[0].set_title("Paper, Fig. 8 (row-normalised, %)")
        P.heatmap(axes[1], pct, C.CLASS_SHORT, C.CLASS_SHORT, fmt="{:.1f}", vmax=100, hide_below=0.05, fontsize=7)
        axes[1].set_title("This reproduction (row-normalised, %)")
        P.heatmap(axes[2], err, C.CLASS_SHORT, C.CLASS_SHORT, fmt="{:.0f}", vmax=float(np.nanmax(err)), hide_below=0.5)
        axes[2].set_title(f"This reproduction: the {int(np.nansum(err))} errors (image counts)")
        for ax in axes:
            ax.set_xlabel("predicted class")
        axes[0].set_ylabel("true class")
        fig.suptitle("Fine-tuned ResNet-50, 80/20 split: confusion matrix on the 5,400 test images",
                     x=0.01, ha="left")
        fig.subplots_adjust(wspace=0.22, top=0.86)
        P.save(fig, C.FIG_DIR / "res_confusion_resnet50.png")

        t = r["test"]
        rows = [[C.CLASSES[c], t["support"][c], fmt(100 * t["precision"][c]), fmt(100 * t["recall"][c]),
                 fmt(100 * t["f1"][c])] for c in range(C.N_CLASSES)]
        rows.append(["**all (accuracy / macro-F1)**", sum(t["support"]), "", fmt(100 * t["accuracy"]),
                     fmt(100 * t["macro_f1"])])
        (C.TABLES_DIR / "per_class_resnet50.md").write_text(md_table(
            ["Class", "test images", "precision", "recall", "F1"], rows), encoding="utf-8")
        off = cm.copy()
        np.fill_diagonal(off, 0)
        pairs = sorted(((int(off[i, j] + off[j, i]), i, j) for i in range(C.N_CLASSES)
                        for j in range(i + 1, C.N_CLASSES)), reverse=True)[:8]
        rows = [[f"{C.CLASSES[i]} / {C.CLASSES[j]}", int(off[i, j]), int(off[j, i]), n,
                 f"{100 * n / off.sum():.0f}%"] for n, i, j in pairs if n > 0]
        (C.TABLES_DIR / "confused_pairs_resnet50.md").write_text(md_table(
            ["Class pair (A / B)", "A predicted as B", "B predicted as A", "errors",
             "share of all errors"], rows), encoding="utf-8")

    panels = [(f"sift-kp@{ct:g}-k500-svm", "BoVW (SIFT, k=500) + SVM"),
              ("shallow-scratch-r64", "CNN, two layers"),
              ("googlenet-scratch-r128", "GoogLeNet from scratch"),
              ("resnet50-scratch-r128", "ResNet-50 from scratch"),
              (FT_GLN, "GoogLeNet fine-tuned"),
              (FT_R50, "ResNet-50 fine-tuned")]
    have = [(m, n) for m, n in panels if f"{m}-tf80-s0" in raw]
    if have:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.4))
        for ax, (m, n) in zip(axes.ravel(), have):
            rr = raw[f"{m}-tf80-s0"]
            cm = np.array(rr["test"]["confusion_matrix"])
            P.heatmap(ax, 100.0 * cm / cm.sum(1, keepdims=True), C.CLASS_SHORT, C.CLASS_SHORT,
                      fmt="{:.0f}", vmax=100, hide_below=0.5, fontsize=7)
            ax.set_title(f"{n}  -  {100 * rr['test']['accuracy']:.1f}%")
        for ax in axes.ravel()[len(have):]:
            ax.set_visible(False)
        fig.suptitle("Row-normalised confusion matrices (%), 80/20 split; rows = true class, columns = prediction",
                     x=0.01, ha="left")
        fig.subplots_adjust(wspace=0.25, hspace=0.38, top=0.92)
        P.save(fig, C.FIG_DIR / "res_confusion_methods.png")

    lab = labels(ct)
    d = df[(df.train_frac == 0.8) & (df.seed == 0) & df.method.isin(lab)].sort_values("test_acc")
    if len(d):
        m = np.array([100 * np.array(raw[n]["test"]["recall"]) for n in d.name])
        fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(d) + 1.5))
        P.heatmap(ax, m, [lab[k] for k in d.method], C.CLASS_SHORT, fmt="{:.0f}", vmin=0, vmax=100)
        ax.set_title("Recall per class (%), 80/20 split - methods ordered by overall accuracy")
        P.save(fig, C.FIG_DIR / "res_per_class_recall.png")
        pd.DataFrame(m, index=[lab[k] for k in d.method], columns=C.CLASSES).round(2).to_csv(
            C.TABLES_DIR / "per_class_recall.csv")


def make_error_examples(ds: data.EuroSAT) -> None:
    f = C.PREDS_DIR / f"{FT_R50}-tf80-s0.npz"
    if not f.exists():
        return
    z = np.load(f)
    idx, yt, yp, prob = z["idx"], z["y_true"], z["y_pred"], z["prob"].astype(np.float32)
    wrong = np.flatnonzero(yt != yp)
    conf = prob[np.arange(len(yt)), yp]
    p_true = prob[np.arange(len(yt)), yt]
    pairs = pd.Series(list(zip(yt[wrong], yp[wrong]))).value_counts()
    n_cols, n_rows = 6, min(6, len(pairs))
    imgs, titles, row_labels = [], [], []
    blank = np.full((C.IMG_SIZE, C.IMG_SIZE, 3), 252, dtype=np.uint8)
    for (t, p), n in list(pairs.items())[:n_rows]:
        sel = wrong[(yt[wrong] == t) & (yp[wrong] == p)]
        sel = sel[np.argsort(-conf[sel])][:n_cols]
        row_labels.append(f"{C.CLASS_SHORT[t]}\npredicted {C.CLASS_SHORT[p]}\n({n} images)")
        for k in range(n_cols):
            if k < len(sel):
                imgs.append(ds.X[idx[sel[k]]])
                titles.append(f"p={conf[sel[k]]:.2f}")
            else:
                imgs.append(blank)
                titles.append("")
    fig = P.image_grid(imgs, n_cols=n_cols, titles=titles, row_labels=row_labels, cell=0.95,
                       label_width=1.75, title_size=7,
                       suptitle="Fine-tuned ResNet-50: test errors of the most frequent confusions "
                                "(p = confidence in the wrong class)")
    P.save(fig, C.FIG_DIR / "res_errors_resnet50.png")

    # the errors the network is most sure about (candidates for ambiguous or doubtful labels)
    top = wrong[np.argsort(-conf[wrong])][:16]
    fig = P.image_grid([ds.X[idx[i]] for i in top], n_cols=8,
                       titles=[f"{C.CLASS_SHORT[yt[i]]}\nas {C.CLASS_SHORT[yp[i]]}, p={conf[i]:.2f}"
                               for i in top], cell=1.12, title_size=7,
                       suptitle="The 16 most confident errors of the fine-tuned ResNet-50 "
                                "(label, prediction, confidence)")
    P.save(fig, C.FIG_DIR / "res_errors_confident.png")

    # errors inside the three groups of look-alike classes identified during exploration
    groups = {"vegetation / agriculture": ["AnnualCrop", "HerbaceousVegetation", "Pasture", "PermanentCrop"],
              "linear structures": ["Highway", "River"],
              "built-up": ["Industrial", "Residential"]}
    within = {}
    for g, names in groups.items():
        ids = [C.CLASSES.index(n) for n in names]
        within[g] = int((np.isin(yt[wrong], ids) & np.isin(yp[wrong], ids)).sum())
    within["other"] = int(len(wrong) - sum(within.values()))

    # confidence of errors vs. correct predictions, and how many errors are 'confident'
    stats = {"n_test": int(len(yt)), "n_errors": int(len(wrong)), "errors_within_group": within,
             "mean_confidence_correct": float(conf[yt == yp].mean()),
             "mean_confidence_errors": float(conf[wrong].mean()),
             "errors_with_confidence_above_0.9": int((conf[wrong] > 0.9).sum()),
             "errors_true_class_is_second_choice": int(sum(
                 np.argsort(-prob[i])[1] == yt[i] for i in wrong)),
             "mean_prob_of_true_class_on_errors": float(p_true[wrong].mean()),
             "error_files": [{"file": ds.paths[idx[i]], "true": C.CLASSES[yt[i]],
                              "pred": C.CLASSES[yp[i]], "confidence": float(conf[i])}
                             for i in wrong[np.argsort(-conf[wrong])]]}
    (C.TABLES_DIR / "errors_resnet50.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


def make_training_curves(raw: dict) -> None:
    runs = [(f"{FT_R50}-tf80-s0", "ResNet-50 fine-tuned (224 px)"),
            ("resnet50-scratch-r128-tf80-s0", "ResNet-50 from scratch (128 px)"),
            ("shallow-scratch-r64-tf80-s0", "CNN, two layers (64 px)")]
    have = [(n, lab) for n, lab in runs if n in raw]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for (n, lab), color in zip(have, P.SERIES):
        h = raw[n]["history"]
        ep = [e["epoch"] for e in h]
        va = [100 * e["val_acc"] for e in h]
        ax.plot(ep, va, color=color, label=lab)
        ax.annotate(f"{lab}  {va[-1]:.1f}", (ep[-1], va[-1]), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=8.5, color=P.INK)
        n_head = sum(e["phase"] == "head" for e in h)
        if n_head:
            ax.axvline(n_head + 0.5, color=P.AXIS, linewidth=0.8)
            ax.text(n_head + 0.3, 71, "last layer only ", ha="right", fontsize=8, color=P.INK2)
            ax.text(n_head + 0.7, 71, " whole network", ha="left", fontsize=8, color=P.INK2)
    ax.set_ylim(70, 100)
    ax.set_xlim(0, max(len(raw[n]["history"]) for n, _ in have) * 1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(loc="lower right")
    ax.set_title("Validation accuracy during training (80/20 split)")
    P.save(fig, C.FIG_DIR / "res_training_curves.png")


def build_report() -> None:
    """REPORT.md = report/REPORT.template.md with every ``{{include:<path>}}`` line replaced by
    the content of that file (the generated tables), so numbers are never copied by hand."""
    template = C.ROOT / "report" / "REPORT.template.md"
    if not template.exists():
        return
    missing = []

    def repl(m: re.Match) -> str:
        f = C.ROOT / m.group(1)
        if not f.exists():
            missing.append(m.group(1))
            return f"*(not available yet: `{m.group(1)}`)*"
        return f.read_text(encoding="utf-8").strip()

    text = re.sub(r"\{\{include:([^}]+)\}\}", repl, template.read_text(encoding="utf-8"))
    # discussion paragraphs that are not written yet (their experiments are still running)
    text = re.sub(r"^%%[A-Z0-9_]+%%$", "*(Discussion pending: these experiments are still running.)*",
                  text, flags=re.M)
    (C.ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"[report] wrote {C.ROOT / 'REPORT.md'}" + (f" (missing: {missing})" if missing else ""))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):          # tables contain non-ASCII dashes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    C.ensure_dirs()
    P.apply_style()
    ct = selected_ct()
    df, raw = load_runs(ct)
    df.sort_values(["method", "train_frac", "seed"]).to_csv(C.TABLES_DIR / "all_runs.csv", index=False)
    print(f"[report] {len(df)} runs, {df.method.nunique()} methods, SIFT contrast threshold {ct:g}")
    ds = data.load_dataset(verbose=False)
    make_table2(df, ct)
    make_table3(df, raw)
    make_ladder(df, ct)
    make_tuning_table(ct)
    make_keypoint_figures(ds, ct)
    make_confusions(df, raw, ct)
    make_error_examples(ds)
    make_training_curves(raw)
    build_report()
    for name in ("table2.md", "table3.md", "ladder_80_20.md"):
        f = C.TABLES_DIR / name
        if f.exists():
            print(f"\n--- {name}\n{f.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
