# EuroSAT: reproducing the land-use / land-cover benchmark (RGB)

Reproduction of the RGB experiments of

> P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark
> for Land Use and Land Cover Classification.* arXiv:1709.00029v2 (IEEE JSTARS, 2019).

organised along the four parts of the course project: data exploration, a reproducible image
pipeline, the paper's classification approaches (SIFT bag of visual words, a two-layer CNN,
ResNet-50 and GoogLeNet from scratch and ImageNet-pretrained) and error analysis. Only the
paper's own experiments are included.

**The write-up is [REPORT.md](REPORT.md).** This file explains how to run the code.

## What is reproduced

| Paper | Here |
|---|---|
| Table II - BoVW (SIFT + SVM, k = 10/100/500), two-layer CNN, ResNet-50 and GoogLeNet trained from scratch on nine train/test splits | yes, all nine splits |
| Table III - ImageNet-pretrained ResNet-50 / GoogLeNet fine-tuned on the 80/20 split (EuroSAT column) | yes |
| Fig. 8 - confusion matrix of the fine-tuned ResNet-50 | yes |
| Table III (UCM, AID, SAT-6, BCS), Table IV and Fig. 9 (13-band multispectral data), Sec. V (applications) | out of scope: the project uses the RGB version of EuroSAT only |

## Layout

```
src/eurosat/         library code
  config.py          paths, class names, the paper's numbers
  data.py            download + checksum, decoding, class-wise train/val/test splits
  features.py        SIFT (sparse / dense), bag of visual words, colour and pixel descriptors
  classical.py       representation -> SVM / logistic regression, model selection on validation data
  models.py          two-layer CNN, ResNet-50, GoogLeNet
  train.py           GPU training loop (in-memory data, augmentation, two-stage fine-tuning)
  experiments.py     the training recipes and the ordered list of CNN runs
  metrics.py         accuracy, per-class precision/recall/F1, confusion matrix, confidence interval
  plotting.py        one figure style
scripts/             the pipeline, step by step (see below)
splits/              every split as lists of image indices (+ index.json: index -> file name)
results/runs/        one JSON per experiment (config, validation curve, test metrics)
results/preds/       test predictions of every experiment
results/tables/      tables of the report (Markdown / CSV / JSON)
figures/             figures of the report
data/                dataset, caches, pretrained weights (created by the scripts, not versioned)
```

## Setup

Python 3.10+, a CUDA GPU for the CNN experiments (8 GB is enough).

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`cu128` is required for RTX 50-series GPUs; older cards work with other CUDA builds of PyTorch.

## Running the pipeline

Every script is idempotent: finished experiments are skipped, so an interrupted step can simply
be started again.

| Step | Command | What it does | Time* |
|---|---|---|---|
| 0 | `python scripts/00_prepare_data.py` | downloads `EuroSAT_RGB.zip` from the authors' Zenodo record (94.7 MB), checks size + MD5, extracts and decodes the 27,000 images, writes the splits | 3 min |
| 1 | `python scripts/01_eda.py` | exploration figures (`figures/eda_*.png`) and `results/tables/eda_summary.json` | 3 min |
| 2a | `python scripts/02a_tune_bovw.py` | chooses the SIFT detector threshold on validation data | 20 min (CPU) |
| 2 | `python scripts/02_classical.py --paper-only --seeds 0 1 2 --jobs 8` | SIFT bag-of-words + SVM (k = 10 / 100 / 500) on the nine splits, three seeds | ~1 h (CPU) |
| 3 | `python scripts/03_cnn.py --queue core` | Table II / Table III / Fig. 8 runs: two-layer CNN, ResNet-50 and GoogLeNet from scratch and fine-tuned | ~2 h (GPU) |
| 3b | `python scripts/03_cnn.py --queue seeds` | seeds 1 and 2 of the two-layer CNN (all splits) and of the fine-tuned networks (80/20) | ~1.5 h (GPU) |
| 4 | `python scripts/04_report.py` | builds all result tables and figures from the saved runs, and `REPORT.md` from `report/REPORT.template.md` | 1 min |
| - | `python scripts/99_check_pipeline.py` | sanity checks: splits are class-wise, disjoint, nested and deterministic; augmentation is a label-preserving symmetry; feature builders behave as documented | 10 s |

*on a laptop with a Ryzen 9 8945HX, 32 GB RAM and an RTX 5060 (8 GB).

Single experiments can be run directly, for example

```bash
python scripts/03_cnn.py --preset finetune --arch resnet50 --fracs 0.8 --seeds 0
python scripts/02_classical.py --features sift-kp@0.005-k500 --fracs 0.8
python scripts/03_cnn.py --smoke          # two-minute end-to-end check of all training recipes
```

## Evaluation protocol in one paragraph

For a training share *r* (10% ... 90%), *r* of **every class** forms the training portion and
the rest is the test set, as in the paper. One tenth of each training portion is held out as a
validation set. Everything that is learned (visual vocabulary, feature scaling, network weights)
is fitted on the training images only; the validation images choose the SVM constant `C` and the
training epoch whose weights are kept; the test images are used once, to report the result.
Splits are deterministic functions of `(train share, seed)` and nested across training shares.
Details and all assumptions are in [REPORT.md](REPORT.md).

## Notes on reproducibility

* Splits, vocabulary sampling, weight initialisation, shuffling and augmentation are seeded.
  GPU convolution algorithms (cuDNN) are not bit-deterministic, so repeated CNN runs differ in
  the last digits (the report quantifies the run-to-run spread with three seeds).
* The dataset archive is verified against the MD5 published on Zenodo
  (`f46e308c4d50d4bf32fedad2d3d62f3b`).
* Pretrained weights are torchvision's `ResNet50_Weights.IMAGENET1K_V1` and
  `GoogLeNet_Weights.IMAGENET1K_V1`.
