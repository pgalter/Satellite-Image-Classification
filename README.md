# EuroSAT — minimal runnable branch

Only the code needed to prepare EuroSAT RGB and run SIFT + BoVW + SVM, a two-layer CNN,
ResNet-50 and GoogLeNet. Based on `eurosat-reproduction` commit `36e23c1`.
No historical predictions, plots, reports, generated splits or datasets are tracked here.
The original branch remains the reference for the previous report and results.

## Files

| Entry | Purpose |
|---|---|
| `eurosat/scripts/00_prepare_data.py` | Download, verify, decode and split the data |
| `eurosat/scripts/02a_tune_bovw.py` | Select a SIFT threshold separately for each split and seed |
| `eurosat/scripts/02_classical.py` | Train and test SIFT + BoVW + SVM |
| `eurosat/scripts/03_cnn.py` | Train and test the CNNs; includes a smoke run |
| `eurosat/src/eurosat/` | Shared data, feature, model and training implementation |

Run all commands below **inside `eurosat/`**, with the environment active.

## 1. Environment (CUDA)

Use Python 3.10 or 3.11 and a compatible NVIDIA driver. The original GPU experiments used
PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.8 and an 8 GB GPU.
CPU-only machines can run data preparation and SVM; the CNN entry point requires CUDA.

```bash
cd eurosat
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch, torchvision, cv2, sklearn; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Windows PowerShell activation: `.\.venv\Scripts\Activate.ps1`.
If a different CUDA wheel is required, select it for the machine's driver before installing
remaining requirements. Do not run CNN jobs until the check prints `CUDA: True`.

## 2. Data

```bash
python scripts/00_prepare_data.py
```

This downloads the authors' **EuroSAT_RGB.zip** (94,658,721 bytes), verifies its MD5, and
creates `data/EuroSAT_RGB/`, `data/cache/` and `splits/`. Expected output: 27,000 RGB images,
64 x 64 pixels; the 80/20 split has 19,440 train, 2,160 validation and 5,400 test images.

If automatic download fails, get **EuroSAT_RGB.zip** from
[the official Zenodo record](https://zenodo.org/records/7711810), put the ZIP at
`eurosat/data/raw/EuroSAT_RGB.zip`, then rerun the command. Do not use the multispectral ZIP.
MD5: `f46e308c4d50d4bf32fedad2d3d62f3b`.

## 3. First SVM experiment (CPU)

```bash
python scripts/02a_tune_bovw.py --fracs 0.8 --seeds 0 --jobs 2
python scripts/02_classical.py --ks 500 --fracs 0.8 --seeds 0 --jobs 2
```

The first command compares five SIFT thresholds and vocabulary sizes 100 and 500,
then selects the threshold with the highest validation accuracy at K=500.
The second command trains the selected pipeline and evaluates the held-out test set.
To also run K=10 and K=100, use `--ks 10 100 500`.

**Tune each split/seed separately.** Reusing the original 80/20 seed-0 choice on other
splits can use images that belong to those experiments' test sets. For the full SVM grid:

```bash
python scripts/02a_tune_bovw.py --fracs all --seeds 0 1 2 --jobs 2
python scripts/02_classical.py --ks 10 100 500 --fracs all --seeds 0 1 2 --jobs 2
```

This costs substantially more than the old single-split tuning. Start with the first
experiment before scheduling the full grid.

## 4. CNN check, then training (CUDA)

```bash
python scripts/03_cnn.py --smoke
python scripts/03_cnn.py --preset shallow --fracs 0.8 --seeds 0 --save-model
python scripts/03_cnn.py --preset finetune --arch resnet50 --fracs 0.8 --seeds 0 --save-model
```

The smoke run exercises all five recipes for one epoch per phase; its temporary result files
are removed on success. Pretrained weights are downloaded automatically on first use.
For GoogLeNet, replace `resnet50` with `googlenet`. For a deep network from random
initialisation, replace `finetune` with `scratch`.

Optional complete original CNN schedule:

```bash
python scripts/03_cnn.py --queue core
python scripts/03_cnn.py --queue seeds
```

If GPU memory is insufficient, pass `--batch-size 16 --eval-batch-size 32`.
Record these changes because training batch size changes the experiment. Use a new
`--tag` with `--preset` when changing a recipe. For a short shallow diagnostic, use
`--epochs 1 --min-steps 0 --tag diagnostic`; `--epochs` alone does not disable the
recipe's minimum-step requirement. Queue modes retain their fixed training recipes.

## Outputs and reruns

- `results/runs/*.json`: configuration, validation/test metrics; CNNs also include epoch logs.
- `results/preds/*.npz`: test indices, labels and predictions.
- `results/checkpoints/*.pt`: CNN weights when `--save-model` is used; SVM weights are not saved.
- `results/tables/bovw_tuning-tf*-s*.json` and `.csv`: per-split SIFT tuning results.

Completed experiments are skipped only when the result JSON and predictions both exist.
CNN config changes under the same name raise an error instead of silently reusing old metrics.
`--force` reruns and replaces the requested experiment; a run interrupted mid-training restarts
from the beginning, not from an intermediate epoch. CNN worker locks are released by the OS
on process exit; runs that failed twice make the queue exit nonzero until explicitly retried.
Use separate GPUs (e.g. `CUDA_VISIBLE_DEVICES`) if launching multiple CNN workers.

Generated artifacts are ignored by Git. The minimal branch intentionally omits EDA,
report generation, colour/pixel baselines and dense-SIFT variants. The SIFT feature method,
CNN architectures and default training recipes are retained; per-split threshold selection
is a protocol correction, so new BoVW results may differ from the original report.

## Validation scope

19 CPU regression and synthetic integration checks passed during preparation of this branch
(Linux, Python 3.12, PyTorch 2.7.1 CPU). These include all four entry points, split isolation,
SIFT/SVM integration, CNN forward/backward passes, checkpoint saving and worker locking.
No full EuroSAT training or CUDA benchmark is claimed here. Run data preparation and the
CUDA smoke command on the target machine before launching long experiments.
