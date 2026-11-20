"""The experiment grid: training recipes ("presets") and the ordered list of CNN runs.

Every hyper-parameter that the paper leaves open is fixed here, once, for all splits.
"""
from __future__ import annotations

from . import config as C
from .train import TrainConfig

# --------------------------------------------------------------------------- recipes
PRESETS: dict[str, dict] = {
    # Paper: "CNN (two layers)".  Native 64x64 input.
    "shallow": dict(arch="shallow", init="scratch", res=64, batch_size=128, epochs=40,
                    min_steps=2000, optimizer="adam", lr=1e-3, lr_final=0.0, warmup_epochs=0.0,
                    weight_decay=0.0),
    # Paper, Table II: ResNet-50 / GoogLeNet with random initialisation.
    "scratch": dict(init="scratch", res=128, batch_size=64, epochs=30, min_steps=3000,
                    optimizer="sgd", lr=0.05, lr_final=0.0, warmup_epochs=2.0, weight_decay=5e-4),
    # Paper, Table III: ImageNet-pretrained; "we first trained the last layer with a learning
    # rate of 0.01.  Afterwards, we fine-tuned through the entire network with a low learning
    # rate between 0.001 and 0.0001" -> 5 epochs last layer @ 0.01, then 15 epochs with the
    # learning rate decaying from 1e-3 to 1e-4 (cosine).
    "finetune": dict(init="imagenet", res=224, batch_size=64, head_epochs=5, head_lr=0.01,
                     head_min_steps=300, epochs=15, min_steps=1500, optimizer="sgd", lr=1e-3,
                     lr_final=1e-4, warmup_epochs=0.0, weight_decay=1e-4),
}


def make_cfg(preset: str, arch: str | None = None, train_frac: float = 0.8, seed: int = 0,
             **overrides) -> TrainConfig:
    kw = dict(PRESETS[preset])
    if arch is not None:
        kw["arch"] = arch
    kw.update(train_frac=train_frac, seed=seed)
    kw.update(overrides)
    return TrainConfig(**kw)


# ------------------------------------------------------------------------------ queue
# Splits are visited so that the most informative ones finish first.
FRAC_ORDER = [0.8, 0.1, 0.5, 0.9, 0.2, 0.3, 0.4, 0.6, 0.7]
assert sorted(FRAC_ORDER) == C.TRAIN_FRACS


def queue(name: str) -> list[tuple[TrainConfig, bool]]:
    """Ordered (config, save_model) pairs.

    ``core``   - everything needed for Table II, Table III and Fig. 8 (seed 0).
    ``seeds``  - the same experiments repeated with two more seeds (new split + new
                 initialisation) where they are cheap enough: the two-layer CNN on all nine
                 splits and the two fine-tuned networks on 80/20.
    """
    q: list[tuple[TrainConfig, bool]] = []
    if name in ("core", "all"):
        q.append((make_cfg("finetune", "resnet50", 0.8, 0), True))            # Table III, Fig. 8
        q.append((make_cfg("finetune", "googlenet", 0.8, 0), False))          # Table III
        q += [(make_cfg("shallow", None, f, 0), False) for f in FRAC_ORDER]   # Table II
        q += [(make_cfg("scratch", "googlenet", f, 0), False) for f in FRAC_ORDER]
        q += [(make_cfg("scratch", "resnet50", f, 0), False) for f in FRAC_ORDER]
    if name in ("seeds", "all"):
        for s in (1, 2):
            q += [(make_cfg("shallow", None, f, s), False) for f in FRAC_ORDER]
        for s in (1, 2):
            q.append((make_cfg("finetune", "resnet50", 0.8, s), False))
        for s in (1, 2):
            q.append((make_cfg("finetune", "googlenet", 0.8, s), False))
    if not q:
        raise ValueError(f"unknown queue {name!r}")
    return q
