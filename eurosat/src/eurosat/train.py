"""Training and evaluation of the CNNs.

The whole dataset (27,000 x 3 x 64 x 64 uint8 = 332 MB) lives on the GPU; batches are built there
(indexing -> augmentation -> upsampling -> normalisation), so there are no data-loader workers and
every run sees exactly the images selected by its split.

Protocol of one run (identical for every network):
    1. take the class-wise train / val / test split for (train_frac, seed);
    2. train on *train* only;
    3. after every epoch measure accuracy on *val* and remember the best weights;
    4. evaluate those weights once on *test*.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from . import data as D
from .metrics import summarize
from .models import build_model, count_parameters, head_parameters


@dataclass
class TrainConfig:
    arch: str                      # "shallow" | "resnet50" | "googlenet"
    init: str = "scratch"          # "scratch" (random init) | "imagenet" (ILSVRC-2012 weights)
    train_frac: float = 0.8
    seed: int = 0
    res: int = 64                  # network input side; the 64x64 patch is bilinearly upsampled
    batch_size: int = 64
    epochs: int = 30               # epochs of full-network training
    min_steps: int = 0             # ...but at least this many optimiser steps (small splits)
    optimizer: str = "sgd"         # "sgd" (Nesterov, momentum 0.9) | "adam"
    lr: float = 0.05               # peak learning rate of the full-network phase
    lr_final: float = 0.0          # learning rate at the end of the cosine schedule
    warmup_epochs: float = 2.0     # linear warm-up
    weight_decay: float = 5e-4
    head_epochs: int = 0           # fine-tuning stage 1: only the last layer is trained
    head_min_steps: int = 0
    head_lr: float = 0.01
    augment: bool = True           # random flips + transposition (8 symmetries of the square)
    amp: bool = True               # mixed precision
    tag: str = ""

    @property
    def name(self) -> str:
        n = (f"{self.arch}-{self.init}-r{self.res}"
             f"-tf{int(round(self.train_frac * 100)):02d}-s{self.seed}")
        return n + (f"-{self.tag}" if self.tag else "")


class GPUData:
    """The decoded dataset as uint8 tensors on the training device."""

    def __init__(self, ds: D.EuroSAT, device: torch.device):
        self.device = device
        self.X = torch.from_numpy(ds.X).to(device).permute(0, 3, 1, 2).contiguous()  # N,3,64,64
        self.y = torch.from_numpy(ds.y).to(device)
        self.y_np = ds.y


def channel_stats(gd: GPUData, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean / std (0..1 scale) of the images ``idx`` - computed on training data only."""
    x = gd.X[torch.as_tensor(idx, device=gd.device)].float().div_(255)
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def d4_augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply to every image one of the 8 symmetries of the square, chosen uniformly at random.

    A satellite patch has no canonical orientation, so flips and 90-degree rotations preserve the
    label by construction.  No colour or scale jitter is used.
    """
    b = x.shape[0]
    r = torch.rand(3, b, device=x.device, generator=gen) < 0.5
    x = torch.where(r[0].view(b, 1, 1, 1), x.flip(3), x)
    x = torch.where(r[1].view(b, 1, 1, 1), x.flip(2), x)
    x = torch.where(r[2].view(b, 1, 1, 1), x.transpose(2, 3), x)
    return x


def make_batch(gd: GPUData, idx: torch.Tensor, cfg: TrainConfig, mean: torch.Tensor,
               std: torch.Tensor, train: bool, gen: torch.Generator | None = None) -> torch.Tensor:
    x = gd.X[idx].float().div_(255)
    if train and cfg.augment:
        x = d4_augment(x, gen)
    if cfg.res != x.shape[-1]:
        x = F.interpolate(x, size=(cfg.res, cfg.res), mode="bilinear", align_corners=False)
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


@torch.no_grad()
def predict_logits(model: nn.Module, gd: GPUData, idx: np.ndarray, cfg: TrainConfig,
                   mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Logits (float32, on CPU) for the images ``idx``, in eval mode."""
    was_training = model.training
    model.eval()
    bs = 128 if cfg.res > 128 else 512
    idx_t = torch.as_tensor(idx, device=gd.device)
    out = []
    for i in range(0, len(idx_t), bs):
        x = make_batch(gd, idx_t[i:i + bs], cfg, mean, std, train=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            out.append(model(x).float().cpu())
    model.train(was_training)
    return torch.cat(out)


def _accuracy(logits: torch.Tensor, y: np.ndarray) -> float:
    return float((logits.argmax(1).numpy() == y).mean())


def _loss(out, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy; GoogLeNet trained from scratch adds its two auxiliary heads (weight 0.3)."""
    if isinstance(out, tuple):   # torchvision GoogLeNetOutputs(logits, aux_logits2, aux_logits1)
        loss = F.cross_entropy(out.logits, y) + 0.3 * (
            F.cross_entropy(out.aux_logits1, y) + F.cross_entropy(out.aux_logits2, y))
        return loss, out.logits
    return F.cross_entropy(out, y), out


def run_experiment(cfg: TrainConfig, gd: GPUData, force: bool = False,
                   save_model: bool = False, verbose: bool = True) -> dict:
    """Train + evaluate one configuration; results go to ``results/runs/<name>.json``."""
    C.ensure_dirs()
    out_json = C.RUNS_DIR / f"{cfg.name}.json"
    if out_json.exists() and not force:
        if verbose:
            print(f"[{cfg.name}] already done, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))

    device = gd.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    shuffle_gen = torch.Generator().manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    split = D.make_split(gd.y_np, cfg.train_frac, cfg.seed)
    tr_idx = torch.as_tensor(split["train"], device=device)
    n_train = len(tr_idx)
    steps_per_epoch = n_train // cfg.batch_size          # last partial batch is dropped

    pretrained = cfg.init == "imagenet"
    if pretrained:   # the statistics the pretrained weights expect
        mean = torch.tensor(C.IMAGENET_MEAN, device=device)
        std = torch.tensor(C.IMAGENET_STD, device=device)
    else:            # statistics of this split's training images
        mean, std = channel_stats(gd, split["train"])
    model = build_model(cfg.arch, pretrained).to(device)

    def make_opt(params, lr: float):
        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True,
                               weight_decay=cfg.weight_decay)

    # ---- phases: optional "head" (last layer only, as in the paper) followed by "full"
    phases = []
    if cfg.head_epochs > 0:
        phases.append(("head", max(cfg.head_epochs, math.ceil(cfg.head_min_steps / steps_per_epoch))))
    full_epochs = max(cfg.epochs, math.ceil(cfg.min_steps / steps_per_epoch)) if cfg.epochs > 0 else 0
    if full_epochs > 0:
        phases.append(("full", full_epochs))

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    history: list[dict] = []
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    head_only = None
    epoch = 0
    total_steps = 0
    t_start = time.time()

    for phase, n_epochs in phases:
        if phase == "head":
            for p in model.parameters():
                p.requires_grad_(False)
            params = head_parameters(model)
            for p in params:
                p.requires_grad_(True)
            opt = make_opt(params, cfg.head_lr)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            opt = make_opt(model.parameters(), cfg.lr)
        phase_steps = n_epochs * steps_per_epoch
        warm = int(cfg.warmup_epochs * steps_per_epoch) if phase == "full" else 0
        step_in_phase = 0

        for _ in range(n_epochs):
            epoch += 1
            # frozen backbone => keep BatchNorm statistics fixed too (pure linear classifier phase)
            model.train(phase == "full")
            perm = tr_idx[torch.randperm(n_train, generator=shuffle_gen).to(device)]
            run_loss, run_correct, run_n = 0.0, 0, 0
            t_ep = time.time()
            for b in range(steps_per_epoch):
                if phase == "head":
                    lr = cfg.head_lr
                elif step_in_phase < warm:
                    lr = cfg.lr * (step_in_phase + 1) / warm
                else:
                    prog = (step_in_phase - warm) / max(1, phase_steps - warm)
                    lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * prog))
                for g in opt.param_groups:
                    g["lr"] = lr
                bi = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
                x = make_batch(gd, bi, cfg, mean, std, train=True, gen=gen)
                yb = gd.y[bi]
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                    loss, logits = _loss(model(x), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                step_in_phase += 1
                total_steps += 1
                run_loss += loss.item() * len(bi)
                run_correct += int((logits.argmax(1) == yb).sum())
                run_n += len(bi)
            if not math.isfinite(run_loss):
                raise RuntimeError(f"[{cfg.name}] training diverged (loss is not finite)")
            val_logits = predict_logits(model, gd, split["val"], cfg, mean, std)
            val_acc = _accuracy(val_logits, gd.y_np[split["val"]])
            if val_acc >= best["val_acc"]:   # ties -> the later (longer trained) epoch
                best = {"val_acc": val_acc, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            history.append({"epoch": epoch, "phase": phase, "lr": lr,
                            "train_loss": run_loss / run_n, "train_acc": run_correct / run_n,
                            "val_acc": val_acc, "seconds": time.time() - t_ep})
            if verbose:
                print(f"[{cfg.name}] ep {epoch:3d} {phase:4s} lr {lr:.5f} "
                      f"loss {run_loss / run_n:.4f} train {100 * run_correct / run_n:5.2f} "
                      f"val {100 * val_acc:5.2f} (best {100 * best['val_acc']:5.2f} @ {best['epoch']}) "
                      f"{time.time() - t_ep:5.1f}s", flush=True)

        if phase == "head":
            # "ImageNet features + linear classifier": test accuracy of the last head-phase
            # weights (no selection involved, so reporting it does not touch the protocol).
            t_logits = predict_logits(model, gd, split["test"], cfg, mean, std)
            head_only = {"epochs": n_epochs, "val_acc": history[-1]["val_acc"],
                         "test": summarize(gd.y_np[split["test"]], t_logits.argmax(1).numpy())}
            if verbose:
                print(f"[{cfg.name}] last-layer-only: test {100 * head_only['test']['accuracy']:.2f}",
                      flush=True)

    train_seconds = time.time() - t_start
    model.load_state_dict(best["state"])
    val_logits = predict_logits(model, gd, split["val"], cfg, mean, std)
    test_logits = predict_logits(model, gd, split["test"], cfg, mean, std)
    y_test = gd.y_np[split["test"]]
    result = {
        "name": cfg.name, "kind": "cnn", "config": asdict(cfg),
        "split": C.split_name(cfg.train_frac),
        "n_train": int(n_train), "n_val": int(len(split["val"])), "n_test": int(len(split["test"])),
        "epochs_run": epoch, "steps": total_steps, "best_epoch": best["epoch"],
        "val": summarize(gd.y_np[split["val"]], val_logits.argmax(1).numpy()),
        "test": summarize(y_test, test_logits.argmax(1).numpy()),
        "head_only": head_only,
        "n_params": count_parameters(model),
        "input_mean": [float(v) for v in mean.cpu()], "input_std": [float(v) for v in std.cpu()],
        "train_seconds": train_seconds,
        "history": history,
        "env": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                "cudnn": torch.backends.cudnn.version()},
    }
    np.savez_compressed(C.PREDS_DIR / f"{cfg.name}.npz", idx=split["test"], y_true=y_test,
                        y_pred=test_logits.argmax(1).numpy(),
                        prob=torch.softmax(test_logits, 1).numpy().astype(np.float16))
    if save_model:
        ckpt_dir = C.RESULTS_DIR / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best["state"], "config": asdict(cfg),
                    "mean": mean.cpu(), "std": std.cpu()}, ckpt_dir / f"{cfg.name}.pt")
    out_json.write_text(json.dumps(result), encoding="utf-8")
    if verbose:
        print(f"[{cfg.name}] DONE  val {100 * result['val']['accuracy']:.2f}  "
              f"TEST {100 * result['test']['accuracy']:.2f}  (best epoch {best['epoch']}/{epoch}, "
              f"{train_seconds / 60:.1f} min)", flush=True)
    del model, best
    torch.cuda.empty_cache()
    return result
