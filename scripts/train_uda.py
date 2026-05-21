# scripts/train_uda.py

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clod_framework.data.uda_builder import (
    build_uda_train_loaders,
    build_uda_val_loader,
    get_uda_dataset_configs,
)
from clod_framework.data.yolo_detection_dataset import (
    YOLODetectionDataset,
    build_ultralytics_hyp,
    yolo_detection_collate,
)
from clod_framework.engine.evaluator import DetectionEvaluator
from clod_framework.engine.optim import build_yolov8_optimizer
from clod_framework.losses.detection_loss import DetectionLoss
from clod_framework.methods.confmix_yolov8 import ConfMixYOLOv8Method
from clod_framework.models.builder import build_model
from clod_framework.utils.yaml_utils import load_yaml

# ---------------------------------------------------------------------
# Basic utils
# ---------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    return load_yaml(path)


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_torch_model(model: Any) -> nn.Module:
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model

    if isinstance(model, nn.Module):
        return model

    raise TypeError(f"Unsupported model type: {type(model)!r}")


def set_torch_train(model: Any, mode: bool = True) -> None:
    unwrap_torch_model(model).train(mode)


def save_checkpoint(
    model: Any,
    output_dir: str | Path,
    name: str,
    epoch: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / name

    payload = {
        "epoch": int(epoch),
        "model_state_dict": unwrap_torch_model(model).state_dict(),
        "metrics": metrics or {},
    }

    torch.save(payload, path)
    print(f"[Checkpoint] saved to {path}")
    return path


# ---------------------------------------------------------------------
# Resume utils
# ---------------------------------------------------------------------


def get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None

    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    if state.get("python", None) is not None:
        random.setstate(state["python"])

    if state.get("numpy", None) is not None:
        np.random.set_state(state["numpy"])

    if state.get("torch", None) is not None:
        torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def stage_state_path(
    output_dir: str | Path,
    stage: str,
) -> Path:
    output_dir = Path(output_dir)
    return output_dir / "checkpoints" / f"{stage}_training_state_latest.pt"


def save_training_state(
    *,
    stage: str,
    output_dir: str | Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    next_epoch: int,
    global_step: int,
    best_map: float,
    cfg: dict[str, Any],
    source_checkpoint: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = stage_state_path(output_dir, stage)
    tmp_path = path.with_suffix(".tmp")

    payload = {
        "type": "training_state",
        "stage": str(stage),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "best_map": float(best_map),
        "model_state_dict": unwrap_torch_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": get_rng_state(),
        "config": cfg,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
    }

    torch.save(payload, tmp_path)
    tmp_path.replace(path)

    print(
        f"[Resume] saved {stage} training state: "
        f"epoch={epoch}, next_epoch={next_epoch}, "
        f"global_step={global_step}, best_map={best_map:.6f}, path={path}"
    )

    return path


def load_training_state(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Training state not found: {path}")

    state = torch.load(path, map_location="cpu")

    print(
        f"[Resume] loaded training state from {path}: "
        f"stage={state.get('stage')}, "
        f"next_epoch={state.get('next_epoch')}, "
        f"global_step={state.get('global_step')}, "
        f"best_map={state.get('best_map')}"
    )

    return state


def restore_training_state(
    *,
    state: dict[str, Any],
    expected_stage: str,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    stage = str(state.get("stage", ""))

    if stage != expected_stage:
        raise ValueError(
            f"Resume state stage mismatch: expected {expected_stage}, got {stage}"
        )

    unwrap_torch_model(model).load_state_dict(
        state["model_state_dict"],
        strict=False,
    )

    optimizer.load_state_dict(state["optimizer_state_dict"])
    optimizer_to_device(optimizer, device)

    if "scaler_state_dict" in state:
        try:
            scaler.load_state_dict(state["scaler_state_dict"])
        except Exception as exc:
            print(f"[Resume] warning: failed to load AMP scaler state: {exc}")

    set_rng_state(state.get("rng_state", None))

    start_epoch = int(state.get("next_epoch", 0))
    global_step = int(state.get("global_step", 0))
    best_map = float(state.get("best_map", -1.0))

    print(
        f"[Resume] restored {expected_stage}: "
        f"start_epoch={start_epoch}, "
        f"global_step={global_step}, "
        f"best_map={best_map:.6f}"
    )

    return start_epoch, global_step, best_map


# ---------------------------------------------------------------------
# Config normalization
# ---------------------------------------------------------------------


def normalize_model_variant(model_cfg: dict[str, Any]) -> None:
    if "name" not in model_cfg:
        model_cfg["name"] = "yolov8"

    if "variant" not in model_cfg:
        model_cfg["variant"] = model_cfg.get("model_size", "yolov8n")

    if str(model_cfg["variant"]) in {"n", "s", "m", "l", "x"}:
        model_cfg["variant"] = f"yolov8{model_cfg['variant']}"


def default_detector_pretrained(model_cfg: dict[str, Any]) -> str:
    variant = str(model_cfg.get("variant", "yolov8n"))

    if variant.endswith(".pt"):
        return variant

    if variant.endswith((".yaml", ".yml")):
        return variant.replace(".yaml", ".pt").replace(".yml", ".pt")

    return f"{variant}.pt"


def normalize_uda_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)

    source_cfg, _ = get_uda_dataset_configs(cfg)

    model_cfg = cfg.setdefault("model", {})
    training_cfg = cfg.setdefault("training", cfg.get("trainer", {}))
    method_cfg = cfg.setdefault("method", {})
    eval_cfg = cfg.setdefault("evaluation", {})

    normalize_model_variant(model_cfg)

    if "num_classes" not in model_cfg:
        model_cfg["num_classes"] = int(source_cfg.get("num_classes", source_cfg.get("nc")))

    if "pretrained" not in model_cfg:
        if "weights" in model_cfg:
            model_cfg["pretrained"] = model_cfg["weights"]
            model_cfg.setdefault("pretrained_type", "detector")
        else:
            model_cfg["pretrained"] = default_detector_pretrained(model_cfg)
            model_cfg.setdefault("pretrained_type", "detector")

    training_cfg.setdefault("epochs", training_cfg.get("epochs_per_task", 50))
    training_cfg.setdefault("batch_size", 4)
    training_cfg.setdefault("eval_batch_size", training_cfg["batch_size"])
    training_cfg.setdefault("workers", 4)
    training_cfg.setdefault("pin_memory", True)
    training_cfg.setdefault("img_size", 640)
    training_cfg.setdefault("stride", 32)
    training_cfg.setdefault("rect", False)
    training_cfg.setdefault("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    training_cfg.setdefault("optimizer", "SGD")
    training_cfg.setdefault("lr", 0.01)
    training_cfg.setdefault("momentum", 0.937)
    training_cfg.setdefault("weight_decay", 0.0005)
    training_cfg.setdefault("nbs", 64)
    training_cfg.setdefault("accumulate", 1)

    training_cfg.setdefault("warmup_epochs", 3.0)
    training_cfg.setdefault("warmup_momentum", 0.8)
    training_cfg.setdefault("warmup_bias_lr", 0.1)
    training_cfg.setdefault("min_lr_ratio", 0.01)

    training_cfg.setdefault("grad_clip", 10.0)
    training_cfg.setdefault("amp", True)
    training_cfg.setdefault("augment", True)

    method_cfg.setdefault("name", "confmix")
    method_cfg.setdefault("lambda_mix", 1.0)
    method_cfg.setdefault("pseudo_conf_thres", 0.25)
    method_cfg.setdefault("pseudo_iou_thres", 0.5)
    method_cfg.setdefault("max_det", 300)
    method_cfg.setdefault("gamma_max", 1.0)
    method_cfg.setdefault("use_source_gt_for_mix", True)
    method_cfg.setdefault("uncertainty_power", 1.0)
    method_cfg.setdefault("region_score_key", "combined_conf")

    eval_cfg.setdefault("eval_before_uda", True)
    eval_cfg.setdefault("conf_thres", 0.001)
    eval_cfg.setdefault("iou_thres", 0.7)
    eval_cfg.setdefault("max_det", 300)
    eval_cfg.setdefault("save_csv", True)
    eval_cfg.setdefault("print_results", True)

    cfg["training"] = training_cfg
    cfg["method"] = method_cfg
    cfg["evaluation"] = eval_cfg

    return cfg


def make_source_stage_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = normalize_uda_config(cfg)
    source_cfg, _ = get_uda_dataset_configs(cfg)

    source_stage_cfg = copy.deepcopy(cfg)
    source_pretrain_cfg = source_stage_cfg.get("source_pretrain", {})

    experiment_cfg = source_stage_cfg.setdefault("experiment", {})
    model_cfg = source_stage_cfg.setdefault("model", {})
    training_cfg = source_stage_cfg.setdefault("training", {})

    normalize_model_variant(model_cfg)

    source_stage_cfg["dataset"] = source_cfg
    model_cfg["num_classes"] = int(source_cfg.get("num_classes", source_cfg.get("nc")))

    # Stage 1 should start from base detector pretrained weights, not the source checkpoint.
    if "pretrained" in source_pretrain_cfg:
        model_cfg["pretrained"] = source_pretrain_cfg["pretrained"]
        model_cfg["pretrained_type"] = source_pretrain_cfg.get("pretrained_type", "detector")
    else:
        model_cfg["pretrained"] = model_cfg.get(
            "base_pretrained",
            default_detector_pretrained(model_cfg),
        )
        model_cfg["pretrained_type"] = model_cfg.get("base_pretrained_type", "detector")

    if "epochs" in source_pretrain_cfg:
        training_cfg["epochs"] = int(source_pretrain_cfg["epochs"])

    if "batch_size" in source_pretrain_cfg:
        training_cfg["batch_size"] = int(source_pretrain_cfg["batch_size"])

    if "eval_batch_size" in source_pretrain_cfg:
        training_cfg["eval_batch_size"] = int(source_pretrain_cfg["eval_batch_size"])

    base_name = experiment_cfg.get("name", "uda_experiment")
    default_output_dir = f"outputs/{base_name}_source_pretrain"

    experiment_cfg["name"] = source_pretrain_cfg.get(
        "name",
        f"{base_name}_source_pretrain",
    )
    experiment_cfg["output_dir"] = source_pretrain_cfg.get(
        "output_dir",
        default_output_dir,
    )

    source_stage_cfg["model"] = model_cfg
    source_stage_cfg["training"] = training_cfg
    source_stage_cfg["experiment"] = experiment_cfg

    return source_stage_cfg


def make_uda_stage_cfg(
    cfg: dict[str, Any],
    source_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = normalize_uda_config(cfg)

    source_cfg, _ = get_uda_dataset_configs(cfg)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["num_classes"] = int(source_cfg.get("num_classes", source_cfg.get("nc")))

    if source_checkpoint is not None:
        model_cfg["pretrained"] = str(source_checkpoint)
        model_cfg["pretrained_type"] = "detector"

    cfg["model"] = model_cfg

    return cfg


# ---------------------------------------------------------------------
# Optimizer schedule
# ---------------------------------------------------------------------


def adjust_optimizer(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr0: float,
    lrf: float,
    momentum: float,
    warmup_momentum: float,
    warmup_bias_lr: float,
) -> float:
    if step < warmup_steps:
        warmup_factor = float(step + 1) / float(max(1, warmup_steps))
        lr_now = lr0 * warmup_factor
        momentum_now = warmup_momentum + warmup_factor * (momentum - warmup_momentum)

        for group in optimizer.param_groups:
            group_name = str(group.get("group_name", ""))

            if group_name == "bias_no_decay":
                group["lr"] = warmup_bias_lr + warmup_factor * (lr0 - warmup_bias_lr)
            else:
                group["lr"] = lr_now

            if "momentum" in group:
                group["momentum"] = momentum_now

            if "betas" in group:
                beta2 = group["betas"][1]
                group["betas"] = (momentum_now, beta2)

        return lr_now

    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr_now = lr0 * (lrf + (1.0 - lrf) * cosine)

    for group in optimizer.param_groups:
        group["lr"] = lr_now

        if "momentum" in group:
            group["momentum"] = momentum

        if "betas" in group:
            beta2 = group["betas"][1]
            group["betas"] = (momentum, beta2)

    return lr_now


# ---------------------------------------------------------------------
# Stage 1: source supervised pretrain
# ---------------------------------------------------------------------


def build_source_dataset(
    cfg: dict[str, Any],
    split: str,
    augment: bool,
    include_empty: bool,
) -> YOLODetectionDataset:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})

    if split == "train":
        split_path = dataset_cfg.get("train", "images/train")
    elif split in {"val", "valid", "validation"}:
        split_path = dataset_cfg.get("val", "images/val")
    else:
        split_path = dataset_cfg.get(split, split)

    image_size = int(training_cfg.get("img_size", training_cfg.get("image_size", 640)))
    batch_size = int(training_cfg.get("batch_size", 8))
    if split != "train":
        batch_size = int(training_cfg.get("eval_batch_size", batch_size))

    return YOLODetectionDataset(
        root=dataset_cfg["root"],
        split=split_path,
        image_size=image_size,
        num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc"))),
        names=dataset_cfg.get("names", None),
        class_filter=None,
        include_empty=include_empty,
        augment=augment,
        hyp=build_ultralytics_hyp(aug_cfg),
        stride=int(training_cfg.get("stride", 32)),
        rect=bool(training_cfg.get("rect", False)),
        batch_size=batch_size,
        cache=dataset_cfg.get("cache", False),
        single_cls=bool(dataset_cfg.get("single_cls", False)),
        fraction=float(dataset_cfg.get("fraction", 1.0)),
        pad=float(training_cfg.get("pad", 0.0)),
        prefix=f"source {split}: ",
    )


def build_source_loader(
    cfg: dict[str, Any],
    split: str,
    augment: bool,
    include_empty: bool,
    shuffle: bool,
) -> DataLoader:
    training_cfg = cfg.get("training", {})

    dataset = build_source_dataset(
        cfg=cfg,
        split=split,
        augment=augment,
        include_empty=include_empty,
    )

    batch_size = int(training_cfg.get("batch_size", 8))
    if split != "train":
        batch_size = int(training_cfg.get("eval_batch_size", batch_size))

    workers = int(training_cfg.get("workers", 0))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(training_cfg.get("pin_memory", False)),
        collate_fn=yolo_detection_collate,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def build_source_evaluator(cfg: dict[str, Any], output_dir: Path) -> DetectionEvaluator:
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    return DetectionEvaluator(
        num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc"))),
        metrics_dir=output_dir / "metrics",
        device=training_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        conf_thres=float(eval_cfg.get("conf_thres", 0.001)),
        nms_iou_thres=float(eval_cfg.get("iou_thres", 0.7)),
        max_det=int(eval_cfg.get("max_det", 300)),
        save_csv=bool(eval_cfg.get("save_csv", True)),
        print_results=bool(eval_cfg.get("print_results", True)),
        class_names=dataset_cfg.get("names", None),
    )


def train_source_stage(
    cfg: dict[str, Any],
    resume: bool = False,
    resume_path: str | Path | None = None,
) -> Path:
    cfg = make_source_stage_cfg(cfg)

    experiment_cfg = cfg.get("experiment", {})
    training_cfg = cfg.get("training", {})
    dataset_cfg = cfg["dataset"]

    output_dir = Path(experiment_cfg.get("output_dir", "outputs/source_pretrain"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    set_seed(int(experiment_cfg.get("seed", 0)))

    device = torch.device(training_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"))

    train_loader = build_source_loader(
        cfg=cfg,
        split="train",
        augment=bool(training_cfg.get("augment", True)),
        include_empty=False,
        shuffle=True,
    )

    val_loader = build_source_loader(
        cfg=cfg,
        split="val",
        augment=False,
        include_empty=True,
        shuffle=False,
    )

    model = build_model(cfg)
    model.to(device)

    loss_fn = DetectionLoss(
        model=model,
        device=device,
        box=float(training_cfg.get("box", 7.5)),
        cls=float(training_cfg.get("cls", 0.5)),
        dfl=float(training_cfg.get("dfl", 1.5)),
    )

    optimizer = build_yolov8_optimizer(
        model=model,
        optimizer_name=str(training_cfg.get("optimizer", "SGD")),
        lr=float(training_cfg.get("lr", 0.01)),
        momentum=float(training_cfg.get("momentum", 0.937)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0005)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        accumulate=int(training_cfg.get("accumulate", 1)),
        nbs=int(training_cfg.get("nbs", 64)),
    )

    evaluator = build_source_evaluator(cfg, output_dir)

    epochs = int(training_cfg.get("epochs", 50))
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(float(training_cfg.get("warmup_epochs", 3.0)) * len(train_loader))

    lr0 = float(training_cfg.get("lr", 0.01))
    lrf = float(training_cfg.get("min_lr_ratio", training_cfg.get("lrf", 0.01)))
    momentum = float(training_cfg.get("momentum", 0.937))
    warmup_momentum = float(training_cfg.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(training_cfg.get("warmup_bias_lr", 0.1))
    grad_clip = float(training_cfg.get("grad_clip", 10.0))

    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    num_classes = int(dataset_cfg.get("num_classes", dataset_cfg.get("nc")))
    seen_classes = list(range(num_classes))

    best_map = -1.0
    best_path = output_dir / "checkpoints" / "best.pt"
    global_step = 0
    start_epoch = 0

    if resume:
        source_state_path = (
            Path(resume_path)
            if resume_path is not None
            else stage_state_path(output_dir, "source")
        )

        if source_state_path.exists():
            state = load_training_state(source_state_path)
            start_epoch, global_step, best_map = restore_training_state(
                state=state,
                expected_stage="source",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )
        else:
            print(f"[Resume] source state not found: {source_state_path}, start from epoch 0")

    print(
        "[Stage 1: Source Pretrain] "
        f"dataset={dataset_cfg.get('name', dataset_cfg.get('root'))}, "
        f"epochs={epochs}, "
        f"start_epoch={start_epoch}, "
        f"output_dir={output_dir}"
    )

    if start_epoch >= epochs:
        print(f"[SourcePretrain] already finished: start_epoch={start_epoch}, epochs={epochs}")
        return best_path

    for epoch in range(start_epoch, epochs):
        set_torch_train(model, True)

        running_loss = 0.0
        running_box = 0.0
        running_cls = 0.0
        running_dfl = 0.0
        num_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"source pretrain epoch {epoch + 1}/{epochs}",
            ncols=140,
        )

        for batch in pbar:
            imgs = batch["img"].to(device, non_blocking=True).float()
            if imgs.max() > 2.0:
                imgs = imgs / 255.0

            batch = dict(batch)
            batch["img"] = imgs
            batch["images"] = imgs

            lr_now = adjust_optimizer(
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(batch)
                loss_out = loss_fn(
                    outputs=outputs,
                    targets=batch,
                    return_dict=True,
                )
                loss = loss_out.loss

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_torch_model(model).parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            items = getattr(loss_out, "loss_items", {})
            loss_value = float(loss.detach().cpu())

            running_loss += loss_value
            running_box += float(items.get("box_loss", torch.tensor(0.0)).detach().cpu())
            running_cls += float(items.get("cls_loss", torch.tensor(0.0)).detach().cpu())
            running_dfl += float(items.get("dfl_loss", torch.tensor(0.0)).detach().cpu())

            num_steps += 1
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{running_loss / num_steps:.4f}",
                    "box": f"{running_box / num_steps:.4f}",
                    "cls": f"{running_cls / num_steps:.4f}",
                    "dfl": f"{running_dfl / num_steps:.4f}",
                    "lr": f"{lr_now:.2e}",
                }
            )

        print(
            f"[SourcePretrain] epoch={epoch + 1}/{epochs} "
            f"loss={running_loss / max(1, num_steps):.6f}"
        )

        result = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            epoch=epoch + 1,
            task_id=0,
            seen_classes=seen_classes,
            csv_name="source_val.csv",
        )

        metrics = {
            "mAP50": float(result.mAP50),
            "mAP50-95": float(result.mAP5095),
            "precision": float(result.precision),
            "recall": float(result.recall),
        }

        save_checkpoint(
            model=model,
            output_dir=output_dir,
            name="last.pt",
            epoch=epoch + 1,
            metrics=metrics,
        )

        if result.mAP5095 > best_map:
            best_map = float(result.mAP5095)
            best_path = save_checkpoint(
                model=model,
                output_dir=output_dir,
                name="best.pt",
                epoch=epoch + 1,
                metrics=metrics,
            )

        save_training_state(
            stage="source",
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            next_epoch=epoch + 1,
            global_step=global_step,
            best_map=best_map,
            cfg=cfg,
        )

    print(f"[SourcePretrain] finished. best mAP50-95={best_map:.6f}")
    print(f"[SourcePretrain] best checkpoint: {best_path}")

    return best_path


# ---------------------------------------------------------------------
# Stage 2: ConfMix UDA
# ---------------------------------------------------------------------


def build_uda_evaluator(
    cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    output_dir: Path,
) -> DetectionEvaluator:
    training_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    source_cfg, _ = get_uda_dataset_configs(cfg)

    return DetectionEvaluator(
        num_classes=int(source_cfg.get("num_classes", source_cfg.get("nc"))),
        metrics_dir=output_dir / "metrics",
        device=training_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        conf_thres=float(eval_cfg.get("conf_thres", 0.001)),
        nms_iou_thres=float(eval_cfg.get("iou_thres", 0.7)),
        max_det=int(eval_cfg.get("max_det", 300)),
        save_csv=bool(eval_cfg.get("save_csv", True)),
        print_results=bool(eval_cfg.get("print_results", True)),
        class_names=source_cfg.get("names", target_cfg.get("names", None)),
    )


def train_uda_stage(
    cfg: dict[str, Any],
    source_checkpoint: str | Path | None = None,
    resume: bool = False,
    resume_path: str | Path | None = None,
    skip_pre_uda_eval_on_resume: bool = True,
) -> Path:
    cfg = make_uda_stage_cfg(cfg, source_checkpoint=source_checkpoint)

    experiment_cfg = cfg.get("experiment", {})
    training_cfg = cfg.get("training", {})
    method_cfg = cfg.get("method", {})
    evaluation_cfg = cfg.get("evaluation", {})

    output_dir = Path(
        experiment_cfg.get(
            "output_dir",
            cfg.get("paths", {}).get("outputs", "outputs/confmix_yolov8"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    set_seed(int(experiment_cfg.get("seed", 0)))

    source_loader, target_loader, source_cfg, target_cfg = build_uda_train_loaders(cfg)
    val_loader, target_val_cfg = build_uda_val_loader(cfg)

    model = build_model(cfg)
    device = torch.device(
        training_cfg.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )
    model.to(device)

    method = ConfMixYOLOv8Method(
        model=model,
        device=device,
        lambda_mix=float(method_cfg.get("lambda_mix", 1.0)),
        pseudo_conf_thres=float(method_cfg.get("pseudo_conf_thres", 0.25)),
        pseudo_iou_thres=float(method_cfg.get("pseudo_iou_thres", 0.5)),
        max_det=int(method_cfg.get("max_det", 300)),
        gamma_max=float(method_cfg.get("gamma_max", 1.0)),
        use_source_gt_for_mix=bool(method_cfg.get("use_source_gt_for_mix", True)),
        uncertainty_power=float(method_cfg.get("uncertainty_power", 1.0)),
        region_score_key=str(method_cfg.get("region_score_key", "combined_conf")),
    )

    optimizer = build_yolov8_optimizer(
        model=model,
        optimizer_name=str(training_cfg.get("optimizer", "SGD")),
        lr=float(training_cfg.get("lr", 0.01)),
        momentum=float(training_cfg.get("momentum", 0.937)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0005)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        accumulate=int(training_cfg.get("accumulate", 1)),
        nbs=int(training_cfg.get("nbs", 64)),
    )

    evaluator = build_uda_evaluator(
        cfg=cfg,
        target_cfg=target_val_cfg,
        output_dir=output_dir,
    )

    source_num_classes = int(source_cfg.get("num_classes", source_cfg.get("nc")))
    seen_classes = list(range(source_num_classes))

    epochs = int(training_cfg.get("epochs", 50))
    steps_per_epoch = min(len(source_loader), len(target_loader))

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "Invalid UDA dataloaders: steps_per_epoch <= 0. "
            f"len(source_loader)={len(source_loader)}, len(target_loader)={len(target_loader)}"
        )

    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(float(training_cfg.get("warmup_epochs", 3.0)) * steps_per_epoch)

    lr0 = float(training_cfg.get("lr", 0.01))
    lrf = float(training_cfg.get("min_lr_ratio", training_cfg.get("lrf", 0.01)))
    momentum = float(training_cfg.get("momentum", 0.937))
    warmup_momentum = float(training_cfg.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(training_cfg.get("warmup_bias_lr", 0.1))
    grad_clip = float(training_cfg.get("grad_clip", 10.0))

    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    global_step = 0
    best_map = -1.0
    best_path = output_dir / "checkpoints" / "best.pt"
    start_epoch = 0
    resume_state_loaded = False

    if resume:
        uda_state_path = (
            Path(resume_path)
            if resume_path is not None
            else stage_state_path(output_dir, "uda")
        )

        if uda_state_path.exists():
            state = load_training_state(uda_state_path)

            start_epoch, global_step, best_map = restore_training_state(
                state=state,
                expected_stage="uda",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
            )

            resume_state_loaded = True

            if state.get("source_checkpoint", None) is not None:
                source_checkpoint = state["source_checkpoint"]
        else:
            print(f"[Resume] UDA state not found: {uda_state_path}, start from epoch 0")

    eval_before_uda = bool(evaluation_cfg.get("eval_before_uda", True))

    if resume_state_loaded and skip_pre_uda_eval_on_resume:
        eval_before_uda = False
        print("[Resume] skip Pre-UDA Eval because UDA state was restored.")

    if eval_before_uda:
        print("")
        print("=" * 100)
        print("[Pre-UDA Eval] Evaluating Stage-1 source-pretrained model on target domain")
        print(f"[Pre-UDA Eval] source     = {source_cfg.get('name', source_cfg.get('root'))}")
        print(f"[Pre-UDA Eval] target     = {target_cfg.get('name', target_cfg.get('root'))}")
        print(f"[Pre-UDA Eval] checkpoint = {cfg.get('model', {}).get('pretrained')}")
        print("=" * 100)
        print("")

        pre_uda_result = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            epoch=-1,
            task_id=1,
            seen_classes=seen_classes,
            csv_name="uda_target_val.csv",
        )

        print(
            "[Pre-UDA Eval] "
            f"P={pre_uda_result.precision:.4f}, "
            f"R={pre_uda_result.recall:.4f}, "
            f"mAP50={pre_uda_result.mAP50:.4f}, "
            f"mAP50-95={pre_uda_result.mAP5095:.4f}"
        )
        print("")

    print(
        "[Stage 2: ConfMix UDA] "
        f"source={source_cfg.get('name', source_cfg.get('root'))}, "
        f"target={target_cfg.get('name', target_cfg.get('root'))}, "
        f"pretrained={cfg.get('model', {}).get('pretrained')}, "
        f"epochs={epochs}, "
        f"start_epoch={start_epoch}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"output_dir={output_dir}"
    )

    if start_epoch >= epochs:
        print(f"[UDA] already finished: start_epoch={start_epoch}, epochs={epochs}")
        return best_path

    for epoch in range(start_epoch, epochs):
        set_torch_train(model, True)

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"ConfMix YOLOv8 epoch {epoch + 1}/{epochs}",
            ncols=150,
        )

        running_total = 0.0
        running_source = 0.0
        running_mix = 0.0
        running_pseudo = 0.0
        running_uncertainty = 0.0
        running_certainty = 0.0
        running_combined = 0.0

        for step in pbar:
            source_batch = next(source_iter)
            target_batch = next(target_iter)

            lr_now = adjust_optimizer(
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                warmup_momentum=warmup_momentum,
                warmup_bias_lr=warmup_bias_lr,
            )

            progress = global_step / float(max(1, total_steps - 1))

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = method.training_step(
                    source_batch=source_batch,
                    target_batch=target_batch,
                    progress=progress,
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    unwrap_torch_model(model).parameters(),
                    grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            total_v = float(loss.detach().cpu())
            source_v = float(outputs["source_loss"].detach().cpu())
            mix_v = float(outputs["mix_loss"].detach().cpu())
            pseudo_v = float(outputs["num_pseudo"].detach().cpu())
            gamma_v = float(outputs["gamma"].detach().cpu())
            delta_v = float(outputs["delta"].detach().cpu())

            uncertainty_v = float(
                outputs.get("mean_uncertainty", torch.tensor(0.0, device=device))
                .detach()
                .cpu()
            )
            certainty_v = float(
                outputs.get("mean_certainty", torch.tensor(0.0, device=device))
                .detach()
                .cpu()
            )
            combined_v = float(
                outputs.get("mean_combined_conf", torch.tensor(0.0, device=device))
                .detach()
                .cpu()
            )

            running_total += total_v
            running_source += source_v
            running_mix += mix_v
            running_pseudo += pseudo_v
            running_uncertainty += uncertainty_v
            running_certainty += certainty_v
            running_combined += combined_v

            global_step += 1
            n = step + 1

            pbar.set_postfix(
                {
                    "total": f"{running_total / n:.4f}",
                    "src": f"{running_source / n:.4f}",
                    "mix": f"{running_mix / n:.4f}",
                    "pseudo": f"{running_pseudo / n:.1f}",
                    "gamma": f"{gamma_v:.3f}",
                    "delta": f"{delta_v:.3f}",
                    "unc": f"{running_uncertainty / n:.3f}",
                    "cert": f"{running_certainty / n:.3f}",
                    "comb": f"{running_combined / n:.3f}",
                    "lr": f"{lr_now:.2e}",
                }
            )

        print(
            f"[UDA] epoch={epoch + 1}/{epochs} "
            f"total={running_total / steps_per_epoch:.6f} "
            f"source={running_source / steps_per_epoch:.6f} "
            f"mix={running_mix / steps_per_epoch:.6f} "
            f"pseudo={running_pseudo / steps_per_epoch:.2f} "
            f"unc={running_uncertainty / steps_per_epoch:.4f} "
            f"cert={running_certainty / steps_per_epoch:.4f} "
            f"combined={running_combined / steps_per_epoch:.4f}"
        )

        result = evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            epoch=epoch + 1,
            task_id=1,
            seen_classes=seen_classes,
            csv_name="uda_target_val.csv",
        )

        metrics = {
            "mAP50": float(result.mAP50),
            "mAP50-95": float(result.mAP5095),
            "precision": float(result.precision),
            "recall": float(result.recall),
        }

        save_checkpoint(
            model=model,
            output_dir=output_dir,
            name="last.pt",
            epoch=epoch + 1,
            metrics=metrics,
        )

        if result.mAP5095 > best_map:
            best_map = float(result.mAP5095)
            best_path = save_checkpoint(
                model=model,
                output_dir=output_dir,
                name="best.pt",
                epoch=epoch + 1,
                metrics=metrics,
            )

        save_training_state(
            stage="uda",
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            next_epoch=epoch + 1,
            global_step=global_step,
            best_map=best_map,
            cfg=cfg,
            source_checkpoint=source_checkpoint,
        )

    print(f"[UDA] finished. best target mAP50-95={best_map:.6f}")
    print(f"[UDA] best checkpoint: {best_path}")

    return best_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def resolve_stage(cfg: dict[str, Any], cli_stage: str | None) -> str:
    if cli_stage is not None:
        return cli_stage

    run_cfg = cfg.get("run", {})
    if "stage" in run_cfg:
        return str(run_cfg["stage"])

    if "stage" in cfg:
        return str(cfg["stage"])

    return "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 ConfMix UDA with source pretrain")
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["source", "uda", "both"],
        help="source: source supervised pretrain; uda: ConfMix UDA; both: source then UDA",
    )

    parser.add_argument(
        "--source-ckpt",
        type=str,
        default=None,
        help="Checkpoint to initialize UDA stage. Overrides config model.pretrained for --stage uda.",
    )

    parser.add_argument(
        "--skip-source-if-exists",
        action="store_true",
        help="For --stage both, reuse source_pretrain best.pt if it already exists.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume source or UDA training from latest stage state.",
    )

    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Explicit path to source_training_state_latest.pt or uda_training_state_latest.pt.",
    )

    parser.add_argument(
        "--no-skip-pre-uda-eval-on-resume",
        action="store_true",
        help="When resuming UDA, run Pre-UDA Eval again instead of skipping it.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    stage = resolve_stage(raw_cfg, args.stage)

    print(f"[Config] {args.config}")
    print(f"[RunStage] {stage}")
    print(f"[Resume] {args.resume}")

    if stage == "source":
        best_path = train_source_stage(
            raw_cfg,
            resume=args.resume,
            resume_path=args.resume_path,
        )
        print(f"[Done] source pretrain best checkpoint: {best_path}")
        return

    if stage == "uda":
        best_path = train_uda_stage(
            raw_cfg,
            source_checkpoint=args.source_ckpt,
            resume=args.resume,
            resume_path=args.resume_path,
            skip_pre_uda_eval_on_resume=not args.no_skip_pre_uda_eval_on_resume,
        )
        print(f"[Done] UDA best checkpoint: {best_path}")
        return

    if stage == "both":
        source_cfg = make_source_stage_cfg(raw_cfg)
        source_output_dir = Path(source_cfg["experiment"]["output_dir"])
        source_best = source_output_dir / "checkpoints" / "best.pt"

        uda_cfg_for_paths = make_uda_stage_cfg(
            raw_cfg,
            source_checkpoint=source_best if source_best.exists() else args.source_ckpt,
        )
        uda_output_dir = Path(uda_cfg_for_paths["experiment"]["output_dir"])
        uda_state = stage_state_path(uda_output_dir, "uda")

        # Resume priority for --stage both:
        #   1. If UDA state exists, resume Stage 2 directly.
        #   2. Else source best exists and --skip-source-if-exists, skip Stage 1.
        #   3. Else resume/start Stage 1, then run Stage 2 from source best.
        if args.resume and uda_state.exists():
            print(f"[Resume] found UDA state, resume Stage 2 directly: {uda_state}")

            best_uda_path = train_uda_stage(
                raw_cfg,
                source_checkpoint=source_best if source_best.exists() else args.source_ckpt,
                resume=True,
                resume_path=uda_state,
                skip_pre_uda_eval_on_resume=not args.no_skip_pre_uda_eval_on_resume,
            )

            print(f"[Done] UDA best checkpoint: {best_uda_path}")
            return

        if args.skip_source_if_exists and source_best.exists():
            print(f"[Stage 1] skip source pretrain, found existing checkpoint: {source_best}")
            best_source_path = source_best
        else:
            best_source_path = train_source_stage(
                raw_cfg,
                resume=args.resume,
                resume_path=args.resume_path,
            )

        best_uda_path = train_uda_stage(
            raw_cfg,
            source_checkpoint=best_source_path,
            resume=False,
            resume_path=None,
            skip_pre_uda_eval_on_resume=not args.no_skip_pre_uda_eval_on_resume,
        )

        print(f"[Done] source best checkpoint: {best_source_path}")
        print(f"[Done] UDA best checkpoint: {best_uda_path}")
        return

    raise ValueError(f"Unsupported stage: {stage}")


if __name__ == "__main__":
    main()