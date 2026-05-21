# clod_framework/data/uda_builder.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from clod_framework.data.yolo_detection_dataset import (
    YOLODetectionDataset,
    build_ultralytics_hyp,
    yolo_detection_collate,
)
from clod_framework.utils.yaml_utils import load_yaml


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / path

    if candidate.exists():
        return candidate

    return path


def load_dataset_cfg(dataset_entry: Any) -> dict[str, Any]:
    """
    Supports:
        dataset:
          source: configs/datasets/cityscapes.yaml
          target: configs/datasets/foggy_cityscapes_beta02.yaml

    or inline:
        dataset:
          source:
            root: ...
            train_images: ...
    """

    if isinstance(dataset_entry, dict):
        return dataset_entry

    if isinstance(dataset_entry, str):
        return load_yaml(_resolve_project_path(dataset_entry))

    raise TypeError(f"Unsupported dataset config entry: {type(dataset_entry)!r}")


def normalize_dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)

    if "num_classes" not in out:
        if "nc" in out:
            out["num_classes"] = int(out["nc"])
        else:
            raise KeyError("Dataset config must contain num_classes or nc.")

    if "train" not in out:
        if "train_images" in out:
            out["train"] = out["train_images"]
        elif "images" in out:
            out["train"] = out["images"]
        else:
            out["train"] = "images/train"

    if "val" not in out:
        if "val_images" in out:
            out["val"] = out["val_images"]
        else:
            out["val"] = "images/val"

    return out


def get_uda_dataset_configs(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_cfg = cfg.get("dataset", {})

    if "source" not in dataset_cfg or "target" not in dataset_cfg:
        raise KeyError(
            "UDA config requires dataset.source and dataset.target. "
            "Each can be a YAML path or an inline dict."
        )

    source_cfg = normalize_dataset_cfg(load_dataset_cfg(dataset_cfg["source"]))
    target_cfg = normalize_dataset_cfg(load_dataset_cfg(dataset_cfg["target"]))

    return source_cfg, target_cfg


def build_single_yolo_dataset(
    dataset_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    aug_cfg: dict[str, Any],
    split: str,
    augment: bool,
    include_empty: bool,
) -> YOLODetectionDataset:
    root = dataset_cfg["root"]

    if split == "train":
        split_path = dataset_cfg.get("train", dataset_cfg.get("train_images", "images/train"))
    elif split in {"val", "valid", "validation"}:
        split_path = dataset_cfg.get("val", dataset_cfg.get("val_images", "images/val"))
    else:
        split_path = dataset_cfg.get(split, split)

    image_size = int(training_cfg.get("img_size", training_cfg.get("image_size", 640)))
    batch_size = int(training_cfg.get("batch_size", 8))
    if split != "train":
        batch_size = int(training_cfg.get("eval_batch_size", batch_size))

    hyp = build_ultralytics_hyp(aug_cfg)

    return YOLODetectionDataset(
        root=root,
        split=split_path,
        image_size=image_size,
        num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc"))),
        names=dataset_cfg.get("names", None),
        class_filter=None,
        include_empty=include_empty,
        augment=augment,
        hyp=hyp,
        stride=int(training_cfg.get("stride", 32)),
        rect=bool(training_cfg.get("rect", False)),
        batch_size=batch_size,
        cache=dataset_cfg.get("cache", False),
        single_cls=bool(dataset_cfg.get("single_cls", False)),
        fraction=float(dataset_cfg.get("fraction", 1.0)),
        pad=float(training_cfg.get("pad", 0.0)),
        prefix=f"{dataset_cfg.get('name', split)} {split}: ",
    )


def build_uda_train_loaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, dict[str, Any], dict[str, Any]]:
    source_cfg, target_cfg = get_uda_dataset_configs(cfg)

    training_cfg = cfg.get("training", cfg.get("trainer", {}))
    aug_cfg = cfg.get("augmentation", {})

    source_dataset = build_single_yolo_dataset(
        dataset_cfg=source_cfg,
        training_cfg=training_cfg,
        aug_cfg=aug_cfg,
        split="train",
        augment=bool(training_cfg.get("augment", True)),
        include_empty=False,
    )

    target_dataset = build_single_yolo_dataset(
        dataset_cfg=target_cfg,
        training_cfg=training_cfg,
        aug_cfg=aug_cfg,
        split="train",
        augment=bool(training_cfg.get("augment", True)),
        include_empty=True,
    )

    workers = int(training_cfg.get("workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))
    batch_size = int(training_cfg.get("batch_size", 8))

    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=yolo_detection_collate,
        drop_last=True,
        persistent_workers=workers > 0,
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=yolo_detection_collate,
        drop_last=True,
        persistent_workers=workers > 0,
    )

    return source_loader, target_loader, source_cfg, target_cfg


def build_uda_val_loader(
    cfg: dict[str, Any],
) -> tuple[DataLoader, dict[str, Any]]:
    _, target_cfg = get_uda_dataset_configs(cfg)

    training_cfg = cfg.get("training", cfg.get("trainer", {}))
    aug_cfg = cfg.get("augmentation", {})

    target_val_dataset = build_single_yolo_dataset(
        dataset_cfg=target_cfg,
        training_cfg=training_cfg,
        aug_cfg=aug_cfg,
        split="val",
        augment=False,
        include_empty=True,
    )

    workers = int(training_cfg.get("workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))

    val_loader = DataLoader(
        target_val_dataset,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=yolo_detection_collate,
        drop_last=False,
        persistent_workers=workers > 0,
    )

    return val_loader, target_cfg