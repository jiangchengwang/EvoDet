# clod_framework/data/builder.py

from __future__ import annotations

from typing import Any, Optional, Sequence

from torch.utils.data import DataLoader

from clod_framework.data.yolo_detection_dataset import (
    YOLODetectionDataset,
    build_ultralytics_hyp,
    yolo_detection_collate,
)


def get_task_id(task: Any) -> int:
    if isinstance(task, dict):
        return int(task.get("task_id", task.get("id", 0)))
    return int(getattr(task, "task_id", getattr(task, "id", 0)))


def get_task_classes(task: Any) -> list[int]:
    if isinstance(task, dict):
        return list(map(int, task.get("classes", task.get("class_ids", []))))
    return list(map(int, getattr(task, "classes", getattr(task, "class_ids", []))))


def get_old_classes(task: Any) -> list[int]:
    if isinstance(task, dict):
        return list(map(int, task.get("old_classes", task.get("seen_classes_before", []))))
    return list(map(int, getattr(task, "old_classes", getattr(task, "seen_classes_before", []))))


def get_seen_classes(task: Any) -> list[int]:
    if isinstance(task, dict):
        if "seen_classes" in task:
            return sorted(set(map(int, task["seen_classes"])))
        return sorted(set(get_task_classes(task)) | set(get_old_classes(task)))

    if hasattr(task, "seen_classes"):
        return sorted(set(map(int, task.seen_classes)))

    return sorted(set(get_task_classes(task)) | set(get_old_classes(task)))


def build_yolo_dataset(
    cfg: dict[str, Any],
    task: Any,
    split: str,
    class_filter: Optional[Sequence[int]] = None,
    include_empty: bool = False,
    augment: bool = False,
) -> YOLODetectionDataset:
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})

    root = dataset_cfg["root"]

    if split == "train":
        split_path = dataset_cfg.get("train", "images/train")
    elif split in {"val", "valid", "validation"}:
        split_path = dataset_cfg.get("val", "images/val")
    elif split == "test":
        split_path = dataset_cfg.get("test", "images/test")
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
        num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80))),
        names=dataset_cfg.get("names", None),
        class_filter=class_filter,
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
        prefix=f"{split}: ",
    )


def build_train_loader(
    cfg: dict[str, Any],
    task: Any,
) -> DataLoader:
    training_cfg = cfg.get("training", {})
    current_classes = get_task_classes(task)

    print(f"[Data] task {get_task_id(task)} train classes: {current_classes}")

    dataset = build_yolo_dataset(
        cfg=cfg,
        task=task,
        split="train",
        class_filter=current_classes,
        include_empty=False,
        augment=bool(training_cfg.get("augment", True)),
    )

    if isinstance(task, dict):
        task["train_dataset"] = dataset
    else:
        setattr(task, "train_dataset", dataset)

    workers = int(training_cfg.get("workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=yolo_detection_collate,
        drop_last=False,
        persistent_workers=workers > 0,
    )


def build_val_loader(
    cfg: dict[str, Any],
    task: Any,
) -> DataLoader:
    training_cfg = cfg.get("training", {})
    seen_classes = get_seen_classes(task)

    print(f"[Data] task {get_task_id(task)} eval classes: {seen_classes}")

    dataset = build_yolo_dataset(
        cfg=cfg,
        task=task,
        split="val",
        class_filter=seen_classes,
        include_empty=True,
        augment=False,
    )

    if isinstance(task, dict):
        task["val_dataset"] = dataset
    else:
        setattr(task, "val_dataset", dataset)

    workers = int(training_cfg.get("workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=yolo_detection_collate,
        drop_last=False,
        persistent_workers=workers > 0,
    )