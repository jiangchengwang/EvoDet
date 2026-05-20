# clod_framework/replay/ocdm_memory.py

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clod_framework.data.yolo_detection_dataset import yolo_detection_collate

def xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()

    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros(
            (box1.shape[0], box2.shape[0]),
            device=box1.device,
            dtype=box1.dtype,
        )

    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2 - inter

    return inter / union.clamp(min=1e-7)


def nms_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thres: float = 0.7,
    max_det: int = 300,
) -> list[int]:
    if boxes.numel() == 0:
        return []

    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i.item()))

        if order.numel() == 1:
            break

        ious = box_iou_xyxy(
            boxes[i].view(1, 4),
            boxes[order[1:]],
        ).view(-1)

        order = order[1:][ious <= iou_thres]

    return keep

@dataclass
class ReplaySample:
    image_path: str
    label_path: Optional[str] = None
    labels: list[list[float]] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    task_id: int = -1
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplaySample":
        return cls(
            image_path=str(data["image_path"]),
            label_path=data.get("label_path"),
            labels=[list(map(float, x)) for x in data.get("labels", [])],
            class_ids=[int(x) for x in data.get("class_ids", [])],
            task_id=int(data.get("task_id", -1)),
            score=data.get("score"),
        )


def count_labels(
    data: list[np.ndarray],
    nc: int,
) -> np.ndarray:
    counters = np.zeros(nc, dtype=np.int32)
    for sample in data:
        if len(sample) == 0:
            continue
        counters[np.unique(sample.astype(int))] += 1
    return counters


def count_labelsv2(
    data: list[np.ndarray],
    nc: int,
    ths: Optional[int] = None,
) -> np.ndarray:
    counters = np.zeros(nc, dtype=np.int32)
    for sample in data:
        if len(sample) == 0:
            continue
        values, counts = np.unique(sample.astype(int), return_counts=True)
        if ths is not None:
            counts[counts > ths] = ths
        counters[values] += counts
    return counters


def get_labels_distribution(
    data: list[np.ndarray],
    nc: int,
    rho: float = 1.0,
    ths: Optional[int] = None,
) -> np.ndarray:
    abs_freq = count_labels(data, nc) if ths is None else count_labelsv2(data, nc, ths)
    pow_abs_freq = abs_freq.astype(np.float32) ** rho
    denom = np.sum(pow_abs_freq)
    if denom <= 0:
        return np.ones(nc, dtype=np.float32) / nc
    return pow_abs_freq / denom


def cross_entropy_torch(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-12
    return torch.sum(-p * torch.log(q.clamp(min=eps)), dim=1)


def efficient_memory_update_indices(
    data: list[np.ndarray],
    nc: int,
    num_iter: int,
    target_distr: Optional[torch.Tensor] = None,
) -> list[int]:
    if num_iter <= 0:
        return []

    if target_distr is None:
        target_distr = torch.ones(len(data), nc).float() / nc

    matrix = []
    for sample in data:
        labels_sample = np.zeros(nc, dtype=np.float32)
        if len(sample) > 0:
            labels_sample[np.unique(sample.astype(int))] += 1.0
        matrix.append(labels_sample)

    matrix_labels_samples = torch.from_numpy(np.asarray(matrix)).float()
    abs_freq = torch.sum(matrix_labels_samples, dim=0)

    idxs_to_remove: list[int] = []

    for _ in range(num_iter):
        n = matrix_labels_samples.shape[0]
        abs_freq_matrix = abs_freq.repeat(n, 1)
        diff_matrix = abs_freq_matrix - matrix_labels_samples
        normalize = torch.sum(diff_matrix, dim=1).reshape(-1, 1).clamp(min=1.0)
        q = diff_matrix / normalize

        scores = cross_entropy_torch(target_distr, q)
        if idxs_to_remove:
            scores[idxs_to_remove] = float("inf")

        index = torch.argmin(scores).item()
        abs_freq = diff_matrix[index].clone()
        idxs_to_remove.append(index)

    return idxs_to_remove


class OCDMMemory:
    """
    Original-style OCDM:
        - pool = old memory + current task candidates
        - if pool > capacity, remove samples that make distribution closest to uniform target
        - stats saved as tab-separated headerless ocdm.csv, matching original format
        - LwF mode refreshes old memory and current dataset labels with current model
    """

    def __init__(
        self,
        memory_size: int,
        num_classes: int,
        max_num_classes: Optional[int] = None,
        seed: int = 0,
        stats_path: str | Path | None = None,
        ths: Optional[int] = None,
        count_dup: bool = True,
    ) -> None:
        self.memory_size = int(memory_size)
        self.num_classes = int(num_classes)
        self.max_num_classes = int(max_num_classes or num_classes)
        self.seed = int(seed)
        self.stats_path = Path(stats_path) if stats_path is not None else None
        self.ths = ths
        self.count_dup_enabled = bool(count_dup)

        self.samples: list[ReplaySample] = []
        self.ntasks = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def as_samples(self) -> list[ReplaySample]:
        return list(self.samples)

    def update_from_dataset(
            self,
            dataset: Any,
            task_id: int,
            seen_classes: Sequence[int],
            model: Any | None = None,
            device: str | torch.device = "cuda",
            batch_size: int = 8,
            workers: int = 0,
            pseudo_conf_thres: float = 0.25,
            pseudo_iou_thres: float = 0.7,
            stats_path: str | Path | None = None,
            batch_size_ocdm: int = -1,
    ) -> dict[str, Any]:
        """
        Original-style OCDM batch-wise update.

        Difference from whole-pool update:
            - Refresh old memory labels first.
            - Generate current task candidates.
            - Insert current candidates batch by batch.
            - If memory exceeds capacity after each batch, remove samples that make
              the memory class distribution closest to the target distribution.

        Args:
            batch_size_ocdm:
                -1 or <=0 means use all current candidates in one batch.
                positive value means update memory incrementally in chunks.
        """

        self.ntasks += 1

        seen_classes = sorted(set(int(c) for c in seen_classes))

        # 1. Refresh old memory labels with current model.
        refreshed_memory: list[ReplaySample] = []

        if model is not None and self.samples:
            from clod_framework.data.replay_yolo_dataset import ReplayYOLODataset

            replay_dataset = ReplayYOLODataset(
                samples=self.samples,
                image_size=getattr(dataset, "image_size", getattr(dataset, "imgsz", 640)),
            )

            refreshed_memory = collect_pseudo_candidates_from_dataset(
                dataset=replay_dataset,
                model=model,
                task_id=None,
                seen_classes=seen_classes,
                device=device,
                batch_size=batch_size,
                workers=workers,
                conf_thres=pseudo_conf_thres,
                iou_thres=pseudo_iou_thres,
                fallback_task_ids=[s.task_id for s in self.samples],
            )

            # If pseudo refresh fails, keep old memory.
            if not refreshed_memory:
                refreshed_memory = list(self.samples)
        else:
            refreshed_memory = list(self.samples)

        # 2. Generate candidates from current task dataset.
        if model is not None:
            old_aug = getattr(dataset, "augment", None)
            if old_aug is not None:
                dataset.augment = False

            current_candidates = collect_pseudo_candidates_from_dataset(
                dataset=dataset,
                model=model,
                task_id=task_id,
                seen_classes=seen_classes,
                device=device,
                batch_size=batch_size,
                workers=workers,
                conf_thres=pseudo_conf_thres,
                iou_thres=pseudo_iou_thres,
            )

            if old_aug is not None:
                dataset.augment = old_aug

            if not current_candidates:
                current_candidates = collect_gt_candidates_from_yolo_dataset(
                    dataset=dataset,
                    task_id=task_id,
                    class_filter=seen_classes,
                )
        else:
            current_candidates = collect_gt_candidates_from_yolo_dataset(
                dataset=dataset,
                task_id=task_id,
                class_filter=seen_classes,
            )

        current_source_paths = {s.image_path for s in current_candidates}

        # 3. Batch-wise OCDM update.
        self.samples = self._batchwise_update_memory(
            initial_memory=refreshed_memory,
            candidates=current_candidates,
            batch_size_ocdm=batch_size_ocdm,
        )

        imgs_added = sum(1 for s in self.samples if s.image_path in current_source_paths)
        count_dup = self.count_duplicates() if self.count_dup_enabled else 0
        nc = self.num_classes

        row = self._stats_row(
            imgs_added=imgs_added,
            nc=nc,
            count_dup=count_dup,
        )

        path = Path(stats_path) if stats_path is not None else self.stats_path
        if path is not None:
            self.append_stats(path, row)

        return {
            "imgs_added": imgs_added,
            "nc": nc,
            "count_dup": count_dup,
            "memory_size": len(self.samples),
            "num_current_candidates": len(current_candidates),
            "num_refreshed_memory": len(refreshed_memory),
        }

    def _batchwise_update_memory(
            self,
            initial_memory: list[ReplaySample],
            candidates: list[ReplaySample],
            batch_size_ocdm: int = -1,
    ) -> list[ReplaySample]:
        """
        Batch-wise OCDM update.

        Original OCDM updates memory incrementally:
            memory + current_batch -> remove excess -> memory
            repeat until all candidates are processed.
        """

        memory = list(initial_memory)

        if batch_size_ocdm is None or int(batch_size_ocdm) <= 0:
            batch_size_ocdm = len(candidates) if candidates else 1

        batch_size_ocdm = max(1, int(batch_size_ocdm))

        for start in range(0, len(candidates), batch_size_ocdm):
            batch = candidates[start: start + batch_size_ocdm]
            pool = memory + batch

            if len(pool) <= self.memory_size:
                memory = pool
                continue

            remove_n = len(pool) - self.memory_size
            data = [np.asarray(s.class_ids, dtype=np.int64) for s in pool]

            remove_indices = set(
                efficient_memory_update_indices(
                    data=data,
                    nc=self.num_classes,
                    num_iter=remove_n,
                )
            )

            memory = [
                sample
                for idx, sample in enumerate(pool)
                if idx not in remove_indices
            ]

        return memory[: self.memory_size]

    def update_from_pool(
            self,
            pool: list[ReplaySample],
            imgs_added_source_paths: set[str],
            stats_path: str | Path | None = None,
            batch_size_ocdm: int = -1,
    ) -> dict[str, Any]:
        """
        Compatibility wrapper.

        If older code still calls update_from_pool(), this uses the same
        batch-wise distribution removal strategy.
        """

        self.samples = self._batchwise_update_memory(
            initial_memory=[],
            candidates=pool,
            batch_size_ocdm=batch_size_ocdm,
        )

        imgs_added = sum(1 for s in self.samples if s.image_path in imgs_added_source_paths)
        count_dup = self.count_duplicates() if self.count_dup_enabled else 0
        nc = self.num_classes

        row = self._stats_row(
            imgs_added=imgs_added,
            nc=nc,
            count_dup=count_dup,
        )

        path = Path(stats_path) if stats_path is not None else self.stats_path
        if path is not None:
            self.append_stats(path, row)

        return {
            "imgs_added": imgs_added,
            "nc": nc,
            "count_dup": count_dup,
            "memory_size": len(self.samples),
        }

    def class_arrays(self) -> list[np.ndarray]:
        return [np.asarray(s.class_ids, dtype=np.int64) for s in self.samples]

    def class_histogram(self) -> np.ndarray:
        return get_labels_distribution(
            self.class_arrays(),
            nc=self.num_classes,
            ths=self.ths,
        )

    def count_duplicates(self) -> int:
        counts = {}
        for sample in self.samples:
            counts[sample.image_path] = counts.get(sample.image_path, 0) + 1
        return sum(1 for _, v in counts.items() if v > 1)

    def _stats_row(
        self,
        imgs_added: int,
        nc: int,
        count_dup: int,
    ) -> list[float]:
        to_save = np.zeros(
            self.max_num_classes + 2 + (1 if self.count_dup_enabled else 0),
            dtype=np.float32,
        )

        distr = self.class_histogram()
        to_save[: self.num_classes] = distr
        to_save[-(2 + (1 if self.count_dup_enabled else 0))] = imgs_added
        to_save[-(1 + (1 if self.count_dup_enabled else 0))] = nc

        if self.count_dup_enabled:
            to_save[-1] = count_dup

        return to_save.tolist()

    def append_stats(
        self,
        path: str | Path,
        row: Sequence[float],
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if path.exists() else "w"

        with path.open(mode, newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(row)

    def state_dict(self) -> dict[str, Any]:
        return {
            "memory_size": self.memory_size,
            "num_classes": self.num_classes,
            "max_num_classes": self.max_num_classes,
            "seed": self.seed,
            "ths": self.ths,
            "count_dup": self.count_dup_enabled,
            "ntasks": self.ntasks,
            "samples": [s.to_dict() for s in self.samples],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.memory_size = int(state["memory_size"])
        self.num_classes = int(state["num_classes"])
        self.max_num_classes = int(state.get("max_num_classes", self.num_classes))
        self.seed = int(state.get("seed", 0))
        self.ths = state.get("ths")
        self.count_dup_enabled = bool(state.get("count_dup", True))
        self.ntasks = int(state.get("ntasks", 0))
        self.rng = random.Random(self.seed)
        self.samples = [ReplaySample.from_dict(x) for x in state.get("samples", [])]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OCDMMemory":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        mem = cls(
            memory_size=int(state["memory_size"]),
            num_classes=int(state["num_classes"]),
            max_num_classes=int(state.get("max_num_classes", state["num_classes"])),
            seed=int(state.get("seed", 0)),
            ths=state.get("ths"),
            count_dup=bool(state.get("count_dup", True)),
        )
        mem.load_state_dict(state)
        return mem


def collect_gt_candidates_from_yolo_dataset(
    dataset: Any,
    task_id: int,
    class_filter: Sequence[int],
) -> list[ReplaySample]:
    class_filter = set(int(c) for c in class_filter)
    candidates: list[ReplaySample] = []

    for sample in getattr(dataset, "samples", []):
        labels = dataset._load_labels(sample.label_path)
        if labels.numel() == 0:
            continue

        rows = []
        class_ids = set()

        for row in labels.tolist():
            cls = int(row[0])
            if cls not in class_filter:
                continue
            rows.append([float(x) for x in row])
            class_ids.add(cls)

        if rows:
            candidates.append(
                ReplaySample(
                    image_path=str(sample.image_path),
                    label_path=str(sample.label_path),
                    labels=rows,
                    class_ids=sorted(class_ids),
                    task_id=int(task_id),
                    score=1.0,
                )
            )

    return candidates


@torch.no_grad()
def collect_pseudo_candidates_from_dataset(
    dataset: Any,
    model: Any,
    task_id: Optional[int],
    seen_classes: Sequence[int],
    device: str | torch.device = "cuda",
    batch_size: int = 8,
    workers: int = 0,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    fallback_task_ids: Optional[Sequence[int]] = None,
) -> list[ReplaySample]:
    """
    Collect pseudo labels from a dataset for OCDM replay memory.

    Important:
        Pseudo labels must be generated on normal single images.
        Do NOT use Mosaic/MixUp/CutMix here, otherwise pseudo boxes will not
        correspond to the original image path saved into ReplaySample.

    This version:
        1. Temporarily disables dataset augmentation if supported.
        2. Rebuilds dataset buffer to avoid Ultralytics Mosaic empty-buffer errors.
        3. Restores dataset state after pseudo-label generation.
        4. Restores model training state after generation.
    """

    device = torch.device(device)
    seen_set = set(int(c) for c in seen_classes)

    # ------------------------------------------------------------
    # 1. Put dataset into pseudo-label mode.
    # ------------------------------------------------------------
    restore_dataset_mode = None

    if hasattr(dataset, "enable_pseudo_label_mode"):
        restore_dataset_mode = dataset.enable_pseudo_label_mode()
    else:
        saved_dataset_state = {
            "augment": getattr(dataset, "augment", None),
            "transforms": getattr(dataset, "transforms", None),
            "buffer": list(getattr(dataset, "buffer", [])) if hasattr(dataset, "buffer") else None,
            "hyp_values": {},
        }

        hyp = getattr(dataset, "hyp", None)
        if hyp is not None:
            for key in ["mosaic", "mixup", "cutmix", "copy_paste"]:
                if hasattr(hyp, key):
                    saved_dataset_state["hyp_values"][key] = getattr(hyp, key)
                    setattr(hyp, key, 0.0)

        if hasattr(dataset, "augment"):
            dataset.augment = False

        if hasattr(dataset, "labels") and hasattr(dataset, "buffer"):
            dataset.buffer = list(range(len(dataset.labels)))

        if hasattr(dataset, "build_transforms"):
            try:
                dataset.transforms = dataset.build_transforms(getattr(dataset, "hyp", None))
            except TypeError:
                try:
                    dataset.transforms = dataset.build_transforms()
                except Exception:
                    pass
            except Exception:
                pass

        def restore_dataset_mode() -> None:
            if saved_dataset_state["augment"] is not None and hasattr(dataset, "augment"):
                dataset.augment = saved_dataset_state["augment"]

            hyp_restore = getattr(dataset, "hyp", None)
            if hyp_restore is not None:
                for key, value in saved_dataset_state["hyp_values"].items():
                    setattr(hyp_restore, key, value)

            if saved_dataset_state["buffer"] is not None and hasattr(dataset, "buffer"):
                dataset.buffer = saved_dataset_state["buffer"]

            if saved_dataset_state["transforms"] is not None and hasattr(dataset, "transforms"):
                dataset.transforms = saved_dataset_state["transforms"]

    # ------------------------------------------------------------
    # 2. Put model into eval mode.
    # ------------------------------------------------------------
    torch_model = model.model if hasattr(model, "model") else model
    was_training = bool(getattr(torch_model, "training", False))

    if hasattr(model, "eval"):
        model.eval()
    elif hasattr(torch_model, "eval"):
        torch_model.eval()

    candidates: list[ReplaySample] = []
    global_index = 0

    try:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=False,
            collate_fn=yolo_detection_collate,
            drop_last=False,
            persistent_workers=workers > 0,
        )

        for batch in tqdm(loader, desc="pseudo labels", ncols=120):
            images = batch["img"].to(device, non_blocking=True).float()

            if images.numel() > 0 and images.max() > 2.0:
                images = images / 255.0

            preds = model.predict_raw(images)

            # Prefer the full model class count if available.
            # Do not use max(seen_classes)+1 when the model head is fixed to 80 classes,
            # because the prediction tensor still contains all 80 class channels.
            if hasattr(model, "num_classes"):
                num_classes = int(model.num_classes)
            elif hasattr(torch_model, "model") and hasattr(torch_model.model[-1], "nc"):
                num_classes = int(torch_model.model[-1].nc)
            elif hasattr(torch_model, "nc"):
                num_classes = int(torch_model.nc)
            else:
                num_classes = max(seen_set) + 1 if seen_set else 1

            pred_list = normalize_yolov8_predictions(
                preds=preds,
                num_classes=num_classes,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )

            paths = batch["paths"]
            label_paths = batch.get("label_paths", [None] * len(paths))
            letterboxes = batch["letterbox"]

            for i, pred in enumerate(pred_list):
                rows = []
                class_ids = set()
                scores = []

                if pred.numel() > 0:
                    boxes_ltrb = pred[:, :4].clone()
                    conf = pred[:, 4]
                    cls = pred[:, 5].long()

                    keep = torch.tensor(
                        [int(c.item()) in seen_set for c in cls],
                        dtype=torch.bool,
                        device=pred.device,
                    )

                    boxes_ltrb = boxes_ltrb[keep]
                    conf = conf[keep]
                    cls = cls[keep]

                    if boxes_ltrb.numel() > 0:
                        boxes_orig = letterbox_xyxy_to_original_xyxy(
                            boxes_ltrb,
                            letterboxes[i],
                        )
                        labels_xywh = original_xyxy_to_normalized_xywh(
                            boxes_orig,
                            letterboxes[i],
                        )

                        for c, box, score in zip(cls, labels_xywh, conf):
                            x, y, w, h = box.tolist()

                            if w <= 1e-6 or h <= 1e-6:
                                continue

                            rows.append(
                                [
                                    float(int(c.item())),
                                    float(x),
                                    float(y),
                                    float(w),
                                    float(h),
                                ]
                            )
                            class_ids.add(int(c.item()))
                            scores.append(float(score.item()))

                if rows:
                    if task_id is None and fallback_task_ids is not None:
                        if global_index < len(fallback_task_ids):
                            sample_task_id = int(fallback_task_ids[global_index])
                        else:
                            sample_task_id = -1
                    elif task_id is None:
                        sample_task_id = -1
                    else:
                        sample_task_id = int(task_id)

                    candidates.append(
                        ReplaySample(
                            image_path=str(paths[i]),
                            label_path=(
                                str(label_paths[i])
                                if label_paths is not None and i < len(label_paths)
                                else None
                            ),
                            labels=rows,
                            class_ids=sorted(class_ids),
                            task_id=sample_task_id,
                            score=sum(scores) / max(1, len(scores)),
                        )
                    )

                global_index += 1

    finally:
        # ------------------------------------------------------------
        # 3. Restore model state.
        # ------------------------------------------------------------
        if was_training:
            if hasattr(model, "train"):
                model.train()
            elif hasattr(torch_model, "train"):
                torch_model.train()

        # ------------------------------------------------------------
        # 4. Restore dataset state.
        # ------------------------------------------------------------
        if restore_dataset_mode is not None:
            restore_dataset_mode()

    return candidates

def normalize_yolov8_predictions(
    preds: Any,
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
) -> list[torch.Tensor]:
    if isinstance(preds, tuple):
        preds = preds[0]

    if isinstance(preds, list):
        if len(preds) == 2 and isinstance(preds[0], torch.Tensor):
            preds = preds[0]
        elif len(preds) > 0 and isinstance(preds[0], torch.Tensor):
            return preds

    if not isinstance(preds, torch.Tensor):
        return []

    if preds.ndim == 2:
        preds = preds.unsqueeze(0)

    if preds.ndim != 3:
        return []

    out = []

    for p in preds:
        if p.shape[0] in {4 + num_classes, 5 + num_classes, 6} and p.shape[0] < p.shape[1]:
            p = p.transpose(0, 1).contiguous()

        if p.shape[-1] == 6:
            det = p
        elif p.shape[-1] >= 4 + num_classes:
            boxes_xywh = p[:, :4]
            cls_scores = p[:, 4 : 4 + num_classes]
            conf, cls = cls_scores.max(dim=1)
            boxes_xyxy = xywh_to_xyxy(boxes_xywh)
            det = torch.cat([boxes_xyxy, conf[:, None], cls.float()[:, None]], dim=1)
        else:
            out.append(p.new_zeros((0, 6)))
            continue

        det = det[det[:, 4] >= conf_thres]

        if det.numel() == 0:
            out.append(det.new_zeros((0, 6)))
            continue

        final = []
        for c in det[:, 5].unique():
            idx = torch.where(det[:, 5] == c)[0]
            keep = nms_indices(det[idx, :4], det[idx, 4], iou_thres=iou_thres, max_det=300)
            if keep:
                final.append(det[idx[keep]])

        out.append(torch.cat(final, dim=0) if final else det.new_zeros((0, 6)))

    return out


def letterbox_xyxy_to_original_xyxy(
    boxes: torch.Tensor,
    letterbox: Mapping[str, float],
) -> torch.Tensor:
    scale = float(letterbox["scale"])
    pad_left = float(letterbox["pad_left"])
    pad_top = float(letterbox["pad_top"])
    orig_w = float(letterbox["orig_w"])
    orig_h = float(letterbox["orig_h"])

    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes[:, :4] /= scale
    boxes[:, [0, 2]].clamp_(0.0, orig_w)
    boxes[:, [1, 3]].clamp_(0.0, orig_h)
    return boxes


def original_xyxy_to_normalized_xywh(
    boxes: torch.Tensor,
    letterbox: Mapping[str, float],
) -> torch.Tensor:
    orig_w = float(letterbox["orig_w"])
    orig_h = float(letterbox["orig_h"])

    x1, y1, x2, y2 = boxes.unbind(dim=1)

    x = ((x1 + x2) / 2) / orig_w
    y = ((y1 + y2) / 2) / orig_h
    w = (x2 - x1) / orig_w
    h = (y2 - y1) / orig_h

    out = torch.stack([x, y, w, h], dim=1)
    out.clamp_(0.0, 1.0)
    return out