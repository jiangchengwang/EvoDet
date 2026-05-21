# clod_framework/data/yolo_detection_dataset.py

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence
import inspect
import torch
import numpy as np

try:
    from ultralytics.data.dataset import YOLODataset
except Exception as exc:
    raise ImportError(
        "OfficialUltralyticsYOLODataset requires ultralytics.data.dataset.YOLODataset. "
        "Please check your ultralytics installation."
    ) from exc


def build_ultralytics_hyp(
    aug_cfg: Optional[dict[str, Any]] = None,
) -> SimpleNamespace:
    aug_cfg = aug_cfg or {}

    return SimpleNamespace(
        degrees=float(aug_cfg.get("degrees", 0.0)),
        translate=float(aug_cfg.get("translate", 0.1)),
        scale=float(aug_cfg.get("scale", 0.5)),
        shear=float(aug_cfg.get("shear", 0.0)),
        perspective=float(aug_cfg.get("perspective", 0.0)),

        hsv_h=float(aug_cfg.get("hsv_h", 0.015)),
        hsv_s=float(aug_cfg.get("hsv_s", 0.7)),
        hsv_v=float(aug_cfg.get("hsv_v", 0.4)),

        flipud=float(aug_cfg.get("flipud", 0.0)),
        fliplr=float(aug_cfg.get("fliplr", 0.5)),

        mosaic=float(aug_cfg.get("mosaic", 1.0)),
        mixup=float(aug_cfg.get("mixup", 0.0)),
        cutmix=float(aug_cfg.get("cutmix", 0.0)),
        copy_paste=float(aug_cfg.get("copy_paste", 0.0)),
        copy_paste_mode=str(aug_cfg.get("copy_paste_mode", "flip")),

        mask_ratio=int(aug_cfg.get("mask_ratio", 4)),
        overlap_mask=bool(aug_cfg.get("overlap_mask", True)),
        bgr=float(aug_cfg.get("bgr", 0.0)),
    )


def normalize_class_names(
    names: Optional[Sequence[str] | dict[int, str]],
    num_classes: int,
) -> dict[int, str]:
    if names is None:
        return {i: f"class_{i}" for i in range(num_classes)}

    if isinstance(names, dict):
        out = {int(k): str(v) for k, v in names.items()}
    else:
        out = {i: str(v) for i, v in enumerate(names)}

    for i in range(num_classes):
        out.setdefault(i, f"class_{i}")

    return out


class OfficialYOLODetectionDataset(YOLODataset):
    """
    EvoDet wrapper around Ultralytics official YOLODataset.

    Benefits:
        - Official image loading
        - Official label cache
        - Official v8_transforms / Mosaic / MixUp / LetterBox / Format
        - Less fragile than manually emulating YOLODataset internals

    Returned item is adapted back to EvoDet format:
        img/images/image
        labels / targets: [cls, x, y, w, h]
        image_path / label_path / orig_shape / letterbox
    """

    def __init__(
            self,
            root: str | Path,
            split: str | Path,
            image_size: int = 640,
            num_classes: int = 80,
            names: Optional[Sequence[str] | dict[int, str]] = None,
            class_filter: Optional[Sequence[int]] = None,
            include_empty: bool = False,
            augment: bool = False,
            hyp: Optional[SimpleNamespace] = None,
            stride: int = 32,
            rect: bool = False,
            batch_size: int = 16,
            cache: bool | str = False,
            single_cls: bool = False,
            fraction: float = 1.0,
            pad: float = 0.0,
            prefix: str = "",
    ) -> None:
        self.root = Path(root)
        self.split = Path(split)
        self.image_size = int(image_size)
        self.imgsz = int(image_size)

        self.num_classes = int(num_classes)
        self.class_filter = (
            set(int(c) for c in class_filter)
            if class_filter is not None
            else None
        )
        self.include_empty = bool(include_empty)

        self.names = normalize_class_names(names, self.num_classes)

        self.evodet_data = {
            "names": self.names,
            "nc": self.num_classes,
            "channels": 3,
        }

        img_path = self._resolve_img_path(self.root, self.split)
        hyp = hyp or build_ultralytics_hyp()
        self.hyp = hyp
        # Do NOT pass task="detect" here.
        # Older Ultralytics versions do not support it.
        #
        # Also do NOT call:
        #     super().__init__(**filtered_kwargs)
        #
        # because some Ultralytics YOLODataset versions use *args/**kwargs and
        # inspect.signature() cannot see the required img_path. Therefore img_path
        # must be passed as the first positional argument.
        base_kwargs = {
            "imgsz": self.imgsz,
            "cache": cache,
            "augment": bool(augment),
            "hyp": hyp,
            "prefix": prefix,
            "rect": bool(rect),
            "batch_size": int(batch_size),
            "stride": int(stride),
            "pad": float(pad),
            "single_cls": bool(single_cls),
            "classes": None,
            "fraction": float(fraction),
            "data": self.evodet_data,
        }

        try:
            super().__init__(
                str(img_path),
                **base_kwargs,
            )
        except TypeError as exc:
            retry_kwargs = dict(base_kwargs)
            last_exc = exc

            # Different Ultralytics versions support slightly different BaseDataset
            # arguments. Remove the least essential ones one by one.
            unsupported_candidates = [
                "fraction",
                "classes",
                "single_cls",
                "pad",
                "data",
            ]

            for key in unsupported_candidates:
                if key not in retry_kwargs:
                    continue

                retry_kwargs.pop(key)

                try:
                    print(
                        "[OfficialYOLODetectionDataset] retry YOLODataset init "
                        f"after removing unsupported kwarg: {key}"
                    )
                    super().__init__(
                        str(img_path),
                        **retry_kwargs,
                    )
                    break
                except TypeError as retry_exc:
                    last_exc = retry_exc
                    continue
            else:
                raise last_exc

        # Keep EvoDet attributes after Ultralytics initialization as well,
        # because official YOLODataset may overwrite some common fields.
        self.root = Path(root)
        self.split = Path(split)
        self.image_size = int(image_size)
        self.imgsz = int(image_size)
        self.hyp = hyp
        self.num_classes = int(num_classes)
        self.class_filter = (
            set(int(c) for c in class_filter)
            if class_filter is not None
            else None
        )
        self.include_empty = bool(include_empty)
        self.names = normalize_class_names(names, self.num_classes)
        self.evodet_data = {
            "names": self.names,
            "nc": self.num_classes,
            "channels": 3,
        }

        self._apply_class_filter_after_init()

        print(
            "[OfficialYOLODetectionDataset] "
            f"split={split}, augment={augment}, "
            f"class_filter={sorted(self.class_filter) if self.class_filter is not None else None}, "
            f"include_empty={self.include_empty}, "
            f"num_images={len(self)}"
        )

    def get_labels(self) -> list[dict[str, Any]]:
        """
        Do NOT filter labels here.

        Ultralytics YOLODataset requires:
            self.im_files[i] <-> self.labels[i]

        If we remove labels inside get_labels(), self.im_files will still contain
        the original image list, causing image-label mismatch.

        Task-specific filtering is done after official initialization in
        _apply_class_filter_after_init().
        """
        return super().get_labels()

    def _apply_class_filter_after_init(self) -> None:
        """
        Apply task class filtering AFTER official YOLODataset initialization.

        This keeps:
            self.labels
            self.im_files
            self.label_files
            self.npy_files
            self.ims
            self.im_hw0
            self.im_hw

        aligned.
        """

        if self.class_filter is None:
            if self.include_empty:
                return

            keep_indices = []
            new_labels = []

            for i, lb in enumerate(self.labels):
                cls = lb.get("cls", None)
                if cls is not None and len(cls) > 0:
                    keep_indices.append(i)
                    new_labels.append(lb)

            self._keep_indices_and_labels(keep_indices, new_labels)
            return

        keep_indices = []
        new_labels = []

        for i, lb in enumerate(self.labels):
            cls = lb.get("cls", None)
            bboxes = lb.get("bboxes", None)

            if cls is None or bboxes is None or len(cls) == 0:
                if self.include_empty:
                    empty_lb = dict(lb)
                    empty_lb["cls"] = cls[:0] if cls is not None else np.zeros((0, 1), dtype=np.float32)
                    empty_lb["bboxes"] = bboxes[:0] if bboxes is not None else np.zeros((0, 4), dtype=np.float32)
                    empty_lb["segments"] = []
                    keep_indices.append(i)
                    new_labels.append(empty_lb)
                continue

            cls_np = np.asarray(cls).reshape(-1)

            keep_mask = np.array(
                [int(c) in self.class_filter for c in cls_np],
                dtype=bool,
            )

            if not keep_mask.any():
                if self.include_empty:
                    empty_lb = dict(lb)
                    empty_lb["cls"] = cls[:0]
                    empty_lb["bboxes"] = bboxes[:0]
                    empty_lb["segments"] = []
                    keep_indices.append(i)
                    new_labels.append(empty_lb)
                continue

            new_lb = dict(lb)
            new_lb["cls"] = cls[keep_mask].reshape(-1, 1)
            new_lb["bboxes"] = bboxes[keep_mask]

            if "segments" in new_lb and isinstance(new_lb["segments"], list):
                if len(new_lb["segments"]) == len(keep_mask):
                    new_lb["segments"] = [
                        seg
                        for seg, keep in zip(new_lb["segments"], keep_mask.tolist())
                        if keep
                    ]
                else:
                    new_lb["segments"] = []

            keep_indices.append(i)
            new_labels.append(new_lb)

        self._keep_indices_and_labels(keep_indices, new_labels)

    def _keep_indices_and_labels(
            self,
            keep_indices: list[int],
            new_labels: list[dict[str, Any]],
    ) -> None:
        old_num = len(getattr(self, "labels", []))

        self.labels = new_labels

        if hasattr(self, "im_files"):
            self.im_files = [self.im_files[i] for i in keep_indices]

        if hasattr(self, "label_files"):
            self.label_files = [self.label_files[i] for i in keep_indices]

        if hasattr(self, "npy_files"):
            self.npy_files = [self.npy_files[i] for i in keep_indices]

        for attr in ["ims", "im_hw0", "im_hw"]:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if isinstance(value, list) and len(value) == old_num:
                    setattr(self, attr, [value[i] for i in keep_indices])

        self.ni = len(self.labels)
        self.indices = range(self.ni)

        self._reset_ultralytics_buffer_after_filter()

        if getattr(self, "rect", False):
            try:
                self.set_rectangle()
            except Exception as exc:
                print(
                    "[OfficialYOLODetectionDataset] warning: set_rectangle failed "
                    f"after filtering: {exc}"
                )

        if self.ni == 0:
            raise RuntimeError(
                "No samples left after class filtering. "
                f"class_filter={sorted(self.class_filter) if self.class_filter is not None else None}, "
                f"include_empty={self.include_empty}"
            )

    def _reset_ultralytics_buffer_after_filter(self) -> None:
        """
        Reset Ultralytics buffer after task-specific filtering.

        This follows original YOLOv8 behavior:

            self.buffer = []
            self.max_buffer_length = min(self.ni, self.batch_size * 8, 1000)
                                     if augment else 0

        Important:
            buffer is not a full candidate pool.
            It is a recent-image sliding window maintained by load_image().
        """

        if not hasattr(self, "buffer"):
            return

        self.buffer = []

        batch_size = int(getattr(self, "batch_size", 16) or 16)

        if bool(getattr(self, "augment", False)):
            self.max_buffer_length = min(
                int(self.ni),
                batch_size * 8,
                1000,
            )
        else:
            self.max_buffer_length = 0

    def enable_pseudo_label_mode(self):
        """
        Temporarily disable training augmentations for pseudo-label generation.

        Pseudo labels must be generated on normal single images, not Mosaic/MixUp
        images. Otherwise pseudo boxes will not correspond to original image paths.
        """

        saved_state = {
            "augment": getattr(self, "augment", False),
            "transforms": getattr(self, "transforms", None),
            "buffer": list(getattr(self, "buffer", [])) if hasattr(self, "buffer") else None,
            "hyp_values": {},
        }

        hyp = getattr(self, "hyp", None)

        if hyp is None:
            hyp = build_ultralytics_hyp()
            self.hyp = hyp

        for key in ["mosaic", "mixup", "cutmix", "copy_paste"]:
            if hasattr(hyp, key):
                saved_state["hyp_values"][key] = getattr(hyp, key)
                setattr(hyp, key, 0.0)

        self.augment = False

        if hasattr(self, "buffer"):
            self.buffer = []

        if hasattr(self, "max_buffer_length"):
            self.max_buffer_length = 0

        # Rebuild transforms in no-augmentation mode.
        try:
            self.transforms = self.build_transforms(hyp)
        except TypeError:
            self.transforms = self.build_transforms()

        def restore():
            self.augment = saved_state["augment"]

            hyp_restore = getattr(self, "hyp", None)
            if hyp_restore is not None:
                for key, value in saved_state["hyp_values"].items():
                    setattr(hyp_restore, key, value)

            if saved_state["buffer"] is not None and hasattr(self, "buffer"):
                self.buffer = saved_state["buffer"]

            if saved_state["transforms"] is not None:
                self.transforms = saved_state["transforms"]

        return restore

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)

        img = item["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        if img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1).contiguous()

        if img.dtype != torch.float32:
            img = img.float()

        if img.numel() > 0 and img.max() > 2.0:
            img = img / 255.0

        cls = item.get("cls", torch.zeros((0, 1), dtype=torch.float32))
        bboxes = item.get("bboxes", torch.zeros((0, 4), dtype=torch.float32))

        if not isinstance(cls, torch.Tensor):
            cls = torch.as_tensor(cls, dtype=torch.float32)
        else:
            cls = cls.float()

        if not isinstance(bboxes, torch.Tensor):
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        else:
            bboxes = bboxes.float()

        cls = cls.view(-1, 1)
        bboxes = bboxes.view(-1, 4)

        if cls.numel() == 0 or bboxes.numel() == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.cat([cls, bboxes], dim=1)
            labels[:, 1:].clamp_(0.0, 1.0)

        im_file = item.get("im_file", None)
        image_path = str(im_file) if im_file is not None else str(self.im_files[index])

        label_path = self._image_to_label_path(Path(image_path))

        ori_shape = item.get("ori_shape", None)
        if ori_shape is None:
            ori_shape = item.get("shape", None)

        if ori_shape is None:
            ori_shape = (int(img.shape[-2]), int(img.shape[-1]))

        letterbox = self._infer_letterbox(
            item=item,
            img=img,
            ori_shape=ori_shape,
        )

        return {
            "image": img,
            "images": img,
            "img": img,
            "labels": labels,
            "targets": labels,
            "image_path": image_path,
            "label_path": str(label_path),
            "orig_shape": tuple(int(x) for x in ori_shape),
            "letterbox": letterbox,
            "is_replay": False,
            "task_id": -1,
            "replay_task_id": -1,
        }

    def _resolve_img_path(
        self,
        root: Path,
        split: Path,
    ) -> Path:
        if split.is_absolute():
            return split

        return root / split

    def _image_to_label_path(
        self,
        image_path: Path,
    ) -> Path:
        s = str(image_path)

        if "/images/" in s:
            return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")

        try:
            rel = image_path.relative_to(self.root)
            parts = list(rel.parts)
            if parts and parts[0] == "images":
                parts[0] = "labels"
                return (self.root / Path(*parts)).with_suffix(".txt")
        except ValueError:
            pass

        return image_path.with_suffix(".txt")

    def _infer_letterbox(
        self,
        item: dict[str, Any],
        img: torch.Tensor,
        ori_shape: Sequence[int],
    ) -> dict[str, float]:
        input_h, input_w = int(img.shape[-2]), int(img.shape[-1])
        orig_h, orig_w = int(ori_shape[0]), int(ori_shape[1])

        ratio_pad = item.get("ratio_pad", None)

        if ratio_pad is not None:
            try:
                ratio = ratio_pad[0]
                pad = ratio_pad[1]

                if isinstance(ratio, (tuple, list)):
                    scale = float(ratio[0])
                else:
                    scale = float(ratio)

                pad_left = float(pad[0])
                pad_top = float(pad[1])

                return {
                    "scale": scale,
                    "pad_left": pad_left,
                    "pad_top": pad_top,
                    "input_w": float(input_w),
                    "input_h": float(input_h),
                    "orig_w": float(orig_w),
                    "orig_h": float(orig_h),
                }
            except Exception:
                pass

        scale = min(input_w / max(1, orig_w), input_h / max(1, orig_h))
        resized_w = round(orig_w * scale)
        resized_h = round(orig_h * scale)

        pad_left = (input_w - resized_w) / 2
        pad_top = (input_h - resized_h) / 2

        return {
            "scale": float(scale),
            "pad_left": float(pad_left),
            "pad_top": float(pad_top),
            "input_w": float(input_w),
            "input_h": float(input_h),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }


# Backward-compatible name.
YOLODetectionDataset = OfficialYOLODetectionDataset


def yolo_detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["img"] for item in batch], dim=0)

    targets = []
    batch_idx = []
    cls = []
    bboxes = []

    for i, item in enumerate(batch):
        labels = item["labels"]

        if labels.numel() == 0:
            continue

        idx_col = torch.full((labels.shape[0], 1), i, dtype=torch.float32)
        targets.append(torch.cat([idx_col, labels], dim=1))

        batch_idx.append(torch.full((labels.shape[0],), i, dtype=torch.long))
        cls.append(labels[:, 0:1])
        bboxes.append(labels[:, 1:5])

    if targets:
        targets_tensor = torch.cat(targets, dim=0)
        batch_idx_tensor = torch.cat(batch_idx, dim=0)
        cls_tensor = torch.cat(cls, dim=0)
        bboxes_tensor = torch.cat(bboxes, dim=0)
    else:
        targets_tensor = torch.zeros((0, 6), dtype=torch.float32)
        batch_idx_tensor = torch.zeros((0,), dtype=torch.long)
        cls_tensor = torch.zeros((0, 1), dtype=torch.float32)
        bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

    return {
        "img": images,
        "images": images,
        "targets": targets_tensor,
        "batch_idx": batch_idx_tensor,
        "cls": cls_tensor,
        "bboxes": bboxes_tensor,
        "paths": [item["image_path"] for item in batch],
        "label_paths": [item.get("label_path", None) for item in batch],
        "orig_shape": [item.get("orig_shape", None) for item in batch],
        "letterbox": [item.get("letterbox", None) for item in batch],
        "is_replay": torch.tensor(
            [bool(item.get("is_replay", False)) for item in batch],
            dtype=torch.bool,
        ),
        "task_id": torch.tensor(
            [int(item.get("task_id", -1)) for item in batch],
            dtype=torch.long,
        ),
        "replay_task_id": torch.tensor(
            [int(item.get("replay_task_id", -1)) for item in batch],
            dtype=torch.long,
        ),
    }