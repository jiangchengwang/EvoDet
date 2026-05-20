# clod_framework/data/replay_yolo_dataset.py

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from clod_framework.replay.ocdm_memory import ReplaySample


class ReplayYOLODataset(Dataset):
    """
    Replay dataset for stored YOLO samples.

    ReplaySample.labels format:
        [[cls, x, y, w, h], ...]

    The stored labels should be normalized xywh on the ORIGINAL image.
    This dataset will letterbox the image and update labels to normalized xywh
    on the letterboxed training image.
    """

    def __init__(
        self,
        samples: list[ReplaySample],
        image_size: int = 640,
    ) -> None:
        self.samples = samples
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        orig_w, orig_h = image.size

        labels = torch.tensor(sample.labels, dtype=torch.float32)

        if labels.numel() == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        image_tensor, labels, letterbox = self._letterbox_and_update_labels(
            image=image,
            labels=labels,
            new_size=self.image_size,
        )

        return {
            "image": image_tensor,
            "images": image_tensor,
            "img": image_tensor,
            "labels": labels,
            "targets": labels,
            "image_path": sample.image_path,
            "label_path": sample.label_path,
            "orig_shape": (orig_h, orig_w),
            "letterbox": letterbox,
            "is_replay": True,
            "replay_task_id": sample.task_id,
        }

    def _letterbox_and_update_labels(
        self,
        image: Image.Image,
        labels: torch.Tensor,
        new_size: int,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        orig_w, orig_h = image.size

        scale = min(new_size / orig_w, new_size / orig_h)
        resized_w = int(round(orig_w * scale))
        resized_h = int(round(orig_h * scale))

        pad_w = new_size - resized_w
        pad_h = new_size - resized_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        image_resized = image.resize((resized_w, resized_h), Image.BILINEAR)
        canvas = Image.new("RGB", (new_size, new_size), color)
        canvas.paste(image_resized, (pad_left, pad_top))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        letterbox = {
            "scale": float(scale),
            "pad_left": float(pad_left),
            "pad_top": float(pad_top),
            "input_w": float(new_size),
            "input_h": float(new_size),
            "orig_w": float(orig_w),
            "orig_h": float(orig_h),
        }

        if labels.numel() == 0:
            return tensor, labels, letterbox

        labels = labels.clone()

        # labels are normalized xywh on original image
        x = labels[:, 1] * orig_w
        y = labels[:, 2] * orig_h
        w = labels[:, 3] * orig_w
        h = labels[:, 4] * orig_h

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # original image coordinates -> letterboxed image coordinates
        x1 = x1 * scale + pad_left
        x2 = x2 * scale + pad_left
        y1 = y1 * scale + pad_top
        y2 = y2 * scale + pad_top

        x1.clamp_(0, new_size)
        x2.clamp_(0, new_size)
        y1.clamp_(0, new_size)
        y2.clamp_(0, new_size)

        # letterboxed xyxy -> normalized xywh on letterboxed image
        labels[:, 1] = ((x1 + x2) / 2) / new_size
        labels[:, 2] = ((y1 + y2) / 2) / new_size
        labels[:, 3] = (x2 - x1) / new_size
        labels[:, 4] = (y2 - y1) / new_size

        valid = (labels[:, 3] > 1e-6) & (labels[:, 4] > 1e-6)
        labels = labels[valid]

        return tensor, labels, letterbox