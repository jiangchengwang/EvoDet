# clod_framework/methods/finetune.py

from __future__ import annotations

from typing import Any, Optional

import torch

from clod_framework.losses.detection_loss import DetectionLoss


class FinetuneMethod:
    def __init__(
        self,
        model: Any,
        detection_loss: Optional[Any] = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.detection_loss = detection_loss
        self.device = torch.device(device) if device is not None else self._infer_device(model)

    def on_task_start(self, task: Any) -> None:
        pass

    def training_step(self, batch: Any, task: Any = None) -> dict[str, Any]:
        outputs = self._forward_model(batch)

        if self.detection_loss is None:
            self.detection_loss = DetectionLoss(
                model=self.model,
                device=self.device,
            )

        result = self.detection_loss(
            outputs=outputs,
            targets=batch,
            return_dict=True,
        )

        return result.as_log_dict()

    def on_task_end(self, task: Any, metrics: dict[str, Any] | None = None) -> None:
        pass

    def _forward_model(self, batch: Any) -> Any:
        images = self._extract_images(batch)

        if images is None:
            keys = list(batch.keys()) if isinstance(batch, dict) else type(batch)
            raise ValueError(
                "Batch does not contain image tensor. "
                "Expected one of keys: 'img', 'images', 'image'. "
                f"Got: {keys}"
            )

        return self.model(images)

    def _extract_images(self, batch: Any):
        if isinstance(batch, torch.Tensor):
            return batch

        if isinstance(batch, dict):
            for key in ("img", "images", "image"):
                if key in batch:
                    return batch[key]

        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            if isinstance(batch[0], torch.Tensor):
                return batch[0]

        return None

    def _infer_device(self, model: Any) -> torch.device:
        if hasattr(model, "parameters"):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")