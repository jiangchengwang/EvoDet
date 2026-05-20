# clod_framework/models/base_detector.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseDetector(nn.Module, ABC):
    """
    Base detector interface.

    Model layer只负责：
        - forward
        - predict
        - save / load
        - device 管理

    不负责：
        - task split
        - replay memory
        - LwF
        - OCDM
        - UDA
    """

    def __init__(self, num_classes: int, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def forward(self, images: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @torch.no_grad()
    def predict_raw(self, images: torch.Tensor) -> Any:
        was_training = self.training
        self.eval()

        outputs = self.forward(images)

        if was_training:
            self.train()

        return outputs

    def save_checkpoint(
        self,
        path: Union[str, Path],
        task_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "task_id": task_id,
            "model_class": self.__class__.__name__,
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)
        return path

    def save_task_model(
        self,
        output_dir: Union[str, Path],
        task_id: int,
        filename: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"model_task_{task_id}.pt"

        return self.save_checkpoint(
            path=output_dir / filename,
            task_id=task_id,
            extra=extra,
        )

    def load_checkpoint(
        self,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = False,
    ) -> Dict[str, Any]:
        path = Path(path)
        map_location = map_location or self.device

        checkpoint = torch.load(path, map_location=map_location)
        state_dict = checkpoint.get("state_dict", checkpoint)

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        self.to(self.device)

        return {
            "checkpoint": checkpoint,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)

        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                self.device = torch.device(first)

        if "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])

        return self