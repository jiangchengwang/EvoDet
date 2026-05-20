from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from clod_framework.data.task import Task
from clod_framework.models.base_detector import BaseDetector


class BaseMethod(ABC):
    """Algorithm plugin interface."""

    def __init__(self, model: BaseDetector) -> None:
        self.model = model

    def on_task_start(self, task: Task) -> None:
        pass

    @abstractmethod
    def training_step(self, batch: Any, task: Task) -> dict[str, Any]:
        raise NotImplementedError

    def validation_step(self, batch: Any, task: Task) -> dict[str, Any]:
        outputs = self.model.forward(batch)
        return {"outputs": outputs}

    def on_task_end(self, task: Task) -> None:
        pass
