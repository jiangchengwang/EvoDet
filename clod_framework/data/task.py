from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Task:
    """A class-incremental/domain/transfer task descriptor."""

    task_id: int
    class_ids: Sequence[int]
    seen_class_ids: Sequence[int]
    train_dataset: Any | None = None
    val_dataset: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_new_classes(self) -> int:
        return len(self.class_ids)

    @property
    def num_seen_classes(self) -> int:
        return len(self.seen_class_ids)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    train_images: str
    val_images: str
    nc: int
    names: Sequence[str]
