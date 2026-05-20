# clod_framework/replay/replay_buffer.py

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


@dataclass
class ReplaySample:
    image_path: str
    label_path: Optional[str] = None
    labels: List[List[float]] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    task_id: int = -1
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplaySample":
        return cls(
            image_path=str(data["image_path"]),
            label_path=data.get("label_path"),
            labels=[list(map(float, row)) for row in data.get("labels", [])],
            class_ids=[int(x) for x in data.get("class_ids", [])],
            task_id=int(data.get("task_id", -1)),
            score=data.get("score"),
            meta=dict(data.get("meta", {})),
        )


class ReplayBuffer:
    """
    Pure replay memory.

    Important design rule:
        - Do NOT store trainer.
        - Do NOT store dataloader.
        - Do NOT store optimizer.
        - Do NOT store CUDA tensors.
        - Do NOT deepcopy method/trainer/model.

    This buffer stores only serializable metadata:
        - image_path
        - label_path
        - labels
        - class_ids
        - task_id
        - optional score/meta
    """

    def __init__(
        self,
        memory_size: int,
        num_classes: int,
        seed: int = 0,
        samples: Optional[Sequence[ReplaySample]] = None,
    ) -> None:
        self.memory_size = int(memory_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.samples: List[ReplaySample] = list(samples) if samples is not None else []
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[ReplaySample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> ReplaySample:
        return self.samples[index]

    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def clear(self) -> None:
        self.samples.clear()

    def add(self, sample: ReplaySample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.memory_size:
            self.samples = self.samples[-self.memory_size :]

    def extend(self, samples: Iterable[ReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def update(self, candidates: Iterable[ReplaySample], **_: Any) -> Dict[str, Any]:
        before = len(self.samples)
        self.extend(candidates)
        added = len(self.samples) - before
        return {
            "imgs_added": added,
            "memory_size": len(self.samples),
            "count_dup": self.count_duplicates(),
        }

    def sample(self, k: int) -> List[ReplaySample]:
        k = min(int(k), len(self.samples))
        if k <= 0:
            return []
        return self.rng.sample(self.samples, k)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "memory_size": self.memory_size,
            "num_classes": self.num_classes,
            "seed": self.seed,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.memory_size = int(state["memory_size"])
        self.num_classes = int(state["num_classes"])
        self.seed = int(state.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.samples = [ReplaySample.from_dict(x) for x in state.get("samples", [])]

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReplayBuffer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)

        buffer = cls(
            memory_size=int(state["memory_size"]),
            num_classes=int(state["num_classes"]),
            seed=int(state.get("seed", 0)),
        )
        buffer.load_state_dict(state)
        return buffer

    def class_histogram(self, normalize: bool = False) -> List[float]:
        hist = [0.0 for _ in range(self.num_classes)]

        for sample in self.samples:
            for class_id in sample.class_ids:
                if 0 <= int(class_id) < self.num_classes:
                    hist[int(class_id)] += 1.0

        if normalize:
            total = sum(hist)
            if total > 0:
                hist = [x / total for x in hist]

        return hist

    def image_paths(self) -> List[str]:
        return [sample.image_path for sample in self.samples]

    def count_duplicates(self) -> int:
        paths = self.image_paths()
        return len(paths) - len(set(paths))

    def as_dataset(self) -> "ReplayDataset":
        return ReplayDataset(self)


class ReplayDataset(Dataset):
    """
    Torch Dataset wrapper around ReplayBuffer.

    If image_loader is None, __getitem__ returns metadata only.
    You can pass your framework image loader later to return real tensors.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        image_loader: Optional[Any] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.image_loader = image_loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.replay_buffer[index]
        item: Dict[str, Any] = sample.to_dict()
        item["is_replay"] = True

        if self.image_loader is not None:
            image = self.image_loader(sample.image_path)
            item["image"] = image
            item["images"] = image

        if sample.labels:
            item["labels"] = torch.tensor(sample.labels, dtype=torch.float32)
            item["targets"] = item["labels"]
        else:
            item["labels"] = torch.zeros((0, 5), dtype=torch.float32)
            item["targets"] = item["labels"]

        if self.transform is not None:
            item = self.transform(item)

        return item


def collect_candidates_from_dataset(
    dataset: Any,
    task_id: int,
    max_samples: Optional[int] = None,
) -> List[ReplaySample]:
    """
    Best-effort conversion from a detection dataset to replay samples.

    This function supports common dataset styles:
        - dataset.im_files / dataset.label_files / dataset.labels
        - dataset[i] returning dict with image_path, label_path, labels
    """

    candidates: List[ReplaySample] = []

    if hasattr(dataset, "im_files"):
        image_paths = list(getattr(dataset, "im_files"))
        label_paths = list(getattr(dataset, "label_files", [None] * len(image_paths)))
        labels_list = getattr(dataset, "labels", [None] * len(image_paths))

        for idx, image_path in enumerate(image_paths):
            labels = _extract_labels_from_object(labels_list[idx] if idx < len(labels_list) else None)
            class_ids = _extract_class_ids(labels)

            candidates.append(
                ReplaySample(
                    image_path=str(image_path),
                    label_path=str(label_paths[idx]) if idx < len(label_paths) and label_paths[idx] else None,
                    labels=labels,
                    class_ids=class_ids,
                    task_id=task_id,
                )
            )

            if max_samples is not None and len(candidates) >= max_samples:
                break

        return candidates

    n = len(dataset)
    limit = n if max_samples is None else min(n, int(max_samples))

    for idx in range(limit):
        item = dataset[idx]

        image_path = _get_first(item, ("image_path", "img_path", "path", "im_file"))
        label_path = _get_first(item, ("label_path", "txt_path"))

        if image_path is None:
            image_path = str(idx)

        labels = _extract_labels_from_object(_get_first(item, ("labels", "targets", "bboxes")))
        class_ids = _extract_class_ids(labels)

        candidates.append(
            ReplaySample(
                image_path=str(image_path),
                label_path=str(label_path) if label_path is not None else None,
                labels=labels,
                class_ids=class_ids,
                task_id=task_id,
            )
        )

    return candidates


def _get_first(obj: Any, keys: Sequence[str]) -> Any:
    if isinstance(obj, Mapping):
        for key in keys:
            if key in obj:
                return obj[key]
    return None


def _extract_labels_from_object(obj: Any) -> List[List[float]]:
    if obj is None:
        return []

    if isinstance(obj, torch.Tensor):
        if obj.numel() == 0:
            return []
        return obj.detach().cpu().float().tolist()

    if isinstance(obj, Mapping):
        if "cls" in obj and "bboxes" in obj:
            cls = obj["cls"]
            bboxes = obj["bboxes"]

            if isinstance(cls, torch.Tensor):
                cls = cls.detach().cpu().view(-1).tolist()

            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().float().tolist()

            return [[float(c)] + [float(v) for v in box] for c, box in zip(cls, bboxes)]

        for key in ("labels", "targets"):
            if key in obj:
                return _extract_labels_from_object(obj[key])

    if isinstance(obj, list):
        if not obj:
            return []

        if all(isinstance(x, (list, tuple)) for x in obj):
            return [[float(v) for v in row] for row in obj]

    return []


def _extract_class_ids(labels: Sequence[Sequence[float]]) -> List[int]:
    ids = []
    for row in labels:
        if not row:
            continue
        ids.append(int(row[0]))
    return sorted(set(ids))