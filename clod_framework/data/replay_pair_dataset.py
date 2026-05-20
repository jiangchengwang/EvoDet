# clod_framework/data/replay_pair_dataset.py

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from clod_framework.data.replay_yolo_dataset import ReplayYOLODataset
from clod_framework.data.yolo_detection_dataset import yolo_detection_collate
from clod_framework.replay.ocdm_memory import OCDMMemory


class PairedReplayDataset(Dataset):
    """
    Batch semantics:
        first half  = current task images
        second half = replay images

    This matches original YOLO_LwF replay loss assumptions.
    """

    def __init__(
        self,
        current_dataset: Dataset,
        replay_dataset: Dataset,
        shuffle_replay: bool = True,
        seed: int = 0,
        current_task_id: int = 0,
    ) -> None:
        if len(replay_dataset) == 0:
            raise ValueError("Replay dataset is empty.")

        self.current_dataset = current_dataset
        self.replay_dataset = replay_dataset
        self.shuffle_replay = bool(shuffle_replay)
        self.rng = random.Random(seed)
        self.current_task_id = int(current_task_id)

    def __len__(self) -> int:
        return len(self.current_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_item = dict(self.current_dataset[index])
        current_item["is_replay"] = False
        current_item["task_id"] = self.current_task_id
        current_item["replay_task_id"] = -1

        if self.shuffle_replay:
            replay_index = self.rng.randint(0, len(self.replay_dataset) - 1)
        else:
            replay_index = index % len(self.replay_dataset)

        replay_item = dict(self.replay_dataset[replay_index])
        replay_item["is_replay"] = True

        return {
            "current": current_item,
            "replay": replay_item,
        }


def paired_replay_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    current_items = [x["current"] for x in batch]
    replay_items = [x["replay"] for x in batch]

    merged = current_items + replay_items
    collated = yolo_detection_collate(merged)

    num_current = len(current_items)
    num_replay = len(replay_items)

    collated["num_current"] = num_current
    collated["num_replay"] = num_replay

    if collated["targets"].numel() == 0:
        num_labels = 0
    else:
        num_labels = int((collated["targets"][:, 0] < num_current).sum().item())

    collated["num_labels"] = num_labels

    task_ids = []
    for item in merged:
        if item.get("is_replay", False):
            task_ids.append(int(item.get("replay_task_id", item.get("task_id", -1))))
        else:
            task_ids.append(int(item.get("task_id", -1)))

    collated["task_id"] = torch.tensor(task_ids, dtype=torch.long)

    return collated


def build_paired_replay_loader(
    current_dataset: Dataset,
    memory: OCDMMemory,
    batch_size: int,
    image_size: int,
    workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    current_task_id: int = 0,
) -> DataLoader:
    if memory.is_empty():
        raise ValueError("Cannot build paired replay loader from empty memory.")

    if batch_size < 2:
        raise ValueError("For replay training, batch_size must be >= 2.")

    current_half_batch = max(1, batch_size // 2)

    replay_dataset = ReplayYOLODataset(
        samples=memory.as_samples(),
        image_size=image_size,
    )

    paired_dataset = PairedReplayDataset(
        current_dataset=current_dataset,
        replay_dataset=replay_dataset,
        shuffle_replay=True,
        seed=seed,
        current_task_id=current_task_id,
    )

    return DataLoader(
        paired_dataset,
        batch_size=current_half_batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=paired_replay_collate,
        drop_last=False,
        persistent_workers=workers > 0,
    )