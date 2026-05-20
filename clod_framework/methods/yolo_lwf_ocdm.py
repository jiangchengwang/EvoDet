# clod_framework/methods/yolo_lwf_ocdm.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from clod_framework.data.builder import get_seen_classes
from clod_framework.data.replay_pair_dataset import build_paired_replay_loader
from clod_framework.data.yolo_detection_dataset import yolo_detection_collate
from clod_framework.losses.yolo_lwf_replay_loss import OriginalStyleYOLOLwFReplayLoss
from clod_framework.replay.ocdm_memory import OCDMMemory


class YOLOLwFOCDMMethod:

    def __init__(
            self,
            model: Any,
            output_dir: str | Path,
            num_classes: int,
            model_source: str = "yolov8n.pt",
            device: str | torch.device = "cuda",
            lambda_distill: float = 1.0,
            memory_size: int = 6000,
            image_size: int = 640,
            lwf_c1: float = 1.0,
            lwf_c2: float = 1.0,
            lwf_c3: float | None = None,
            lwf_gain: float = 1.0,
            use_new_images_for_lwf: bool = True,
            use_replay_labels: bool = True,
            pseudo_conf_thres: float = 0.25,
            pseudo_iou_thres: float = 0.7,
            classes_per_task: list[list[int]] | None = None,
            debug_loss_scaling: bool = False,
            batch_size_ocdm: int = -1,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.num_classes = int(num_classes)
        self.model_source = model_source
        self.device = torch.device(device)
        self.lambda_distill = float(lambda_distill)
        self.image_size = int(image_size)
        self.pseudo_conf_thres = float(pseudo_conf_thres)
        self.pseudo_iou_thres = float(pseudo_iou_thres)
        self.classes_per_task = classes_per_task or []
        self.batch_size_ocdm = int(batch_size_ocdm)

        self.loss_fn = OriginalStyleYOLOLwFReplayLoss(
            model=self.model,
            device=self.device,
            c1=lwf_c1,
            c2=lwf_c2,
            c3=lwf_c3,
            use_new_images_for_lwf=use_new_images_for_lwf,
            use_replay_labels=use_replay_labels,
            lwf_gain=lwf_gain,
            debug_loss_scaling=debug_loss_scaling,
        )

        self.memory = OCDMMemory(
            memory_size=memory_size,
            num_classes=self.num_classes,
            max_num_classes=self.num_classes,
            stats_path=self.output_dir / "metrics" / "ocdm.csv",
            count_dup=True,
        )

        self.teacher = None
        self.current_task = None

    def on_task_start(self, task: Any) -> None:
        self.current_task = task
        task_id = int(task.task_id)

        if task_id == 0:
            self.teacher = None
            print("[YOLO-LwF+OCDM] task 0: no teacher")
            return

        teacher_path = self.output_dir / "checkpoints" / f"model_task_{task_id - 1}.pt"
        print(f"[YOLO-LwF+OCDM] loading teacher: {teacher_path}")

        self.teacher = self.model.__class__.load_teacher(
            checkpoint_path=teacher_path,
            model=self.model_source,
            num_classes=self.num_classes,
            device=self.device,
            strict_load=False,
        )

        self.teacher.eval()
        self.teacher.requires_grad_(False)

    def build_train_loader(
        self,
        cfg: dict[str, Any],
        task: Any,
        current_dataset: Any,
    ) -> DataLoader:
        training_cfg = cfg.get("training", {})
        workers = int(training_cfg.get("workers", 0))
        pin_memory = bool(training_cfg.get("pin_memory", False))
        batch_size = int(training_cfg.get("batch_size", 8))
        seed = int(cfg.get("experiment", {}).get("seed", 0))

        if self.memory.is_empty():
            return DataLoader(
                current_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin_memory,
                collate_fn=yolo_detection_collate,
                drop_last=False,
                persistent_workers=workers > 0,
            )

        print(
            "[YOLO-LwF+OCDM] paired replay loader: "
            f"current={len(current_dataset)}, replay={len(self.memory)}"
        )

        return build_paired_replay_loader(
            current_dataset=current_dataset,
            memory=self.memory,
            batch_size=batch_size,
            image_size=self.image_size,
            workers=workers,
            pin_memory=pin_memory,
            seed=seed,
            current_task_id=int(task.task_id),
        )

    def training_step(
            self,
            batch: Any,
            task: Any = None,
    ) -> dict[str, torch.Tensor]:
        task = task if task is not None else self.current_task

        current_classes = list(getattr(task, "classes", []))
        old_classes = list(getattr(task, "old_classes", []))

        student_outputs = self.model(batch)

        teacher_outputs = None
        if self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(batch)

        out = self.loss_fn(
            student_outputs=student_outputs,
            batch=batch,
            teacher_outputs=teacher_outputs,
            current_classes=current_classes,
            old_classes=old_classes,
            classes_per_task=self.classes_per_task,
            return_dict=True,
        )

        return {
            "loss": out.loss,

            # Main logging keys.
            "current_loss": out.loss_items["current_loss"].detach(),
            "replay_loss": out.loss_items["replay_loss"].detach(),
            "lwf_loss": out.loss_items["lwf_loss"].detach(),
            "continual_loss": out.loss_items["continual_loss"].detach(),
            "total_loss": out.loss_items["total_loss"].detach(),

            # Backward-compatible aliases for old tqdm display.
            "det_loss": out.loss_items["current_loss"].detach(),
            "distill_loss": out.loss_items["lwf_loss"].detach(),

            **out.loss_items,
        }

    def on_task_end(
        self,
        task: Any,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        train_dataset = getattr(task, "train_dataset", None)

        if train_dataset is None:
            print("[YOLO-LwF+OCDM] no train_dataset found, skip memory update")
            return

        cfg = cfg or {}
        training_cfg = cfg.get("training", {})
        method_cfg = cfg.get("method", {})
        replay_cfg = method_cfg.get("replay", {})

        seen_classes = get_seen_classes(task)

        stats = self.memory.update_from_dataset(
            dataset=train_dataset,
            task_id=int(task.task_id),
            seen_classes=seen_classes,
            model=self.model,
            device=self.device,
            batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 8))),
            workers=int(training_cfg.get("workers", 0)),
            pseudo_conf_thres=float(replay_cfg.get("pseudo_conf_thres", self.pseudo_conf_thres)),
            pseudo_iou_thres=float(replay_cfg.get("pseudo_iou_thres", self.pseudo_iou_thres)),
            stats_path=self.output_dir / "metrics" / "ocdm.csv",
            batch_size_ocdm=int(replay_cfg.get("batch_size_ocdm", self.batch_size_ocdm)),
        )

        self.save_memory_for_task(int(task.task_id))

        print(
            "[YOLO-LwF+OCDM] memory updated: "
            f"size={stats['memory_size']}, "
            f"imgs_added={stats['imgs_added']}, "
            f"nc={stats['nc']}, "
            f"count_dup={stats['count_dup']}, "
            f"current_candidates={stats.get('num_current_candidates', 'NA')}, "
            f"refreshed_memory={stats.get('num_refreshed_memory', 'NA')}"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "model_source": self.model_source,
            "image_size": self.image_size,
            "pseudo_conf_thres": self.pseudo_conf_thres,
            "pseudo_iou_thres": self.pseudo_iou_thres,
            "classes_per_task": self.classes_per_task,
            "batch_size_ocdm": self.batch_size_ocdm,
            "memory": self.memory.state_dict(),
            "current_task_id": int(self.current_task.task_id) if self.current_task is not None else None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return

        self.num_classes = int(state.get("num_classes", self.num_classes))
        self.model_source = state.get("model_source", self.model_source)
        self.image_size = int(state.get("image_size", self.image_size))
        self.pseudo_conf_thres = float(state.get("pseudo_conf_thres", self.pseudo_conf_thres))
        self.pseudo_iou_thres = float(state.get("pseudo_iou_thres", self.pseudo_iou_thres))
        self.classes_per_task = state.get("classes_per_task", self.classes_per_task)
        self.batch_size_ocdm = int(state.get("batch_size_ocdm", self.batch_size_ocdm))

        memory_state = state.get("memory", None)
        if memory_state is not None:
            self.memory.load_state_dict(memory_state)

    def save_memory_for_task(self, task_id: int) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        latest_path = ckpt_dir / "replay_memory.json"
        task_path = ckpt_dir / f"replay_memory_task_{int(task_id)}.json"

        self.memory.save(latest_path)
        self.memory.save(task_path)

        print(f"[YOLO-LwF+OCDM] saved replay memory to {task_path}")

    def load_memory_from_path(self, path: str | Path) -> None:
        path = Path(path)
        loaded = OCDMMemory.load(path)
        self.memory.load_state_dict(loaded.state_dict())
        print(f"[YOLO-LwF+OCDM] loaded replay memory from {path}")