# scripts/train_incremental.py

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Allow running as:
#   python /workspace/EvoDet/scripts/train_incremental.py --config ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clod_framework.data.builder import (  # noqa: E402
    build_train_loader,
    build_val_loader,
    get_seen_classes,
)
from clod_framework.engine.evaluator import DetectionEvaluator  # noqa: E402
from clod_framework.engine.optim import build_yolov8_optimizer  # noqa: E402
from clod_framework.methods.finetune import FinetuneMethod  # noqa: E402
from clod_framework.methods.yolo_lwf_ocdm import YOLOLwFOCDMMethod  # noqa: E402
from clod_framework.models.builder import (  # noqa: E402
    build_model,
    get_model_num_classes,
    resolve_model_source_for_teacher,
)


@dataclass
class IncrementalTask:
    task_id: int
    classes: list[int]
    old_classes: list[int]
    seen_classes: list[int]
    train_dataset: Any = None
    val_dataset: Any = None


class CheckpointManager:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_task_model(self, model: Any, task_id: int) -> Path:
        path = self.checkpoint_dir / f"model_task_{task_id}.pt"
        torch_model = unwrap_torch_model(model)

        payload = {
            "type": "task_model",
            "task_id": int(task_id),
            "model_state_dict": torch_model.state_dict(),
        }

        torch.save(payload, path)
        print(f"[Checkpoint] saved task {task_id} model to {path}")
        return path

    def load_model(self, model: Any, path: str | Path, strict: bool = False) -> dict[str, Any]:
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu")

        if isinstance(ckpt, dict):
            state = (
                ckpt.get("model_state_dict", None)
                or ckpt.get("state_dict", None)
                or ckpt.get("model", None)
                or ckpt
            )
        else:
            state = ckpt

        if hasattr(state, "state_dict"):
            state = state.state_dict()

        missing, unexpected = unwrap_torch_model(model).load_state_dict(state, strict=strict)

        print(
            f"[Checkpoint] loaded model from {path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
        )

        return ckpt if isinstance(ckpt, dict) else {}

    def latest_training_state_path(self) -> Path:
        return self.checkpoint_dir / "training_state_latest.pt"

    def save_training_state(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        method: Any,
        task_id: int,
        next_epoch: int,
        global_step: int,
        cfg: dict[str, Any],
    ) -> Path:
        path = self.latest_training_state_path()
        tmp_path = path.with_suffix(".tmp")

        payload = {
            "type": "training_state",
            "task_id": int(task_id),
            "next_epoch": int(next_epoch),
            "global_step": int(global_step),
            "model_state_dict": unwrap_torch_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "method_state_dict": method.state_dict() if hasattr(method, "state_dict") else {},
            "rng_state": get_rng_state(),
            "config": cfg,
        }

        torch.save(payload, tmp_path)
        tmp_path.replace(path)

        return path

    def load_training_state(self, path: str | Path | None = None) -> dict[str, Any]:
        path = Path(path) if path is not None else self.latest_training_state_path()

        if not path.exists():
            raise FileNotFoundError(f"Training state not found: {path}")

        state = torch.load(path, map_location="cpu")
        print(
            f"[Checkpoint] loaded training state from {path}: "
            f"task={state.get('task_id')}, next_epoch={state.get('next_epoch')}, "
            f"global_step={state.get('global_step')}"
        )
        return state

    def find_latest_completed_task(self, max_task_id: int) -> int:
        """
        A task is considered safely completed only if both files exist:
            model_task_{k}.pt
            replay_memory_task_{k}.json

        This prevents incorrectly resuming task1 when task0 model was saved
        but OCDM memory update failed.
        """

        latest = -1

        for task_id in range(max_task_id + 1):
            model_path = self.checkpoint_dir / f"model_task_{task_id}.pt"
            memory_path = self.checkpoint_dir / f"replay_memory_task_{task_id}.json"

            if model_path.exists() and memory_path.exists():
                latest = task_id
            else:
                break

        return latest

    def clear_training_state(self) -> None:
        path = self.latest_training_state_path()
        if path.exists():
            path.unlink()

def unwrap_torch_model(model: Any) -> nn.Module:
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model

    if isinstance(model, nn.Module):
        return model

    raise TypeError(f"Unsupported model type: {type(model)!r}")


def set_torch_train(model: Any, mode: bool = True) -> None:
    """
    Avoid accidentally calling Ultralytics high-level Model.train().
    For our wrapper, model.model is the raw torch DetectionModel.
    """
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        model.model.train(mode)
    elif isinstance(model, nn.Module):
        model.train(mode)
    else:
        raise TypeError(f"Unsupported model type: {type(model)!r}")


def move_model_to_device(model: Any, device: torch.device) -> None:
    if hasattr(model, "to"):
        model.to(device)
    elif hasattr(model, "model") and isinstance(model.model, nn.Module):
        model.model.to(device)
    else:
        raise TypeError(f"Unsupported model type: {type(model)!r}")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config: {path}")

    return cfg


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["cuda"] = None

    return state


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])

    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])

    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and state.get("cuda", None) is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)

def build_tasks(cfg: dict[str, Any]) -> list[IncrementalTask]:
    dataset_cfg = cfg.get("dataset", {})
    task_cfg = cfg.get("task", {})

    num_classes = int(dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80)))
    initial_classes = int(task_cfg.get("initial_classes", 40))
    increment = int(task_cfg.get("increment", 10))

    class_order = task_cfg.get("class_order", "natural")

    if isinstance(class_order, str):
        if class_order == "natural":
            order = list(range(num_classes))
        else:
            raise ValueError(
                f"Unsupported class_order string: {class_order}. "
                "Use 'natural' or an explicit list of class ids."
            )
    else:
        order = [int(x) for x in class_order]

    if sorted(order) != list(range(num_classes)):
        raise ValueError(
            f"class_order must contain all classes 0..{num_classes - 1}. "
            f"Got len={len(order)}."
        )

    tasks: list[IncrementalTask] = []

    start = 0
    task_id = 0

    first = order[:initial_classes]
    seen = list(first)

    tasks.append(
        IncrementalTask(
            task_id=task_id,
            classes=list(first),
            old_classes=[],
            seen_classes=list(seen),
        )
    )

    start = initial_classes
    task_id += 1

    while start < num_classes:
        current = order[start : start + increment]
        old = list(seen)
        seen = old + list(current)

        tasks.append(
            IncrementalTask(
                task_id=task_id,
                classes=list(current),
                old_classes=list(old),
                seen_classes=list(seen),
            )
        )

        start += increment
        task_id += 1

    print("[Tasks]")
    for task in tasks:
        print(
            f"  task={task.task_id}, "
            f"classes={task.classes}, "
            f"old_classes={task.old_classes}, "
            f"seen_classes={task.seen_classes}"
        )

    return tasks


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None

    return float(value)


def build_method(
    cfg: dict[str, Any],
    model: Any,
    tasks: list[IncrementalTask],
) -> Any:
    experiment_cfg = cfg.get("experiment", {})
    dataset_cfg = cfg.get("dataset", {})
    method_cfg = cfg.get("method", {})
    training_cfg = cfg.get("training", {})

    method_name = str(method_cfg.get("name", "finetune")).lower()
    output_dir = Path(experiment_cfg.get("output_dir", "outputs/default"))

    device = training_cfg.get(
        "device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    if method_name == "finetune":
        return FinetuneMethod(
            model=model,
            device=device,
        )

    if method_name in {"yolo_lwf_ocdm", "lwf_ocdm", "yololwf_ocdm"}:
        replay_cfg = method_cfg.get("replay", {})
        distill_cfg = method_cfg.get("distill", {})

        classes_per_task = [list(map(int, task.classes)) for task in tasks]

        model_source = resolve_model_source_for_teacher(cfg)

        return YOLOLwFOCDMMethod(
            model=model,
            output_dir=output_dir,
            num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80))),
            model_source=model_source,
            device=device,
            lambda_distill=float(method_cfg.get("lambda_distill", 1.0)),
            memory_size=int(replay_cfg.get("memory_size", 6000)),
            image_size=int(training_cfg.get("img_size", training_cfg.get("image_size", 640))),
            lwf_c1=float(distill_cfg.get("c1", distill_cfg.get("lambda_cls", 1.0))),
            lwf_c2=float(distill_cfg.get("c2", distill_cfg.get("lambda_reg", 1.0))),
            lwf_c3=optional_float(distill_cfg.get("c3", None)),
            lwf_gain=float(distill_cfg.get("lwf_gain", method_cfg.get("lambda_distill", 1.0))),
            use_new_images_for_lwf=bool(distill_cfg.get("use_new_images", True)),
            use_replay_labels=bool(replay_cfg.get("use_labels", True)),
            pseudo_conf_thres=float(replay_cfg.get("pseudo_conf_thres", 0.5)),
            pseudo_iou_thres=float(replay_cfg.get("pseudo_iou_thres", 0.7)),
            classes_per_task=classes_per_task,
            debug_loss_scaling=bool(distill_cfg.get("debug_loss_scaling", False)),
            batch_size_ocdm=int(replay_cfg.get("batch_size_ocdm", -1)),
        )

    raise ValueError(f"Unsupported method: {method_name}")


def build_evaluator(cfg: dict[str, Any]) -> DetectionEvaluator:
    experiment_cfg = cfg.get("experiment", {})
    dataset_cfg = cfg.get("dataset", {})
    evaluation_cfg = cfg.get("evaluation", {})
    training_cfg = cfg.get("training", {})

    output_dir = Path(experiment_cfg.get("output_dir", "outputs/default"))
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    class_names = dataset_cfg.get("names", None)

    return DetectionEvaluator(
        num_classes=int(dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80))),
        metrics_dir=metrics_dir,
        device=training_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu"),
        conf_thres=float(evaluation_cfg.get("conf_thres", 0.001)),
        nms_iou_thres=float(evaluation_cfg.get("iou_thres", 0.7)),
        max_det=int(evaluation_cfg.get("max_det", 300)),
        save_csv=bool(evaluation_cfg.get("save_csv", True)),
        print_results=bool(evaluation_cfg.get("print_results", True)),
        class_names=class_names,
    )


def rebuild_train_loader_with_method(
    cfg: dict[str, Any],
    task: IncrementalTask,
    method: Any,
):
    base_loader = build_train_loader(cfg, task)
    current_dataset = base_loader.dataset
    task.train_dataset = current_dataset

    if hasattr(method, "build_train_loader"):
        return method.build_train_loader(
            cfg=cfg,
            task=task,
            current_dataset=current_dataset,
        )

    return base_loader


def make_zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def scalar_from_output(
    outputs: dict[str, Any],
    key: str,
    device: torch.device,
    fallback_key: str | None = None,
    fallback_value: float = 0.0,
) -> torch.Tensor:
    if key in outputs:
        value = outputs[key]
    elif fallback_key is not None and fallback_key in outputs:
        value = outputs[fallback_key]
    else:
        value = torch.tensor(float(fallback_value), device=device)

    if isinstance(value, torch.Tensor):
        return value.detach()

    return torch.tensor(float(value), device=device)


def train_one_task(
    cfg: dict[str, Any],
    model: Any,
    method: Any,
    task: IncrementalTask,
    checkpoints: CheckpointManager,
    evaluator: DetectionEvaluator,
    resume_state: dict[str, Any] | None = None,
) -> None:
    training_cfg = cfg.get("training", {})

    device = torch.device(
        training_cfg.get(
            "device",
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )
    )

    epochs = int(training_cfg.get("epochs_per_task", 100))
    lr0 = float(training_cfg.get("lr", training_cfg.get("lr0", 0.01)))
    lrf = float(training_cfg.get("min_lr_ratio", training_cfg.get("lrf", 0.01)))
    weight_decay = float(training_cfg.get("weight_decay", 0.0005))
    momentum = float(training_cfg.get("momentum", 0.937))
    warmup_momentum = float(training_cfg.get("warmup_momentum", 0.8))
    warmup_bias_lr = float(training_cfg.get("warmup_bias_lr", 0.1))
    warmup_epochs = float(training_cfg.get("warmup_epochs", 3.0))
    grad_clip = float(training_cfg.get("grad_clip", 10.0))
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"

    accumulate = int(training_cfg.get("accumulate", 1))
    nbs = int(training_cfg.get("nbs", 64))
    optimizer_name = str(training_cfg.get("optimizer", "SGD"))

    print_loss_scaling = bool(training_cfg.get("print_loss_scaling", True))
    loss_debug_interval = int(training_cfg.get("loss_debug_interval", 500))

    move_model_to_device(model, device)

    method.on_task_start(task)

    train_loader = rebuild_train_loader_with_method(cfg, task, method)
    val_loader = build_val_loader(cfg, task)
    task.val_dataset = val_loader.dataset

    optimizer = build_yolov8_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr0,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=int(training_cfg.get("batch_size", 8)),
        accumulate=accumulate,
        nbs=nbs,
    )

    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(max(1, warmup_epochs * len(train_loader)))

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    global_step = 0

    start_epoch = 0

    if resume_state is not None:
        state_task_id = int(resume_state.get("task_id", -1))

        if state_task_id == int(task.task_id):
            print(
                f"[Resume] restoring task={task.task_id}, "
                f"next_epoch={resume_state.get('next_epoch', 0)}, "
                f"global_step={resume_state.get('global_step', 0)}"
            )

            unwrap_torch_model(model).load_state_dict(
                resume_state["model_state_dict"],
                strict=False,
            )

            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            optimizer_to_device(optimizer, device)

            if "scaler_state_dict" in resume_state:
                try:
                    scaler.load_state_dict(resume_state["scaler_state_dict"])
                except Exception as exc:
                    print(f"[Resume] warning: failed to load scaler state: {exc}")

            if hasattr(method, "load_state_dict"):
                method.load_state_dict(resume_state.get("method_state_dict", {}))

            set_rng_state(resume_state.get("rng_state", None))

            start_epoch = int(resume_state.get("next_epoch", 0))
            global_step = int(resume_state.get("global_step", 0))

    def adjust_optimizer(step: int) -> float:
        if step < warmup_steps:
            warmup_factor = float(step + 1) / float(warmup_steps)

            lr_now = lr0 * warmup_factor
            momentum_now = warmup_momentum + warmup_factor * (momentum - warmup_momentum)

            for group in optimizer.param_groups:
                group_name = str(group.get("group_name", ""))

                if group_name == "bias_no_decay":
                    group["lr"] = warmup_bias_lr + warmup_factor * (lr0 - warmup_bias_lr)
                else:
                    group["lr"] = lr_now

                if "momentum" in group:
                    group["momentum"] = momentum_now

                if "betas" in group:
                    beta2 = group["betas"][1]
                    group["betas"] = (momentum_now, beta2)

            return lr_now

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_now = lr0 * (lrf + (1.0 - lrf) * cosine)

        for group in optimizer.param_groups:
            group["lr"] = lr_now

            if "momentum" in group:
                group["momentum"] = momentum

            if "betas" in group:
                beta2 = group["betas"][1]
                group["betas"] = (momentum, beta2)

        return lr_now

    for epoch in range(start_epoch, epochs):
        set_torch_train(model, True)

        running_total = 0.0
        running_current = 0.0
        running_replay = 0.0
        running_lwf = 0.0
        running_continual = 0.0
        num_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"train task {task.task_id} epoch {epoch + 1}/{epochs}",
            ncols=150,
        )

        for batch in pbar:
            lr_now = adjust_optimizer(global_step)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = method.training_step(batch, task)
                loss = outputs["loss"]

            scaler.scale(loss).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_torch_model(model).parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_t = scalar_from_output(outputs, "total_loss", device, fallback_key="loss")
            current_t = scalar_from_output(outputs, "current_loss", device, fallback_key="det_loss")
            replay_t = scalar_from_output(outputs, "replay_loss", device)
            lwf_t = scalar_from_output(outputs, "lwf_loss", device, fallback_key="distill_loss")
            continual_t = scalar_from_output(outputs, "continual_loss", device)

            total_value = float(total_t.detach().cpu())
            current_value = float(current_t.detach().cpu())
            replay_value = float(replay_t.detach().cpu())
            lwf_value = float(lwf_t.detach().cpu())
            continual_value = float(continual_t.detach().cpu())

            running_total += total_value
            running_current += current_value
            running_replay += replay_value
            running_lwf += lwf_value
            running_continual += continual_value

            num_steps += 1
            global_step += 1

            avg_total = running_total / max(1, num_steps)
            avg_current = running_current / max(1, num_steps)
            avg_replay = running_replay / max(1, num_steps)
            avg_lwf = running_lwf / max(1, num_steps)

            pbar.set_postfix(
                {
                    "total": f"{avg_total:.4f}",
                    "cur": f"{avg_current:.4f}",
                    "rep": f"{avg_replay:.4f}",
                    "lwf": f"{avg_lwf:.4f}",
                    "lr": f"{lr_now:.2e}",
                }
            )

            if print_loss_scaling and (
                num_steps == 1
                or (loss_debug_interval > 0 and num_steps % loss_debug_interval == 0)
            ):
                current_scale = outputs.get("current_scale", None)
                continual_scale = outputs.get("continual_scale", None)
                lwf_gain = outputs.get("lwf_gain", None)
                num_current = outputs.get("num_current", None)
                num_replay = outputs.get("num_replay", None)
                num_labels = outputs.get("num_labels", None)

                def fmt(x: Any) -> str:
                    if x is None:
                        return "NA"
                    if isinstance(x, torch.Tensor):
                        return f"{float(x.detach().cpu()):.4f}"
                    return f"{float(x):.4f}"

                print(
                    "[LossScaling] "
                    f"task={task.task_id}, "
                    f"epoch={epoch + 1}, "
                    f"step={num_steps}/{len(train_loader)}, "
                    f"current={current_value:.6f}, "
                    f"replay={replay_value:.6f}, "
                    f"lwf={lwf_value:.6f}, "
                    f"continual={continual_value:.6f}, "
                    f"total={total_value:.6f}, "
                    f"current_scale={fmt(current_scale)}, "
                    f"continual_scale={fmt(continual_scale)}, "
                    f"lwf_gain={fmt(lwf_gain)}, "
                    f"num_current={fmt(num_current)}, "
                    f"num_replay={fmt(num_replay)}, "
                    f"num_labels={fmt(num_labels)}"
                )

        print(
            f"task={task.task_id} epoch={epoch + 1}/{epochs} "
            f"total={running_total / max(1, num_steps):.6f} "
            f"current={running_current / max(1, num_steps):.6f} "
            f"replay={running_replay / max(1, num_steps):.6f} "
            f"lwf={running_lwf / max(1, num_steps):.6f}"
        )

        seen_classes = get_seen_classes(task)

        print(
            f"[Eval] task={task.task_id}, "
            f"current_classes={list(task.classes)}, "
            f"seen_classes={seen_classes}"
        )

        evaluator.evaluate(
            model=model,
            dataloader=val_loader,
            epoch=epoch + 1,
            task_id=task.task_id,
            seen_classes=seen_classes,
        )

        state_path = checkpoints.save_training_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            method=method,
            task_id=int(task.task_id),
            next_epoch=epoch + 1,
            global_step=global_step,
            cfg=cfg,
        )

        print(f"[Checkpoint] saved resumable training state to {state_path}")

    checkpoints.save_task_model(model, task.task_id)

    if hasattr(method, "on_task_end"):
        try:
            method.on_task_end(task, cfg=cfg)
        except TypeError:
            method.on_task_end(task)

    # Task has completed successfully, so the epoch-level state is no longer needed.
    checkpoints.clear_training_state()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EvoDet class-incremental training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest training state, or from latest completed task boundary.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Optional explicit path to training_state_latest.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    experiment_cfg = cfg.get("experiment", {})
    output_dir = Path(experiment_cfg.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    seed = int(experiment_cfg.get("seed", 0))
    set_seed(seed)

    print(f"[Config] loaded: {args.config}")
    print(f"[Output] {output_dir}")
    print(f"[Seed] {seed}")

    tasks = build_tasks(cfg)

    model = build_model(cfg)

    print(f"[Model] num_classes={get_model_num_classes(cfg)}")

    method = build_method(cfg, model, tasks=tasks)
    evaluator = build_evaluator(cfg)
    checkpoints = CheckpointManager(output_dir)

    start_task_id = 0
    resume_state = None

    if args.resume:
        # 1. Prefer epoch-level training state.
        resume_path = Path(args.resume_path) if args.resume_path else checkpoints.latest_training_state_path()

        if resume_path.exists():
            resume_state = checkpoints.load_training_state(resume_path)

            start_task_id = int(resume_state["task_id"])

            if "method_state_dict" in resume_state and hasattr(method, "load_state_dict"):
                method.load_state_dict(resume_state["method_state_dict"])

            print(f"[Resume] epoch-level resume from task {start_task_id}")

        else:
            # 2. Fall back to task-boundary resume.
            latest_completed = checkpoints.find_latest_completed_task(max_task_id=len(tasks) - 1)

            if latest_completed >= 0:
                model_path = output_dir / "checkpoints" / f"model_task_{latest_completed}.pt"
                memory_path = output_dir / "checkpoints" / f"replay_memory_task_{latest_completed}.json"

                checkpoints.load_model(model, model_path, strict=False)

                if hasattr(method, "load_memory_from_path"):
                    method.load_memory_from_path(memory_path)

                start_task_id = latest_completed + 1

                print(
                    f"[Resume] task-boundary resume: "
                    f"latest_completed_task={latest_completed}, "
                    f"start_task={start_task_id}"
                )
            else:
                print("[Resume] no valid checkpoint found, start from task 0")

    for task in tasks[start_task_id:]:
        print(f"Starting task {task.task_id}: classes={task.classes}")

        task_resume_state = None
        if resume_state is not None and int(resume_state.get("task_id", -1)) == int(task.task_id):
            task_resume_state = resume_state

        train_one_task(
            cfg=cfg,
            model=model,
            method=method,
            task=task,
            checkpoints=checkpoints,
            evaluator=evaluator,
            resume_state=task_resume_state,
        )

        # Only use loaded epoch state once.
        resume_state = None

    print("[Done] incremental training finished")


if __name__ == "__main__":
    main()