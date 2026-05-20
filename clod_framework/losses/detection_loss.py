# clod_framework/losses/detection_loss.py

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor


@dataclass
class DetectionLossOutput:
    loss: Tensor
    loss_items: Dict[str, Tensor]

    def as_log_dict(self) -> Dict[str, Tensor]:
        out = {"loss": self.loss}
        out.update(self.loss_items)
        return out


class YOLOv8DetectionLoss(nn.Module):
    """
    Ultralytics YOLOv8 detection loss adapter.

    Required batch format for Ultralytics v8DetectionLoss:
        batch = {
            "img": Tensor[B, 3, H, W],
            "batch_idx": Tensor[N],
            "cls": Tensor[N, 1],
            "bboxes": Tensor[N, 4],  # normalized xywh
        }

    This wrapper accepts:
        - outputs from YOLOv8DetectorOutput
        - dict outputs with raw/preds/pred/outputs
        - Tensor/list raw outputs
        - framework batch with targets
    """

    def __init__(
        self,
        model: Any,
        device: Optional[Union[str, torch.device]] = None,
        loss_names: Optional[Sequence[str]] = None,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
    ) -> None:
        super().__init__()

        self.model = self._unwrap_model(model)
        self.device = torch.device(device) if device is not None else self._infer_device(self.model)

        self.loss_names = tuple(loss_names or ("box_loss", "cls_loss", "dfl_loss"))

        self.box_gain = float(box)
        self.cls_gain = float(cls)
        self.dfl_gain = float(dfl)

        self._ensure_ultralytics_loss_attrs(self.model)

        self.ultralytics_loss = self._build_ultralytics_loss(self.model)

    def forward(
        self,
        outputs: Any,
        targets: Any = None,
        batch: Optional[Mapping[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[Tensor, DetectionLossOutput]:
        preds = self._unwrap_outputs(outputs)

        if batch is None:
            batch = self.prepare_batch(targets=targets)

        batch = self._move_batch_to_device(batch, self.device)

        loss, loss_items = self.ultralytics_loss(preds, batch)

        loss_items_dict = self._format_loss_items(loss_items)

        if return_dict:
            return DetectionLossOutput(
                loss=loss,
                loss_items=loss_items_dict,
            )

        return loss

    def prepare_batch(
        self,
        targets: Any = None,
        images: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if isinstance(targets, Mapping):
            if self._is_ultralytics_batch(targets):
                return self._normalize_ultralytics_batch(targets)

            if "targets" in targets:
                images = targets.get("img", targets.get("images", images))
                targets = targets["targets"]
            else:
                raise ValueError(
                    "Unsupported target dict. Expected keys "
                    "'batch_idx', 'cls', 'bboxes' or a 'targets' entry."
                )

        if isinstance(targets, Tensor):
            return self._targets_tensor_to_batch(targets, images=images)

        if isinstance(targets, (list, tuple)):
            return self._targets_list_to_batch(targets)

        if targets is None:
            return self._empty_batch(images=images)

        raise TypeError(f"Unsupported targets type: {type(targets)!r}")

    def _build_ultralytics_loss(self, model: Any) -> Any:
        import importlib

        candidates = [
            ("ultralytics.utils.loss", "v8DetectionLoss"),
        ]

        errors = []

        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
                loss_cls = getattr(module, class_name)
                return loss_cls(model)
            except Exception as exc:
                errors.append(f"{module_name}.{class_name}: {repr(exc)}")

        raise ImportError(
            "Could not import a compatible YOLOv8 detection loss from Ultralytics.\n"
            "Tried:\n  - "
            + "\n  - ".join(errors)
        )

    def _ensure_ultralytics_loss_attrs(self, model: Any) -> None:
        """
        v8DetectionLoss expects a DetectionModel produced by the Ultralytics trainer.

        When we manually build:
            DetectionModel(cfg='yolov8n.yaml', nc=4)

        the model often lacks:
            model.args

        Add the minimum args required by v8DetectionLoss.
        """

        if not hasattr(model, "args") or model.args is None:
            model.args = SimpleNamespace(
                box=self.box_gain,
                cls=self.cls_gain,
                dfl=self.dfl_gain,
            )
        else:
            if not hasattr(model.args, "box"):
                model.args.box = self.box_gain
            if not hasattr(model.args, "cls"):
                model.args.cls = self.cls_gain
            if not hasattr(model.args, "dfl"):
                model.args.dfl = self.dfl_gain

        # Some Ultralytics versions also look for model.nc / model.names.
        head = self._get_head(model)

        if head is not None:
            if hasattr(head, "nc") and not hasattr(model, "nc"):
                model.nc = int(head.nc)

            if hasattr(head, "nc") and not hasattr(model, "names"):
                model.names = {i: f"class_{i}" for i in range(int(head.nc))}

        if not hasattr(model, "stride") and head is not None and hasattr(head, "stride"):
            model.stride = head.stride

    def _get_head(self, model: Any) -> Optional[Any]:
        modules = getattr(model, "model", None)

        if isinstance(modules, nn.Sequential) and len(modules) > 0:
            return modules[-1]

        if isinstance(modules, (list, tuple)) and len(modules) > 0:
            return modules[-1]

        if hasattr(model, "head"):
            return model.head

        return None

    def _unwrap_model(self, model: Any) -> Any:
        # YOLOv8Detector wrapper
        if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
            return getattr(model, "model")

        # Ultralytics YOLO wrapper
        if hasattr(model, "yolo") and hasattr(model.yolo, "model"):
            return model.yolo.model

        return model

    def _unwrap_outputs(self, outputs: Any) -> Any:
        if hasattr(outputs, "raw"):
            return outputs.raw

        if isinstance(outputs, Mapping):
            for key in ("preds", "pred", "raw", "outputs"):
                if key in outputs:
                    return outputs[key]

        return outputs

    def _infer_device(self, model: Any) -> torch.device:
        if isinstance(model, nn.Module):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_ultralytics_batch(self, batch: Mapping[str, Any]) -> bool:
        return "batch_idx" in batch and "cls" in batch and "bboxes" in batch

    def _normalize_ultralytics_batch(self, batch: Mapping[str, Any]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}

        if "img" in batch:
            out["img"] = batch["img"]
        elif "images" in batch:
            out["img"] = batch["images"]

        batch_idx = batch["batch_idx"]
        cls = batch["cls"]
        bboxes = batch["bboxes"]

        if not isinstance(batch_idx, Tensor):
            batch_idx = torch.as_tensor(batch_idx)

        if not isinstance(cls, Tensor):
            cls = torch.as_tensor(cls)

        if not isinstance(bboxes, Tensor):
            bboxes = torch.as_tensor(bboxes)

        out["batch_idx"] = batch_idx.long().view(-1)
        out["cls"] = cls.float().view(-1, 1)
        out["bboxes"] = bboxes.float().view(-1, 4)

        return out

    def _targets_tensor_to_batch(
        self,
        targets: Tensor,
        images: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if targets.numel() == 0:
            return self._empty_batch(images=images, device=targets.device)

        if targets.ndim != 2:
            raise ValueError(f"Target tensor must be 2D, got shape {tuple(targets.shape)}.")

        if targets.shape[1] == 6:
            batch_idx = targets[:, 0].long()
            cls = targets[:, 1:2].float()
            bboxes = targets[:, 2:6].float()
        elif targets.shape[1] == 5:
            if images is not None and images.ndim == 4 and images.shape[0] != 1:
                raise ValueError(
                    "Targets with shape [N, 5] do not include image indices. "
                    "Use [N, 6] targets for batch_size > 1."
                )
            batch_idx = torch.zeros(
                targets.shape[0],
                dtype=torch.long,
                device=targets.device,
            )
            cls = targets[:, 0:1].float()
            bboxes = targets[:, 1:5].float()
        else:
            raise ValueError(
                "Target tensor must have shape [N, 5] or [N, 6]. "
                f"Got {tuple(targets.shape)}."
            )

        batch: Dict[str, Tensor] = {
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

        if images is not None:
            batch["img"] = images

        return batch

    def _targets_list_to_batch(self, targets: Sequence[Any]) -> Dict[str, Tensor]:
        rows = []

        device = self.device

        for image_idx, target in enumerate(targets):
            if target is None:
                continue

            if not isinstance(target, Tensor):
                target = torch.as_tensor(target)

            target = target.to(device)

            if target.numel() == 0:
                continue

            if target.ndim != 2 or target.shape[1] not in (5, 6):
                raise ValueError(
                    "Each target item must have shape [N, 5] or [N, 6]. "
                    f"Got {tuple(target.shape)} for image {image_idx}."
                )

            if target.shape[1] == 6:
                rows.append(target)
            else:
                batch_idx = torch.full(
                    (target.shape[0], 1),
                    image_idx,
                    dtype=target.dtype,
                    device=target.device,
                )
                rows.append(torch.cat([batch_idx, target], dim=1))

        if not rows:
            return self._empty_batch(device=device)

        merged = torch.cat(rows, dim=0)
        return self._targets_tensor_to_batch(merged)

    def _empty_batch(
        self,
        images: Optional[Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Tensor]:
        device = torch.device(device) if device is not None else self.device

        batch: Dict[str, Tensor] = {
            "batch_idx": torch.zeros(0, dtype=torch.long, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

        if images is not None:
            batch["img"] = images

        return batch

    def _move_batch_to_device(
        self,
        batch: Mapping[str, Any],
        device: torch.device,
    ) -> Dict[str, Tensor]:
        moved: Dict[str, Tensor] = {}

        for key, value in batch.items():
            if isinstance(value, Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value

        moved["batch_idx"] = moved["batch_idx"].long().view(-1)
        moved["cls"] = moved["cls"].float().view(-1, 1)
        moved["bboxes"] = moved["bboxes"].float().view(-1, 4)

        return moved

    def _format_loss_items(self, loss_items: Any) -> Dict[str, Tensor]:
        if isinstance(loss_items, Mapping):
            return {
                str(k): v if isinstance(v, Tensor) else torch.as_tensor(v, device=self.device)
                for k, v in loss_items.items()
            }

        if isinstance(loss_items, Tensor):
            flat = loss_items.detach().view(-1)
            return {
                name: flat[i] if i < flat.numel() else torch.zeros((), device=self.device)
                for i, name in enumerate(self.loss_names)
            }

        if isinstance(loss_items, Iterable):
            values = list(loss_items)
            out: Dict[str, Tensor] = {}

            for i, name in enumerate(self.loss_names):
                if i < len(values):
                    value = values[i]
                    out[name] = value if isinstance(value, Tensor) else torch.as_tensor(
                        value,
                        device=self.device,
                    )
                else:
                    out[name] = torch.zeros((), device=self.device)

            return out

        return {}


class DetectionLoss(YOLOv8DetectionLoss):
    pass


def build_detection_loss(
    model: Any,
    backend: str = "yolov8",
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> DetectionLoss:
    backend = backend.lower()

    if backend in {"yolov8", "ultralytics", "ultralytics_yolov8"}:
        return DetectionLoss(
            model=model,
            device=device,
            **kwargs,
        )

    raise ValueError(f"Unsupported detection loss backend: {backend}")