# clod_framework/models/yolov8_detector.py

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import re
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

@dataclass
class YOLOv8DetectorOutput:
    raw: Any
    features: Optional[Dict[str, torch.Tensor]] = None


class YOLOv8Detector(nn.Module):
    def __init__(
            self,
            model: str = "yolov8s",
            num_classes: int = 80,
            pretrained: str | bool | None = None,
            pretrained_type: str | None = None,
            backbone_pretrained_max_module: int = 8,
            device: str | torch.device = "cuda",
            strict_load: bool = False,
    ) -> None:
        super().__init__()

        self.model_source = model
        self.num_classes = int(num_classes)
        self.pretrained = pretrained
        self.pretrained_type = pretrained_type
        self.backbone_pretrained_max_module = int(backbone_pretrained_max_module)
        self.device = torch.device(device)
        self.strict_load = bool(strict_load)

        self.model = self._build_detection_model(
            model=model,
            num_classes=self.num_classes,
        )

        self.model.to(self.device)

        if pretrained not in (None, False, "", "none", "None"):
            self._load_pretrained_weights(
                torch_model=self.model,
                pretrained=pretrained,
                pretrained_type=pretrained_type,
                backbone_max_module=self.backbone_pretrained_max_module,
            )

        self._print_detect_head_info()

    def _print_detect_head_info(self) -> None:
        head = self.model.model[-1]

        nc = getattr(head, "nc", None)
        no = getattr(head, "no", None)
        reg_max = getattr(head, "reg_max", None)

        print(
            f"[YOLOv8Detector] Detect head: nc={nc}, no={no}, reg_max={reg_max}"
        )

        try:
            cls_out = head.cv3[-1][-1].out_channels
            print(f"[YOLOv8Detector] class head out_channels={cls_out}")
        except Exception:
            pass

    def _build_detection_model(
            self,
            model: str,
            num_classes: int,
    ) -> nn.Module:
        cfg = self._resolve_detection_cfg(model)

        try:
            torch_model = DetectionModel(
                cfg=cfg,
                ch=3,
                nc=num_classes,
                verbose=False,
            )
        except TypeError:
            torch_model = DetectionModel(
                cfg=cfg,
                ch=3,
                nc=num_classes,
            )

        return torch_model

    def _build_model(
        self,
        model_source: str,
        num_classes: Optional[int],
        pretrained: bool,
        strict_load: bool,
    ) -> nn.Module:
        source = str(model_source)

        if source.endswith((".yaml", ".yml")):
            return self._build_detection_model_from_yaml(
                yaml_source=source,
                num_classes=num_classes,
            )

        if source.endswith(".pt"):
            yaml_source = self._pt_to_yaml_name(source)

            model = self._build_detection_model_from_yaml(
                yaml_source=yaml_source,
                num_classes=num_classes,
            )

            if pretrained:
                self._load_compatible_weights(
                    model=model,
                    weight_path=source,
                    strict=strict_load,
                )

            return model

        raise ValueError(f"Unsupported YOLOv8 model source: {source}")

    def _build_detection_model_from_yaml(
        self,
        yaml_source: str,
        num_classes: Optional[int],
    ) -> nn.Module:
        """
        Build a true YOLOv8 DetectionModel with nc=num_classes.

        This is the important part.

        Do NOT do:
            YOLO("yolov8n.pt").model
            detect.nc = 4
            detect.no = 68

        That only changes metadata and leaves the actual conv output channels at 80 classes.
        """

        from ultralytics.nn.tasks import DetectionModel

        if num_classes is None:
            model = DetectionModel(cfg=yaml_source, ch=3, nc=None, verbose=False)
        else:
            model = DetectionModel(cfg=yaml_source, ch=3, nc=int(num_classes), verbose=False)

        if num_classes is not None:
            model.nc = int(num_classes)
            model.names = {i: f"class_{i}" for i in range(int(num_classes))}

        return model


    def _resolve_detection_cfg(self, model: str) -> str:
        model = str(model)

        # yolov8s-cls.pt -> yolov8s.yaml
        if model.endswith("-cls.pt"):
            return model.replace("-cls.pt", ".yaml")

        # yolov8s.pt -> yolov8s.yaml
        if model.endswith(".pt"):
            return model.replace(".pt", ".yaml")

        # yolov8s.yaml
        if model.endswith((".yaml", ".yml")):
            return model

        # yolov8s -> yolov8s.yaml
        if model.startswith("yolov8"):
            return f"{model}.yaml"

        # s -> yolov8s.yaml
        if model in {"n", "s", "m", "l", "x"}:
            return f"yolov8{model}.yaml"

        return model

    def _load_pretrained_weights(
            self,
            torch_model: nn.Module,
            pretrained: str | bool,
            pretrained_type: str | None = None,
            backbone_max_module: int = 8,
    ) -> None:
        if pretrained is True:
            pretrained_path = self._default_pretrained_path(self.model_source)
        else:
            pretrained_path = str(pretrained)

        if not pretrained_path:
            return

        pretrained_path = str(pretrained_path)

        if pretrained_type is None:
            if pretrained_path.endswith("-cls.pt"):
                pretrained_type = "cls_backbone"
            else:
                pretrained_type = "detector"

        if pretrained_type == "cls_backbone":
            self._load_classification_backbone_weights(
                torch_model=torch_model,
                weight_path=pretrained_path,
                backbone_max_module=backbone_max_module,
            )
        elif pretrained_type in {"detector", "detection", "full"}:
            self._load_compatible_weights(
                torch_model=torch_model,
                weight_path=pretrained_path,
                backbone_only=False,
                backbone_max_module=backbone_max_module,
                log_prefix="[YOLOv8Detector] detection pretrained",
            )
        else:
            raise ValueError(f"Unsupported pretrained_type: {pretrained_type}")

    def _load_classification_backbone_weights(
            self,
            torch_model: nn.Module,
            weight_path: str,
            backbone_max_module: int = 8,
    ) -> None:
        """
        Load only YOLOv8 classification backbone weights.

        This matches original YOLO_LwF behavior more closely:
            - detection architecture is built from yolov8{s}.yaml
            - yolov8{s}-cls.pt provides backbone initialization
            - neck/head remain randomly initialized
        """

        self._load_compatible_weights(
            torch_model=torch_model,
            weight_path=weight_path,
            backbone_only=True,
            backbone_max_module=backbone_max_module,
            log_prefix="[YOLOv8Detector] classification backbone pretrained",
        )

    def _pt_to_yaml_name(self, weight_path: str) -> str:
        name = Path(weight_path).name

        mapping = {
            "yolov8n.pt": "yolov8n.yaml",
            "yolov8s.pt": "yolov8s.yaml",
            "yolov8m.pt": "yolov8m.yaml",
            "yolov8l.pt": "yolov8l.yaml",
            "yolov8x.pt": "yolov8x.yaml",
        }

        if name in mapping:
            return mapping[name]

        stem = Path(weight_path).stem
        if stem.startswith("yolov8"):
            return f"{stem}.yaml"

        raise ValueError(
            f"Cannot infer YOLOv8 yaml from weight path: {weight_path}. "
            "Use a standard YOLOv8 weight name such as yolov8n.pt, "
            "or pass a .yaml model config directly."
        )

    # 在 YOLOv8Detector 类中替换 / 新增下面这些方法

    def _get_project_root(self) -> Path:
        """
        Resolve project root without hard-coding /workspace/EvoDet or /workspace/Dvodet.

        Priority:
            1. EVODET_ROOT env var
            2. current file location: <project>/clod_framework/models/yolov8_detector.py
        """
        env_root = os.environ.get("EVODET_ROOT", "").strip()
        if env_root:
            return Path(env_root).expanduser().resolve()

        return Path(__file__).resolve().parents[2]

    def _resolve_weight_path(self, weight_path: str | Path | None) -> str | None:
        """
        Resolve local weight path without hard-coded absolute directories.

        Supports:
            - absolute path
            - relative path from current working directory
            - relative path from project root
            - file name under project_root/weights
            - file name under project_root/checkpoints

        If not found, returns None so Ultralytics YOLO(weight_name) can try auto-download.
        """

        if weight_path is None:
            return None

        weight_path = str(weight_path).strip()

        if not weight_path:
            return None

        # Allow URLs or remote-like paths to pass through.
        if weight_path.startswith(("http://", "https://", "hf://")):
            return weight_path

        project_root = self._get_project_root()

        raw_path = Path(os.path.expandvars(os.path.expanduser(weight_path)))

        if raw_path.is_absolute():
            return str(raw_path.resolve()) if raw_path.exists() else None

        candidates = [
            Path.cwd() / raw_path,
            project_root / raw_path,
            project_root / "weights" / raw_path,
            project_root / "weights" / raw_path.name,
            project_root / "checkpoints" / raw_path,
            project_root / "checkpoints" / raw_path.name,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

        return None

    def _extract_state_dict(self, weight_path: str | Path) -> dict[str, torch.Tensor]:
        """
        Extract state_dict from local or Ultralytics weights.

        No hard-coded project path is used.
        """

        resolved_path = self._resolve_weight_path(weight_path)

        yolo_source = resolved_path if resolved_path is not None else str(weight_path)

        if str(weight_path).endswith(".pt") or str(yolo_source).endswith(".pt"):
            try:
                print(f"[YOLOv8Detector] loading Ultralytics weight via YOLO(): {yolo_source}")

                from ultralytics import YOLO

                yolo = YOLO(yolo_source)
                source = yolo.model

                if hasattr(source, "float"):
                    source = source.float()

                if not hasattr(source, "state_dict"):
                    raise TypeError(f"Ultralytics model has no state_dict(): {type(source)!r}")

                return self._clean_state_dict(source.state_dict())

            except Exception as yolo_exc:
                print(
                    "[YOLOv8Detector] YOLO() failed to load weight. "
                    "Trying raw torch.load() fallback..."
                )

                if resolved_path is None:
                    project_root = self._get_project_root()
                    raise FileNotFoundError(
                        f"Pretrained weight not found locally and Ultralytics failed to load it: {weight_path}\n"
                        f"Tried relative to:\n"
                        f"  cwd: {Path.cwd()}\n"
                        f"  project_root: {project_root}\n"
                        f"  weights_dir: {project_root / 'weights'}\n"
                        f"Set EVODET_ROOT if your project root is elsewhere."
                    ) from yolo_exc

                ckpt = torch.load(resolved_path, map_location="cpu")
                return self._state_dict_from_checkpoint_object(ckpt)

        if resolved_path is None:
            project_root = self._get_project_root()
            raise FileNotFoundError(
                f"Pretrained weight not found: {weight_path}\n"
                f"Tried relative to:\n"
                f"  cwd: {Path.cwd()}\n"
                f"  project_root: {project_root}\n"
                f"  weights_dir: {project_root / 'weights'}\n"
                f"Set EVODET_ROOT if your project root is elsewhere."
            )

        ckpt = torch.load(resolved_path, map_location="cpu")
        return self._state_dict_from_checkpoint_object(ckpt)

    def _state_dict_from_checkpoint_object(self, ckpt: Any) -> dict[str, torch.Tensor]:
        if isinstance(ckpt, dict):
            source = (
                    ckpt.get("model_state_dict", None)
                    or ckpt.get("state_dict", None)
                    or ckpt.get("model", None)
                    or ckpt.get("ema", None)
                    or ckpt
            )
        else:
            source = ckpt

        if hasattr(source, "float"):
            source = source.float()

        if hasattr(source, "state_dict"):
            state = source.state_dict()
        elif isinstance(source, dict):
            state = source
        else:
            raise TypeError(f"Cannot extract state_dict from {type(source)!r}")

        return self._clean_state_dict(state)

    def _clean_state_dict(self, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clean_state: dict[str, torch.Tensor] = {}

        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue

            clean_key = str(key)

            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            if clean_key.startswith("model.model."):
                clean_key = clean_key.replace("model.model.", "model.", 1)

            clean_state[clean_key] = value

        return clean_state

    def _load_compatible_weights(
            self,
            torch_model: nn.Module,
            weight_path: str,
            backbone_only: bool = False,
            backbone_max_module: int = 8,
            log_prefix: str = "[YOLOv8Detector] pretrained",
    ) -> None:
        weight_path = str(weight_path)

        src_state = self._extract_state_dict(weight_path)
        dst_state = torch_model.state_dict()

        compatible = {}
        skipped = []

        for src_key, src_value in src_state.items():
            if not isinstance(src_value, torch.Tensor):
                continue

            candidate_keys = self._candidate_state_keys(src_key)

            matched_key = None
            for key in candidate_keys:
                if key in dst_state:
                    matched_key = key
                    break

            if matched_key is None:
                skipped.append((src_key, "missing_in_target"))
                continue

            if backbone_only and not self._is_backbone_key(matched_key, backbone_max_module):
                skipped.append((src_key, "not_backbone"))
                continue

            if tuple(dst_state[matched_key].shape) != tuple(src_value.shape):
                skipped.append(
                    (
                        src_key,
                        f"shape_mismatch source={tuple(src_value.shape)} target={tuple(dst_state[matched_key].shape)}",
                    )
                )
                continue

            compatible[matched_key] = src_value.detach().to(dtype=dst_state[matched_key].dtype)

        if not compatible:
            print(
                f"{log_prefix}: transferred 0/{len(dst_state)} items from {weight_path}. "
                "No compatible weights found."
            )
            return

        dst_state.update(compatible)
        torch_model.load_state_dict(dst_state, strict=True)

        print(
            f"{log_prefix}: transferred {len(compatible)}/{len(dst_state)} compatible items "
            f"from {weight_path}; skipped {len(skipped)} items."
        )

        if backbone_only:
            print(
                f"{log_prefix}: backbone_only=True, "
                f"loaded modules model.0 through model.{backbone_max_module}; "
                "neck/head kept randomly initialized."
            )

    def _candidate_state_keys(self, key: str) -> list[str]:
        keys = [key]

        if key.startswith("module."):
            keys.append(key[len("module."):])

        if key.startswith("model.model."):
            keys.append(key.replace("model.model.", "model.", 1))

        if not key.startswith("model.") and re.match(r"^\d+\.", key):
            keys.append(f"model.{key}")

        # de-duplicate while preserving order
        out = []
        seen = set()

        for k in keys:
            if k not in seen:
                out.append(k)
                seen.add(k)

        return out

    def _is_backbone_key(
            self,
            key: str,
            backbone_max_module: int,
    ) -> bool:
        """
        Only allow model.0 ... model.{backbone_max_module}.

        For YOLOv8 classification checkpoints, this prevents classification head
        or non-backbone modules from being loaded into detection neck/head.
        """

        match = re.match(r"^model\.(\d+)\.", key)

        if match is None:
            return False

        module_idx = int(match.group(1))
        return module_idx <= int(backbone_max_module)

    def _default_pretrained_path(self, model: str) -> str:
        model = str(model)

        if model in {"n", "s", "m", "l", "x"}:
            return f"yolov8{model}.pt"

        if model.startswith("yolov8") and not model.endswith((".pt", ".yaml", ".yml")):
            return f"{model}.pt"

        if model.endswith((".pt", ".yaml", ".yml")):
            return model

        return model

    def _print_head_debug(self) -> None:
        head = self.get_head(self.model)

        if head is None:
            print("[YOLOv8Detector] head=None")
            return

        nc = getattr(head, "nc", None)
        no = getattr(head, "no", None)
        reg_max = getattr(head, "reg_max", None)

        print(f"[YOLOv8Detector] Detect head: nc={nc}, no={no}, reg_max={reg_max}")

        try:
            if hasattr(head, "cv3"):
                last = head.cv3[-1][-1]
                if hasattr(last, "out_channels"):
                    print(f"[YOLOv8Detector] class head out_channels={last.out_channels}")
        except Exception:
            pass

    def forward(
        self,
        batch: Any,
        return_features: bool = False,
        raw: bool = True,
    ) -> YOLOv8DetectorOutput:
        images = self._extract_images(batch)

        if images is None:
            keys = list(batch.keys()) if isinstance(batch, dict) else type(batch)
            raise ValueError(
                "YOLOv8Detector.forward expected image tensor or batch containing "
                "'img' / 'images' / 'image'. "
                f"Got: {keys}"
            )

        images = images.to(self.device, non_blocking=True)
        preds = self.model(images)

        features = {} if return_features else None
        return YOLOv8DetectorOutput(raw=preds, features=features)

    @torch.no_grad()
    def predict_raw(self, images: torch.Tensor) -> Any:
        was_training = self.training
        self.eval()

        images = images.to(self.device, non_blocking=True)
        preds = self.model(images)

        if isinstance(preds, tuple):
            preds = preds[0]

        if was_training:
            self.train()

        return preds

    def _extract_images(self, batch: Any) -> Optional[torch.Tensor]:
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

    def save_checkpoint(
        self,
        path: Union[str, Path],
        task_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_source": self.model_source,
            "num_classes": self.num_classes,
            "state_dict": self.model.state_dict(),
            "task_id": task_id,
            "model_class": self.__class__.__name__,
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, path)

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

        path = output_dir / filename

        self.save_checkpoint(
            path=path,
            task_id=task_id,
            extra=extra,
        )

        return path

    def load_checkpoint(
            self,
            checkpoint_path: str | Path,
            strict: bool = False,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

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

        clean_state = {}

        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue

            clean_key = str(key)

            if clean_key.startswith("module."):
                clean_key = clean_key[len("module."):]

            if clean_key.startswith("model.model."):
                clean_key = clean_key.replace("model.model.", "model.", 1)

            clean_state[clean_key] = value

        missing, unexpected = self.model.load_state_dict(clean_state, strict=strict)

        print(
            f"[YOLOv8Detector] checkpoint loaded from {checkpoint_path}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, strict={strict}"
        )

    @classmethod
    def load_teacher(
            cls,
            checkpoint_path: str | Path,
            model: str = "yolov8s",
            num_classes: int = 80,
            device: str | torch.device = "cuda",
            strict_load: bool = False,
            **kwargs,
    ) -> "YOLOv8Detector":
        teacher = cls(
            model=model,
            num_classes=num_classes,
            pretrained=False,
            device=device,
            strict_load=strict_load,
            **kwargs,
        )

        teacher.load_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=strict_load,
        )

        teacher.eval()
        teacher.requires_grad_(False)

        print(f"[YOLOv8Detector] loaded teacher checkpoint from {checkpoint_path}")

        return teacher


    def freeze_backbone(self) -> None:
        backbone = self.backbone

        if backbone is None:
            return

        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        backbone = self.backbone

        if backbone is None:
            return

        for param in backbone.parameters():
            param.requires_grad = True

    @property
    def backbone(self) -> Optional[nn.Module]:
        return self.get_backbone(self.model)

    @property
    def neck(self) -> Optional[nn.Module]:
        return self.get_neck(self.model)

    @property
    def head(self) -> Optional[nn.Module]:
        return self.get_head(self.model)

    @staticmethod
    def get_backbone(model: nn.Module) -> Optional[nn.Module]:
        modules = getattr(model, "model", None)

        if not isinstance(modules, nn.Sequential):
            return None

        if len(modules) < 3:
            return modules

        return nn.Sequential(*list(modules.children())[:-2])

    @staticmethod
    def get_neck(model: nn.Module) -> Optional[nn.Module]:
        modules = getattr(model, "model", None)

        if not isinstance(modules, nn.Sequential):
            return None

        if len(modules) < 3:
            return None

        return nn.Sequential(*list(modules.children())[-2:-1])

    @staticmethod
    def get_head(model: nn.Module) -> Optional[nn.Module]:
        modules = getattr(model, "model", None)

        if isinstance(modules, nn.Sequential) and len(modules) > 0:
            return modules[-1]

        if hasattr(model, "head"):
            return model.head

        return None

    def train(self, mode: bool = True) -> "YOLOv8Detector":
        self.training = mode

        if hasattr(self, "model") and self.model is not None:
            self.model.train(mode)

        return self

    def eval(self) -> "YOLOv8Detector":
        return self.train(False)

    def to(self, *args: Any, **kwargs: Any) -> "YOLOv8Detector":
        if hasattr(self, "model") and self.model is not None:
            self.model.to(*args, **kwargs)

        if args:
            candidate = args[0]
            if isinstance(candidate, (str, torch.device)):
                self.device = torch.device(candidate)

        if "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])

        return self