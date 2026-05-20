# clod_framework/models/builder.py

from __future__ import annotations

from typing import Any

import torch

from clod_framework.models.yolov8_detector import YOLOv8Detector


def build_model(
    cfg: dict[str, Any],
) -> YOLOv8Detector:
    """
    Build detection model from EvoDet config.

    Expected config:

        model:
          name: yolov8
          variant: yolov8s
          pretrained: yolov8s-cls.pt
          pretrained_type: cls_backbone
          backbone_pretrained_max_module: 8
          num_classes: 80

    Important behavior:

        variant:
            Controls the detection architecture.
            Examples:
                yolov8n
                yolov8s
                yolov8m
                yolov8l
                yolov8x
                yolov8s.yaml

        pretrained:
            Controls initialization weights.
            Examples:
                yolov8s.pt          -> detector pretrained
                yolov8s-cls.pt      -> classification backbone pretrained
                null / "" / false   -> random init

        pretrained_type:
            cls_backbone:
                Load only backbone-compatible weights from yolov8*-cls.pt.
                Neck/head remain randomly initialized.

            detector:
                Load all shape-compatible detector weights.

            null:
                Auto infer from pretrained filename:
                    *-cls.pt -> cls_backbone
                    *.pt     -> detector
    """

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    model_name = str(model_cfg.get("name", "yolov8")).lower()

    if model_name not in {"yolov8", "yolo", "ultralytics_yolov8"}:
        raise ValueError(f"Unsupported model name: {model_name}")

    num_classes = int(
        model_cfg.get(
            "num_classes",
            dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80)),
        )
    )

    variant = str(model_cfg.get("variant", "yolov8s"))

    pretrained = model_cfg.get("pretrained", None)
    pretrained = normalize_pretrained_value(pretrained)

    pretrained_type = model_cfg.get("pretrained_type", None)

    backbone_pretrained_max_module = int(
        model_cfg.get("backbone_pretrained_max_module", 8)
    )

    device = training_cfg.get(
        "device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )

    strict_load = bool(model_cfg.get("strict_load", False))

    model = YOLOv8Detector(
        model=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_type=pretrained_type,
        backbone_pretrained_max_module=backbone_pretrained_max_module,
        device=device,
        strict_load=strict_load,
    )

    return model


def normalize_pretrained_value(value: Any) -> str | bool | None:
    """
    Normalize YAML values for pretrained.

    Accepts:
        null
        false
        ""
        "none"
        "None"
        "false"
        "False"
        true
        "yolov8s.pt"
        "yolov8s-cls.pt"
    """

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    value = str(value).strip()

    if value in {"", "none", "None", "null", "Null", "false", "False"}:
        return None

    if value in {"true", "True"}:
        return True

    return value


def resolve_model_source_for_teacher(
    cfg: dict[str, Any],
) -> str:
    """
    Resolve architecture source for teacher model.

    Teacher should be built from detection architecture, not from pretrained.

    Correct:
        variant: yolov8s
        pretrained: yolov8s-cls.pt

        teacher model_source -> yolov8s

    Wrong:
        teacher model_source -> yolov8s-cls.pt

    This function prevents the wrong behavior.
    """

    model_cfg = cfg.get("model", {})
    variant = str(model_cfg.get("variant", "yolov8s"))

    if variant.endswith((".yaml", ".yml")):
        return variant

    if variant.endswith(".pt"):
        return variant.replace(".pt", ".yaml")

    if variant.startswith("yolov8"):
        return variant

    if variant in {"n", "s", "m", "l", "x"}:
        return f"yolov8{variant}"

    return variant


def get_model_num_classes(
    cfg: dict[str, Any],
) -> int:
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    return int(
        model_cfg.get(
            "num_classes",
            dataset_cfg.get("num_classes", dataset_cfg.get("nc", 80)),
        )
    )