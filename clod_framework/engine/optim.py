# clod_framework/engine/optim.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def unwrap_torch_model(model: Any) -> nn.Module:
    """
    Supports:
        - YOLOv8Detector wrapper: model.model
        - raw torch nn.Module
    """

    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model

    if isinstance(model, nn.Module):
        return model

    raise TypeError(f"Unsupported model type: {type(model)!r}")


def build_yolov8_optimizer(
    model: Any,
    optimizer_name: str = "SGD",
    lr: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    batch_size: int = 16,
    accumulate: int = 1,
    nbs: int = 64,
) -> torch.optim.Optimizer:
    """
    YOLOv8-style optimizer parameter groups.

    Parameter groups:
        group 0: weights with decay
        group 1: normalization weights without decay
        group 2: biases without decay

    This avoids applying weight decay to BN/Norm weights and biases.
    """

    torch_model = unwrap_torch_model(model)

    decay_params = []
    norm_params = []
    bias_params = []

    norm_classes = tuple(
        cls
        for name, cls in nn.__dict__.items()
        if "Norm" in name and isinstance(cls, type)
    )

    for module_name, module in torch_model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name == "bias":
                bias_params.append(param)
            elif isinstance(module, norm_classes):
                norm_params.append(param)
            elif param.ndim == 1:
                # Safer no-decay rule for scale-like 1D parameters.
                norm_params.append(param)
            else:
                decay_params.append(param)

    # YOLO-style scaled weight decay.
    # Ultralytics commonly scales weight_decay by batch_size * accumulate / nbs.
    scaled_weight_decay = weight_decay * batch_size * accumulate / nbs

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": scaled_weight_decay,
            "group_name": "decay",
        },
        {
            "params": norm_params,
            "weight_decay": 0.0,
            "group_name": "norm_no_decay",
        },
        {
            "params": bias_params,
            "weight_decay": 0.0,
            "group_name": "bias_no_decay",
        },
    ]

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=True,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(momentum, 0.999),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(momentum, 0.999),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    for group in optimizer.param_groups:
        group["initial_lr"] = lr

    print(
        "[Optimizer] "
        f"{optimizer.__class__.__name__}: "
        f"decay={len(decay_params)}, "
        f"norm_no_decay={len(norm_params)}, "
        f"bias_no_decay={len(bias_params)}, "
        f"weight_decay={scaled_weight_decay:.6g}"
    )

    return optimizer