# clod_framework/losses/yolo_lwf_replay_loss.py

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors


@dataclass
class YOLOLwFReplayLossOutput:
    loss: torch.Tensor
    loss_items: dict[str, torch.Tensor]

    def as_log_dict(self) -> dict[str, torch.Tensor]:
        out = {"loss": self.loss}
        out.update(self.loss_items)
        return out


class YOLOv8OutputLwFLoss(nn.Module):
    """
    Original-style YOLOv8 LwF output loss.

    Supports both:
        old_classes=[...]
        classes=[...]

    Distills:
        - old-class classification logits with BCEWithLogits
        - IoU^6 weighted classification distillation
        - distribution-level regression distillation
        - optional DFL distillation
    """

    def __init__(
        self,
        c1: float,
        c2: float,
        old_classes: Optional[Sequence[int]] = None,
        reg_max: int = 16,
        device: torch.device | str = "cuda",
        c3: Optional[float] = None,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = None if c3 is None else float(c3)

        if old_classes is None:
            old_classes = classes

        self.old_classes = [int(c) for c in (old_classes or [])]
        self.reg_max = int(reg_max)
        self.device = torch.device(device)

        self.log_softmax = nn.LogSoftmax(dim=3)
        self.softmax = nn.Softmax(dim=3)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def forward(
        self,
        student_cls_logits: Optional[torch.Tensor] = None,
        student_reg_logits: Optional[torch.Tensor] = None,
        teacher_output: Any = None,
        anchor_points: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Compatible with original YOLO_LwF naming.
        if student_cls_logits is None:
            student_cls_logits = kwargs.get("student_cl_output", None)

        if student_reg_logits is None:
            student_reg_logits = kwargs.get("student_reg_output", None)

        if teacher_output is None:
            teacher_output = kwargs.get("teacher_output", None)

        if anchor_points is None:
            anchor_points = kwargs.get("anchor_points", None)

        if (
            student_cls_logits is None
            or student_reg_logits is None
            or teacher_output is None
            or anchor_points is None
        ):
            raise ValueError(
                "YOLOv8OutputLwFLoss.forward requires student_cls_logits, "
                "student_reg_logits, teacher_output, and anchor_points."
            )

        if not self.old_classes:
            return student_cls_logits.sum() * 0.0

        batch_size = student_cls_logits.shape[0]
        num_preds = student_cls_logits.shape[1]
        nc = student_cls_logits.shape[-1]
        reg_max = student_reg_logits.shape[-1] // 4
        no = reg_max * 4 + nc

        old_classes = [c for c in self.old_classes if 0 <= c < nc]
        if not old_classes:
            return student_cls_logits.sum() * 0.0

        if isinstance(teacher_output, tuple):
            teacher_output = teacher_output[1]

        if hasattr(teacher_output, "raw"):
            teacher_output = teacher_output.raw

        if isinstance(teacher_output, dict):
            for key in ("raw", "preds", "pred", "outputs"):
                if key in teacher_output:
                    teacher_output = teacher_output[key]
                    break

        if isinstance(teacher_output, list):
            target_distri, target_logit_scores = torch.cat(
                [xi.view(batch_size, no, -1) for xi in teacher_output],
                dim=2,
            ).split((reg_max * 4, nc), dim=1)

        elif isinstance(teacher_output, torch.Tensor):
            if teacher_output.ndim == 3 and teacher_output.shape[1] == no:
                target_distri, target_logit_scores = teacher_output.split((reg_max * 4, nc), dim=1)
            elif teacher_output.ndim == 3 and teacher_output.shape[-1] == no:
                target = teacher_output.permute(0, 2, 1).contiguous()
                target_distri, target_logit_scores = target.split((reg_max * 4, nc), dim=1)
            else:
                return student_cls_logits.sum() * 0.0
        else:
            return student_cls_logits.sum() * 0.0

        target_logit_scores = target_logit_scores.permute(0, 2, 1).contiguous()
        target_distri = target_distri.permute(0, 2, 1).contiguous()

        if target_logit_scores.shape[1] != num_preds:
            min_preds = min(target_logit_scores.shape[1], num_preds)
            student_cls_logits = student_cls_logits[:, :min_preds]
            student_reg_logits = student_reg_logits[:, :min_preds]
            target_logit_scores = target_logit_scores[:, :min_preds]
            target_distri = target_distri[:, :min_preds]
            anchor_points = anchor_points[:min_preds]
            num_preds = min_preds

        old_idx = torch.tensor(
            old_classes,
            device=student_cls_logits.device,
            dtype=torch.long,
        )

        iou_scores = self.score_iou(
            pred_distri=student_reg_logits.detach(),
            target_distri=target_distri.detach(),
            anchors=anchor_points,
        )

        iou_scores = torch.pow(
            iou_scores.repeat(1, 1, len(old_classes)),
            6,
        )

        target_distri_view = target_distri.view(batch_size, num_preds, 4, reg_max)
        pred_distri_view = student_reg_logits.view(batch_size, num_preds, 4, reg_max)

        target_scores = self.sigmoid(target_logit_scores)

        lwf_cls_loss = iou_scores * self.bce(
            student_cls_logits[:, :, old_idx],
            target_scores[:, :, old_idx].detach(),
        )
        lwf_cls_loss = torch.mean(lwf_cls_loss)

        weights, _ = torch.max(target_scores[:, :, old_idx], dim=2)
        weights = weights.unsqueeze(2).repeat(1, 1, 4)

        target_distribution = self.softmax(target_distri_view.detach())
        log_pred_distribution = self.log_softmax(pred_distri_view)

        ces = torch.sum(-target_distribution * log_pred_distribution, dim=3)
        weighted_ces = weights * ces
        lwf_reg_loss = torch.mean(weighted_ces)

        lwf_loss = self.c1 * lwf_cls_loss + self.c2 * lwf_reg_loss

        if self.c3 is not None:
            target = target_distribution.matmul(self.proj.type(target_distribution.dtype))
            dfl_loss = torch.mean(self._df_loss(log_pred_distribution, target) * weights)
            lwf_loss = lwf_loss + self.c3 * dfl_loss

        return lwf_loss

    def score_iou(
        self,
        pred_distri: torch.Tensor,
        target_distri: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        target = torch.unsqueeze(
            self.bbox_decode(anchors, target_distri),
            dim=-2,
        )
        pred = torch.unsqueeze(
            self.bbox_decode(anchors, pred_distri),
            dim=-2,
        )

        scores = (bbox_iou(pred, target, xywh=False, DIoU=True) + 1.0) / 2.0
        return torch.squeeze(scores, dim=-1)

    def bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
    ) -> torch.Tensor:
        b, a, c = pred_dist.shape

        pred_dist = (
            pred_dist.view(b, a, 4, c // 4)
            .softmax(3)
            .matmul(self.proj.type(pred_dist.dtype))
        )

        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _df_loss(
        self,
        log_pred_dist: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        tl = target.long().clamp(min=0, max=self.reg_max - 1)
        tr = (tl + 1).clamp(max=self.reg_max - 1)

        wl = tr.float() - target
        wr = 1.0 - wl

        tl_mask = F.one_hot(tl, self.reg_max).float()
        tr_mask = F.one_hot(tr, self.reg_max).float()

        left_term = -torch.sum(log_pred_dist * tl_mask, dim=-1) * wl
        right_term = -torch.sum(log_pred_dist * tr_mask, dim=-1) * wr

        return left_term + right_term

class OriginalStyleYOLOLwFReplayLoss(nn.Module):
    """
    Original YOLO_LwF+OCDM style loss for current EvoDet training loop.

    对齐点：
        - batch 前半 current，后半 replay
        - current labels 和 replay labels 分开监督
        - current cls loss 只看当前 task classes
        - replay cls loss 支持 per-sample task_id mask
        - LwF loss 使用 old classes，且可对所有 images 或仅 replay images 蒸馏
    """

    def __init__(
            self,
            model: Any,
            device: str | torch.device,
            c1: float = 1.0,
            c2: float = 1.0,
            c3: Optional[float] = None,
            use_new_images_for_lwf: bool = True,
            use_replay_labels: bool = True,
            box_gain: float = 7.5,
            cls_gain: float = 0.5,
            dfl_gain: float = 1.5,
            lwf_gain: float = 1.0,
            debug_loss_scaling: bool = False,
    ) -> None:
        super().__init__()

        self.model = self._unwrap_model(model)
        self.device = torch.device(device)

        self._ensure_args(self.model, box_gain, cls_gain, dfl_gain)

        head = self._get_head(self.model)
        if head is None:
            raise RuntimeError("Could not find YOLOv8 Detect head.")

        self.hyp = self.model.args
        self.stride = head.stride
        self.nc = int(head.nc)
        self.no = int(head.no)
        self.reg_max = int(head.reg_max)
        self.use_dfl = self.reg_max > 1
        self.device = next(self.model.parameters()).device

        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = c3
        self.use_new_images_for_lwf = bool(use_new_images_for_lwf)
        self.use_replay_labels = bool(use_replay_labels)

        # Original YOLO_LwF style scaling:
        # total = current_loss * (2 - lwf_gain) + lwf_gain * (replay_loss + lwf_loss)
        self.lwf_gain = float(lwf_gain)
        self.debug_loss_scaling = bool(debug_loss_scaling)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
        )
        self.bbox_loss = BboxLoss(
            self.reg_max - 1,
            use_dfl=self.use_dfl,
        ).to(self.device)
        self.proj = torch.arange(
            self.reg_max,
            dtype=torch.float,
            device=self.device,
        )

    def forward(
        self,
        student_outputs: Any,
        batch: Mapping[str, Any],
        teacher_outputs: Optional[Any] = None,
        current_classes: Optional[Sequence[int]] = None,
        old_classes: Optional[Sequence[int]] = None,
        classes_per_task: Optional[Sequence[Sequence[int]]] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | YOLOLwFReplayLossOutput:
        current_classes = [int(c) for c in (current_classes or []) if 0 <= int(c) < self.nc]
        old_classes = [int(c) for c in (old_classes or []) if 0 <= int(c) < self.nc]
        classes_per_task = classes_per_task or []

        feats = self._unwrap_outputs(student_outputs)
        if isinstance(feats, tuple):
            feats = feats[1]

        if isinstance(feats, torch.Tensor):
            feats = [feats]

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats],
            dim=2,
        ).split((self.reg_max * 4, self.nc), dim=1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        num_current = int(batch.get("num_current", batch_size))
        num_replay = int(batch.get("num_replay", max(0, batch_size - num_current)))

        current_targets = self._targets_from_batch(batch, start=0, end=num_current)
        current_targets = self.preprocess(
            current_targets.to(self.device),
            num_current,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )

        current_loss, current_items = self._detection_loss_from_processed(
            pred_distri=pred_distri[:num_current],
            pred_scores=pred_scores[:num_current],
            targets=current_targets,
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
            allowed_classes=current_classes,
        )

        replay_loss = pred_scores.sum() * 0.0
        replay_items = torch.zeros(3, device=self.device)

        if self.use_replay_labels and num_replay > 0:
            replay_targets = self._targets_from_batch(
                batch,
                start=num_current,
                end=num_current + num_replay,
            )
            replay_targets = self.preprocess(
                replay_targets.to(self.device),
                num_replay,
                scale_tensor=imgsz[[1, 0, 1, 0]],
            )

            replay_task_ids = None
            if "task_id" in batch:
                replay_task_ids = batch["task_id"][num_current : num_current + num_replay].detach().cpu().tolist()
            elif "replay_task_id" in batch:
                replay_task_ids = batch["replay_task_id"][num_current : num_current + num_replay].detach().cpu().tolist()

            replay_loss, replay_items = self._replay_detection_loss(
                pred_distri=pred_distri[num_current : num_current + num_replay],
                pred_scores=pred_scores[num_current : num_current + num_replay],
                targets=replay_targets,
                anchor_points=anchor_points,
                stride_tensor=stride_tensor,
                replay_task_ids=replay_task_ids,
                classes_per_task=classes_per_task,
                fallback_classes=old_classes,
            )

        lwf_loss = pred_scores.sum() * 0.0

        if teacher_outputs is not None and old_classes:
            if self.use_new_images_for_lwf:
                pred_scores_lwf = pred_scores
                pred_distri_lwf = pred_distri
                teacher_lwf = self._unwrap_outputs(teacher_outputs)
                batch_size_lwf = batch_size
            else:
                pred_scores_lwf = pred_scores[num_current:]
                pred_distri_lwf = pred_distri[num_current:]
                teacher_lwf = self._slice_teacher_outputs(
                    self._unwrap_outputs(teacher_outputs),
                    start=num_current,
                    end=batch_size,
                )
                batch_size_lwf = max(0, batch_size - num_current)

            lwf_module = YOLOv8OutputLwFLoss(
                c1=self.c1,
                c2=self.c2,
                old_classes=old_classes,
                reg_max=self.reg_max,
                device=self.device,
                c3=self.c3,
            )

            lwf_loss = lwf_module(
                student_cls_logits=pred_scores_lwf,
                student_reg_logits=pred_distri_lwf,
                teacher_output=teacher_lwf,
                anchor_points=anchor_points,
            ) * max(1, batch_size_lwf)

        continual_loss = replay_loss + lwf_loss
        total = current_loss * (2.0 - self.lwf_gain) + self.lwf_gain * continual_loss

        loss_items = {
            # Main components.
            "current_loss": current_loss.detach(),
            "replay_loss": replay_loss.detach(),
            "lwf_loss": lwf_loss.detach(),
            "continual_loss": continual_loss.detach(),
            "total_loss": total.detach(),

            # Raw scale metadata.
            "lwf_gain": torch.tensor(float(self.lwf_gain), device=self.device),
            "current_scale": torch.tensor(float(2.0 - self.lwf_gain), device=self.device),
            "continual_scale": torch.tensor(float(self.lwf_gain), device=self.device),

            # Detection sub-items.
            "current_box_loss": current_items[0].detach(),
            "current_cls_loss": current_items[1].detach(),
            "current_dfl_loss": current_items[2].detach(),
            "replay_box_loss": replay_items[0].detach(),
            "replay_cls_loss": replay_items[1].detach(),
            "replay_dfl_loss": replay_items[2].detach(),

            # Batch metadata for sanity check.
            "num_current": torch.tensor(float(num_current), device=self.device),
            "num_replay": torch.tensor(float(num_replay), device=self.device),
            "num_labels": torch.tensor(float(batch.get("num_labels", -1)), device=self.device),
        }

        if return_dict:
            return YOLOLwFReplayLossOutput(loss=total, loss_items=loss_items)

        if self.debug_loss_scaling:
            print(
                "[LossScaling] "
                f"current={float(current_loss.detach().cpu()):.6f}, "
                f"replay={float(replay_loss.detach().cpu()):.6f}, "
                f"lwf={float(lwf_loss.detach().cpu()):.6f}, "
                f"continual={float(continual_loss.detach().cpu()):.6f}, "
                f"total={float(total.detach().cpu()):.6f}, "
                f"lwf_gain={self.lwf_gain:.3f}, "
                f"current_scale={2.0 - self.lwf_gain:.3f}, "
                f"continual_scale={self.lwf_gain:.3f}, "
                f"num_current={num_current}, "
                f"num_replay={num_replay}, "
                f"num_labels={batch.get('num_labels', -1)}"
            )

        return total

    def _detection_loss_from_processed(
        self,
        pred_distri: torch.Tensor,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        allowed_classes: Optional[Sequence[int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype

        loss = torch.zeros(3, device=self.device)

        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        cls_loss = self.bce(pred_scores, target_scores.to(dtype))

        if allowed_classes:
            mask = self._class_mask(allowed_classes, pred_scores.device, cls_loss.dtype)
            cls_loss = cls_loss * mask.view(1, 1, -1)

        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss

    def _replay_detection_loss(
        self,
        pred_distri: torch.Tensor,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        replay_task_ids: Optional[Sequence[int]],
        classes_per_task: Sequence[Sequence[int]],
        fallback_classes: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype

        loss = torch.zeros(3, device=self.device)

        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        cls_loss = self.bce(pred_scores, target_scores.to(dtype))

        if replay_task_ids is not None and classes_per_task:
            mask = self._task_mask(
                task_ids=replay_task_ids,
                num_preds=pred_scores.shape[1],
                classes_per_task=classes_per_task,
                device=pred_scores.device,
                dtype=cls_loss.dtype,
            )
            cls_loss = cls_loss * mask
        elif fallback_classes:
            mask = self._class_mask(fallback_classes, pred_scores.device, cls_loss.dtype)
            cls_loss = cls_loss * mask.view(1, 1, -1)

        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss

    def preprocess(
        self,
        targets: torch.Tensor,
        batch_size: int,
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)

        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]

        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(
        self,
        anchor_points: torch.Tensor,
        pred_dist: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _targets_from_batch(
        self,
        batch: Mapping[str, Any],
        start: int,
        end: int,
    ) -> torch.Tensor:
        targets = batch["targets"]

        if targets.numel() == 0:
            return targets.new_zeros((0, 6))

        mask = (targets[:, 0] >= start) & (targets[:, 0] < end)
        sliced = targets[mask].clone()

        if sliced.numel() > 0:
            sliced[:, 0] -= start

        return sliced.to(self.device)

    def _task_mask(
        self,
        task_ids: Sequence[int],
        num_preds: int,
        classes_per_task: Sequence[Sequence[int]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros((len(task_ids), num_preds, self.nc), device=device, dtype=dtype)

        for i, task_id in enumerate(task_ids):
            task_id = int(task_id)
            if task_id < 0 or task_id >= len(classes_per_task):
                allowed = []
            else:
                allowed = [int(c) for c in classes_per_task[task_id] if 0 <= int(c) < self.nc]

            if allowed:
                idx = torch.tensor(allowed, device=device, dtype=torch.long)
                mask[i, :, idx] = 1.0

        return mask

    def _class_mask(
        self,
        classes: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(self.nc, device=device, dtype=dtype)
        valid = [int(c) for c in classes if 0 <= int(c) < self.nc]
        if valid:
            mask[torch.tensor(valid, device=device, dtype=torch.long)] = 1.0
        return mask

    def _slice_teacher_outputs(self, outputs: Any, start: int, end: int) -> Any:
        outputs = self._unwrap_outputs(outputs)

        if isinstance(outputs, torch.Tensor):
            return outputs[start:end]

        if isinstance(outputs, list):
            return [self._slice_teacher_outputs(x, start, end) for x in outputs]

        if isinstance(outputs, tuple):
            return tuple(self._slice_teacher_outputs(x, start, end) for x in outputs)

        return outputs

    def _unwrap_model(self, model: Any) -> Any:
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            return model.model
        return model

    def _unwrap_outputs(self, outputs: Any) -> Any:
        if hasattr(outputs, "raw"):
            return outputs.raw

        if isinstance(outputs, Mapping):
            for key in ("raw", "preds", "pred", "outputs"):
                if key in outputs:
                    return outputs[key]

        return outputs

    def _get_head(self, model: Any) -> Any:
        modules = getattr(model, "model", None)

        if isinstance(modules, nn.Sequential) and len(modules) > 0:
            return modules[-1]

        if isinstance(modules, (list, tuple)) and len(modules) > 0:
            return modules[-1]

        if hasattr(model, "head"):
            return model.head

        return None

    def _ensure_args(
        self,
        model: Any,
        box: float,
        cls: float,
        dfl: float,
    ) -> None:
        if not hasattr(model, "args") or model.args is None:
            model.args = SimpleNamespace(box=box, cls=cls, dfl=dfl)
        else:
            if not hasattr(model.args, "box"):
                model.args.box = box
            if not hasattr(model.args, "cls"):
                model.args.cls = cls
            if not hasattr(model.args, "dfl"):
                model.args.dfl = dfl


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y