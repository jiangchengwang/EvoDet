
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import dist2bbox

from clod_framework.losses.detection_loss import DetectionLoss


@dataclass
class ConfMixOutput:
    loss: torch.Tensor
    source_loss: torch.Tensor
    mix_loss: torch.Tensor
    num_pseudo: torch.Tensor
    gamma: torch.Tensor
    delta: torch.Tensor


class YOLOv8DFLUncertaintyPostprocessor:
    """
    YOLOv8 DFL uncertainty postprocessor.

    It does not modify YOLOv8 head.

    It reads YOLOv8 raw detection feature maps:
        [B, reg_max * 4 + nc, H, W]

    Then computes:
        det_conf       = max class probability
        bbox_entropy   = normalized entropy of DFL distributions
        bbox_certainty = 1 - bbox_entropy
        combined_conf  = det_conf * bbox_certainty
        score          = (1 - delta) * det_conf + delta * combined_conf

    Returned det format:
        [x1, y1, x2, y2, score, cls, det_conf, uncertainty, combined_conf, certainty]
    """

    def __init__(
        self,
        torch_model: nn.Module,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 300,
        uncertainty_power: float = 1.0,
        eps: float = 1e-9,
    ) -> None:
        self.torch_model = torch_model
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.max_det = int(max_det)
        self.uncertainty_power = float(uncertainty_power)
        self.eps = float(eps)

    @torch.no_grad()
    def __call__(
        self,
        images: torch.Tensor,
        delta: float,
    ) -> list[torch.Tensor]:
        preds, feats = self._forward_with_feats(images)

        if feats is None:
            # Fallback: no DFL uncertainty available.
            return self._fallback_from_decoded_preds(preds, delta=delta)

        return self._postprocess_feats(
            feats=feats,
            image_h=int(images.shape[-2]),
            image_w=int(images.shape[-1]),
            delta=float(delta),
        )

    @torch.no_grad()
    def _forward_with_feats(
        self,
        images: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[list[torch.Tensor]]]:
        was_training = bool(self.torch_model.training)
        self.torch_model.eval()

        try:
            out = self.torch_model(images)

            preds = None
            feats = None

            # Ultralytics YOLOv8 eval usually returns:
            #   (decoded_predictions, raw_feature_maps)
            if isinstance(out, tuple):
                if len(out) >= 1 and isinstance(out[0], torch.Tensor):
                    preds = out[0]
                if len(out) >= 2 and isinstance(out[1], list):
                    feats = out[1]

            elif isinstance(out, list):
                # Some versions may return raw feats directly.
                if len(out) > 0 and isinstance(out[0], torch.Tensor) and out[0].ndim == 4:
                    feats = out

            elif isinstance(out, torch.Tensor):
                preds = out

            return preds, feats

        finally:
            if was_training:
                self.torch_model.train()

    def _get_detect_head(self) -> Any:
        if hasattr(self.torch_model, "model"):
            return self.torch_model.model[-1]

        raise RuntimeError("Cannot locate YOLOv8 Detect head from torch_model.")

    def _postprocess_feats(
        self,
        feats: list[torch.Tensor],
        image_h: int,
        image_w: int,
        delta: float,
    ) -> list[torch.Tensor]:
        head = self._get_detect_head()

        nc = int(getattr(head, "nc"))
        reg_max = int(getattr(head, "reg_max", 16))
        no = int(getattr(head, "no", reg_max * 4 + nc))

        batch_size = feats[0].shape[0]
        device = feats[0].device
        dtype = feats[0].dtype

        pred = torch.cat(
            [x.view(batch_size, no, -1) for x in feats],
            dim=2,
        )

        pred_distri, pred_scores = pred.split((reg_max * 4, nc), dim=1)

        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = self._make_anchors(feats, head)

        boxes = self._decode_boxes(
            pred_distri=pred_distri,
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
            reg_max=reg_max,
        )

        boxes[..., [0, 2]].clamp_(0.0, float(image_w))
        boxes[..., [1, 3]].clamp_(0.0, float(image_h))

        cls_probs = pred_scores.sigmoid()
        det_conf, cls = cls_probs.max(dim=-1)

        uncertainty, certainty = self._dfl_uncertainty(
            pred_distri=pred_distri,
            reg_max=reg_max,
        )

        certainty = certainty.clamp(0.0, 1.0)
        if self.uncertainty_power != 1.0:
            certainty = certainty.pow(self.uncertainty_power)

        combined_conf = det_conf * certainty
        score = (1.0 - delta) * det_conf + delta * combined_conf

        outputs: list[torch.Tensor] = []

        for b in range(batch_size):
            keep = score[b] >= self.conf_thres

            if not bool(keep.any()):
                outputs.append(
                    torch.zeros((0, 10), device=device, dtype=dtype)
                )
                continue

            det = torch.cat(
                [
                    boxes[b, keep],
                    score[b, keep].unsqueeze(1),
                    cls[b, keep].float().unsqueeze(1),
                    det_conf[b, keep].unsqueeze(1),
                    uncertainty[b, keep].unsqueeze(1),
                    combined_conf[b, keep].unsqueeze(1),
                    certainty[b, keep].unsqueeze(1),
                ],
                dim=1,
            )

            det = self._class_aware_nms(det)
            outputs.append(det)

        return outputs

    def _decode_boxes(
        self,
        pred_distri: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        reg_max: int,
    ) -> torch.Tensor:
        b, a, _ = pred_distri.shape

        proj = torch.arange(
            reg_max,
            dtype=pred_distri.dtype,
            device=pred_distri.device,
        )

        dist = (
            pred_distri.view(b, a, 4, reg_max)
            .softmax(dim=3)
            .matmul(proj)
        )

        boxes = dist2bbox(
            dist,
            anchor_points,
            xywh=False,
        )

        boxes = boxes * stride_tensor.view(1, -1, 1)

        return boxes

    def _dfl_uncertainty(
        self,
        pred_distri: torch.Tensor,
        reg_max: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, a, _ = pred_distri.shape

        if reg_max <= 1:
            uncertainty = torch.zeros((b, a), device=pred_distri.device, dtype=pred_distri.dtype)
            certainty = torch.ones((b, a), device=pred_distri.device, dtype=pred_distri.dtype)
            return uncertainty, certainty

        prob = pred_distri.view(b, a, 4, reg_max).softmax(dim=-1)

        entropy = -(prob * (prob + self.eps).log()).sum(dim=-1)
        entropy = entropy / torch.log(
            torch.tensor(float(reg_max), device=prob.device, dtype=prob.dtype)
        )

        uncertainty = entropy.mean(dim=-1).clamp(0.0, 1.0)
        certainty = 1.0 - uncertainty

        return uncertainty, certainty

    def _make_anchors(
        self,
        feats: list[torch.Tensor],
        head: Any,
        grid_cell_offset: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = feats[0].device
        dtype = feats[0].dtype

        stride = getattr(head, "stride", None)
        if stride is None:
            raise RuntimeError("YOLOv8 Detect head has no stride attribute.")

        if isinstance(stride, torch.Tensor):
            stride_values = stride.detach().cpu().tolist()
        else:
            stride_values = list(stride)

        anchor_points = []
        stride_tensor = []

        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape

            sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset

            try:
                yy, xx = torch.meshgrid(sy, sx, indexing="ij")
            except TypeError:
                yy, xx = torch.meshgrid(sy, sx)

            points = torch.stack((xx, yy), dim=-1).view(-1, 2)

            s = float(stride_values[i])
            st = torch.full(
                (h * w, 1),
                s,
                device=device,
                dtype=dtype,
            )

            anchor_points.append(points)
            stride_tensor.append(st)

        return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)

    def _class_aware_nms(
        self,
        det: torch.Tensor,
    ) -> torch.Tensor:
        if det.numel() == 0:
            return det

        keep_all = []

        classes = det[:, 5].unique()

        for c in classes:
            idx = torch.where(det[:, 5] == c)[0]
            boxes = det[idx, :4]
            scores = det[idx, 4]

            order = scores.argsort(descending=True)

            while order.numel() > 0:
                current = order[0]
                keep_all.append(idx[current])

                if order.numel() == 1:
                    break

                ious = box_iou(
                    boxes[current].view(1, 4),
                    boxes[order[1:]],
                ).view(-1)

                order = order[1:][ious <= self.iou_thres]

        if not keep_all:
            return det[:0]

        keep = torch.stack(keep_all)
        kept = det[keep]

        kept = kept[kept[:, 4].argsort(descending=True)]

        if kept.shape[0] > self.max_det:
            kept = kept[: self.max_det]

        return kept

    def _fallback_from_decoded_preds(
        self,
        preds: Optional[torch.Tensor],
        delta: float,
    ) -> list[torch.Tensor]:
        """
        Fallback for versions that do not return raw feats.

        This has no DFL uncertainty. It keeps code runnable but does not provide
        the full ConfMix uncertainty behavior.
        """

        if preds is None:
            raise RuntimeError(
                "YOLOv8 forward did not return raw feats or decoded predictions. "
                "Cannot build pseudo labels."
            )

        # Expected decoded YOLOv8 prediction:
        # [B, 4 + nc, A]
        if preds.ndim != 3:
            raise RuntimeError(f"Unsupported decoded prediction shape: {tuple(preds.shape)}")

        b = preds.shape[0]
        device = preds.device
        dtype = preds.dtype

        # If prediction is [B, A, 4+nc], transpose.
        if preds.shape[1] < preds.shape[2]:
            pred = preds
        else:
            pred = preds.permute(0, 2, 1).contiguous()

        boxes_xywh = pred[:, :4, :].permute(0, 2, 1).contiguous()
        cls_probs = pred[:, 4:, :].permute(0, 2, 1).contiguous()

        boxes_xyxy = xywh2xyxy(boxes_xywh)
        det_conf, cls = cls_probs.max(dim=-1)

        uncertainty = torch.zeros_like(det_conf)
        certainty = torch.ones_like(det_conf)
        combined_conf = det_conf
        score = (1.0 - float(delta)) * det_conf + float(delta) * combined_conf

        outputs = []

        for i in range(b):
            keep = score[i] >= self.conf_thres

            if not bool(keep.any()):
                outputs.append(torch.zeros((0, 10), device=device, dtype=dtype))
                continue

            det = torch.cat(
                [
                    boxes_xyxy[i, keep],
                    score[i, keep].unsqueeze(1),
                    cls[i, keep].float().unsqueeze(1),
                    det_conf[i, keep].unsqueeze(1),
                    uncertainty[i, keep].unsqueeze(1),
                    combined_conf[i, keep].unsqueeze(1),
                    certainty[i, keep].unsqueeze(1),
                ],
                dim=1,
            )

            det = self._class_aware_nms(det)
            outputs.append(det)

        return outputs


class ConfMixYOLOv8Method:
    """
    YOLOv8 ConfMix with DFL uncertainty.

    Loss:
        total = source_supervised_loss + lambda_mix * gamma * mixed_pseudo_loss

    Pseudo score:
        det_conf       = YOLOv8 max class probability
        uncertainty    = normalized DFL entropy
        certainty      = 1 - uncertainty
        combined_conf  = det_conf * certainty
        pseudo_score   = (1 - delta) * det_conf + delta * combined_conf

    Region selection:
        uses mean combined_conf in each region.
    """

    def __init__(
        self,
        model: Any,
        device: str | torch.device = "cuda",
        lambda_mix: float = 1.0,
        pseudo_conf_thres: float = 0.25,
        pseudo_iou_thres: float = 0.5,
        max_det: int = 300,
        gamma_max: float = 1.0,
        use_source_gt_for_mix: bool = True,
        box_loss_gain: float = 7.5,
        cls_loss_gain: float = 0.5,
        dfl_loss_gain: float = 1.5,
        uncertainty_power: float = 1.0,
        region_score_key: str = "combined_conf",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.lambda_mix = float(lambda_mix)
        self.pseudo_conf_thres = float(pseudo_conf_thres)
        self.pseudo_iou_thres = float(pseudo_iou_thres)
        self.max_det = int(max_det)
        self.gamma_max = float(gamma_max)
        self.use_source_gt_for_mix = bool(use_source_gt_for_mix)
        self.region_score_key = str(region_score_key)

        self.detection_loss = DetectionLoss(
            model=model,
            device=self.device,
            box=box_loss_gain,
            cls=cls_loss_gain,
            dfl=dfl_loss_gain,
        )

        self.torch_model = self._unwrap_torch_model(model)

        self.postprocessor = YOLOv8DFLUncertaintyPostprocessor(
            torch_model=self.torch_model,
            conf_thres=self.pseudo_conf_thres,
            iou_thres=self.pseudo_iou_thres,
            max_det=self.max_det,
            uncertainty_power=uncertainty_power,
        )

    def _unwrap_torch_model(self, model: Any) -> nn.Module:
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            return model.model

        if isinstance(model, nn.Module):
            return model

        raise TypeError(f"Unsupported model type: {type(model)!r}")

    def training_step(
        self,
        source_batch: dict[str, Any],
        target_batch: dict[str, Any],
        progress: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        source_images = source_batch["img"].to(self.device, non_blocking=True).float()
        target_images = target_batch["img"].to(self.device, non_blocking=True).float()

        if source_images.numel() > 0 and source_images.max() > 2.0:
            source_images = source_images / 255.0

        if target_images.numel() > 0 and target_images.max() > 2.0:
            target_images = target_images / 255.0

        source_batch = dict(source_batch)
        target_batch = dict(target_batch)

        source_batch["img"] = source_images
        source_batch["images"] = source_images
        target_batch["img"] = target_images
        target_batch["images"] = target_images

        delta = self._sigmoid_ramp(progress)
        gamma = torch.tensor(
            self.gamma_max * delta,
            device=self.device,
            dtype=source_images.dtype,
        )

        # ------------------------------------------------------------
        # 1. Source supervised loss.
        # ------------------------------------------------------------
        source_outputs = self.model(source_batch)
        source_loss_out = self.detection_loss(
            outputs=source_outputs,
            targets=source_batch,
            return_dict=True,
        )
        source_loss = source_loss_out.loss

        # ------------------------------------------------------------
        # 2. Target pseudo labels with DFL uncertainty.
        # ------------------------------------------------------------
        with torch.no_grad():
            target_dets = self.postprocessor(
                images=target_images,
                delta=delta,
            )

        # ------------------------------------------------------------
        # 3. Build ConfMix image and pseudo targets.
        # ------------------------------------------------------------
        mix_batch, mix_stats = self._build_confmix_batch(
            source_batch=source_batch,
            target_images=target_images,
            target_dets=target_dets,
        )

        # ------------------------------------------------------------
        # 4. Mixed pseudo-label loss.
        # ------------------------------------------------------------
        mix_outputs = self.model(mix_batch)
        mix_loss_out = self.detection_loss(
            outputs=mix_outputs,
            targets=mix_batch,
            return_dict=True,
        )
        mix_loss = mix_loss_out.loss

        total_loss = source_loss + self.lambda_mix * gamma * mix_loss

        return {
            "loss": total_loss,
            "source_loss": source_loss.detach(),
            "mix_loss": mix_loss.detach(),
            "gamma": gamma.detach(),
            "delta": torch.tensor(delta, device=self.device),
            "num_pseudo": torch.tensor(float(mix_stats["num_pseudo"]), device=self.device),
            "mean_uncertainty": torch.tensor(float(mix_stats["mean_uncertainty"]), device=self.device),
            "mean_certainty": torch.tensor(float(mix_stats["mean_certainty"]), device=self.device),
            "mean_det_conf": torch.tensor(float(mix_stats["mean_det_conf"]), device=self.device),
            "mean_combined_conf": torch.tensor(float(mix_stats["mean_combined_conf"]), device=self.device),
            "box_loss": source_loss_out.loss_items.get(
                "box_loss",
                torch.zeros((), device=self.device),
            ).detach(),
            "cls_loss": source_loss_out.loss_items.get(
                "cls_loss",
                torch.zeros((), device=self.device),
            ).detach(),
            "dfl_loss": source_loss_out.loss_items.get(
                "dfl_loss",
                torch.zeros((), device=self.device),
            ).detach(),
        }

    def _build_confmix_batch(
        self,
        source_batch: dict[str, Any],
        target_images: torch.Tensor,
        target_dets: list[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        source_images = source_batch["img"].to(self.device, non_blocking=True)
        b, c, h, w = source_images.shape

        mixed_images = source_images.clone()
        batch_targets: list[torch.Tensor] = []

        source_targets = source_batch.get("targets", None)
        if source_targets is not None:
            source_targets = source_targets.to(self.device)

        stat_uncertainty = []
        stat_certainty = []
        stat_det_conf = []
        stat_combined_conf = []

        num_pseudo = 0

        for i in range(b):
            det = target_dets[i]
            region = self._select_region(
                det=det,
                h=h,
                w=w,
            )

            if region is None:
                if source_targets is not None:
                    src_rows = source_targets[source_targets[:, 0] == i]
                    if src_rows.numel() > 0:
                        src_rows = src_rows.clone()
                        src_rows[:, 0] = i
                        batch_targets.append(src_rows)
                continue

            x1, y1, x2, y2 = region

            mixed_images[i, :, y1:y2, x1:x2] = target_images[i, :, y1:y2, x1:x2]

            if self.use_source_gt_for_mix and source_targets is not None:
                src_rows = source_targets[source_targets[:, 0] == i]
                if src_rows.numel() > 0:
                    kept_src = self._filter_source_targets_outside_region(
                        src_rows=src_rows,
                        region=region,
                        image_h=h,
                        image_w=w,
                    )

                    if kept_src.numel() > 0:
                        kept_src[:, 0] = i
                        batch_targets.append(kept_src)

            pseudo_rows, pseudo_stats = self._target_detections_to_targets(
                det=det,
                batch_index=i,
                region=region,
                image_h=h,
                image_w=w,
            )

            if pseudo_rows.numel() > 0:
                num_pseudo += int(pseudo_rows.shape[0])
                batch_targets.append(pseudo_rows)

                stat_uncertainty.extend(pseudo_stats["uncertainty"])
                stat_certainty.extend(pseudo_stats["certainty"])
                stat_det_conf.extend(pseudo_stats["det_conf"])
                stat_combined_conf.extend(pseudo_stats["combined_conf"])

        if batch_targets:
            targets = torch.cat(batch_targets, dim=0)
        else:
            targets = torch.zeros((0, 6), device=self.device, dtype=torch.float32)

        if targets.numel() > 0:
            batch_idx = targets[:, 0].long()
            cls = targets[:, 1:2].float()
            bboxes = targets[:, 2:6].float()
        else:
            batch_idx = torch.zeros((0,), device=self.device, dtype=torch.long)
            cls = torch.zeros((0, 1), device=self.device, dtype=torch.float32)
            bboxes = torch.zeros((0, 4), device=self.device, dtype=torch.float32)

        mix_batch = {
            "img": mixed_images,
            "images": mixed_images,
            "targets": targets,
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

        mix_stats = {
            "num_pseudo": float(num_pseudo),
            "mean_uncertainty": self._safe_mean(stat_uncertainty),
            "mean_certainty": self._safe_mean(stat_certainty),
            "mean_det_conf": self._safe_mean(stat_det_conf),
            "mean_combined_conf": self._safe_mean(stat_combined_conf),
        }

        return mix_batch, mix_stats

    def _select_region(
        self,
        det: torch.Tensor | None,
        h: int,
        w: int,
    ) -> Optional[tuple[int, int, int, int]]:
        if det is None or det.numel() == 0:
            return None

        regions = [
            (0, 0, w // 2, h // 2),       # left-top
            (w // 2, 0, w, h // 2),       # right-top
            (0, h // 2, w // 2, h),       # left-bottom
            (w // 2, h // 2, w, h),       # right-bottom
        ]

        boxes = det[:, :4]

        if self.region_score_key == "score":
            scores = det[:, 4]
        elif self.region_score_key == "det_conf":
            scores = det[:, 6]
        elif self.region_score_key == "uncertainty":
            scores = 1.0 - det[:, 7]
        elif self.region_score_key == "certainty":
            scores = det[:, 9]
        else:
            # Default: original ConfMix equivalent uses combined confidence.
            scores = det[:, 8]

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        best_region = None
        best_score = -1.0

        for region in regions:
            x1, y1, x2, y2 = region

            mask = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)

            if not bool(mask.any()):
                continue

            mean_score = float(scores[mask].mean().detach().cpu())

            if mean_score > best_score:
                best_score = mean_score
                best_region = region

        return best_region

    def _filter_source_targets_outside_region(
        self,
        src_rows: torch.Tensor,
        region: tuple[int, int, int, int],
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        if src_rows.numel() == 0:
            return src_rows

        x1, y1, x2, y2 = region

        boxes_xywh = src_rows[:, 2:6].clone()
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        boxes_xyxy[:, [0, 2]] *= float(image_w)
        boxes_xyxy[:, [1, 3]] *= float(image_h)

        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2

        inside = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)

        keep = ~inside

        return src_rows[keep].clone()

    def _target_detections_to_targets(
        self,
        det: torch.Tensor | None,
        batch_index: int,
        region: tuple[int, int, int, int],
        image_h: int,
        image_w: int,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        empty_stats = {
            "uncertainty": [],
            "certainty": [],
            "det_conf": [],
            "combined_conf": [],
        }

        if det is None or det.numel() == 0:
            return torch.zeros((0, 6), device=self.device, dtype=torch.float32), empty_stats

        x1, y1, x2, y2 = region

        boxes = det[:, :4].clone()
        cls = det[:, 5]
        det_conf = det[:, 6]
        uncertainty = det[:, 7]
        combined_conf = det[:, 8]
        certainty = det[:, 9]

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        inside = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)

        if not bool(inside.any()):
            return torch.zeros((0, 6), device=self.device, dtype=torch.float32), empty_stats

        boxes = boxes[inside]
        cls = cls[inside]
        det_conf = det_conf[inside]
        uncertainty = uncertainty[inside]
        combined_conf = combined_conf[inside]
        certainty = certainty[inside]

        boxes[:, 0].clamp_(float(x1), float(x2))
        boxes[:, 2].clamp_(float(x1), float(x2))
        boxes[:, 1].clamp_(float(y1), float(y2))
        boxes[:, 3].clamp_(float(y1), float(y2))

        wh = boxes[:, 2:4] - boxes[:, 0:2]
        valid = (wh[:, 0] > 2.0) & (wh[:, 1] > 2.0)

        if not bool(valid.any()):
            return torch.zeros((0, 6), device=self.device, dtype=torch.float32), empty_stats

        boxes = boxes[valid]
        cls = cls[valid]
        det_conf = det_conf[valid]
        uncertainty = uncertainty[valid]
        combined_conf = combined_conf[valid]
        certainty = certainty[valid]

        boxes_xywh = xyxy2xywh(boxes)

        boxes_xywh[:, [0, 2]] /= float(image_w)
        boxes_xywh[:, [1, 3]] /= float(image_h)
        boxes_xywh.clamp_(0.0, 1.0)

        batch_idx = torch.full(
            (boxes_xywh.shape[0], 1),
            float(batch_index),
            device=self.device,
            dtype=torch.float32,
        )

        rows = torch.cat(
            [
                batch_idx,
                cls.view(-1, 1).float(),
                boxes_xywh.float(),
            ],
            dim=1,
        )

        stats = {
            "uncertainty": [float(x) for x in uncertainty.detach().cpu().tolist()],
            "certainty": [float(x) for x in certainty.detach().cpu().tolist()],
            "det_conf": [float(x) for x in det_conf.detach().cpu().tolist()],
            "combined_conf": [float(x) for x in combined_conf.detach().cpu().tolist()],
        }

        return rows, stats

    def _sigmoid_ramp(self, progress: float) -> float:
        progress = max(0.0, min(1.0, float(progress)))
        x = torch.tensor(
            -5.0 * progress,
            device=self.device,
            dtype=torch.float32,
        )
        return float(2.0 / (1.0 + torch.exp(x)) - 1.0)

    def _safe_mean(self, values: list[float]) -> float:
        if not values:
            return 0.0

        return float(sum(values) / max(1, len(values)))