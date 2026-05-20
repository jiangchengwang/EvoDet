# clod_framework/engine/evaluator.py

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy


@dataclass
class DetectionEvalResult:
    epoch: int
    task_id: int

    precision: float
    recall: float
    mAP50: float
    mAP5095: float

    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_ap50: list[float]
    per_class_ap: list[float]

    num_images: int = 0
    num_labels: int = 0


class DetectionEvaluator:
    """
    Ultralytics-style detection evaluator.

    This matches YOLO training-log style metrics:
        - P
        - R
        - mAP50
        - mAP50-95

    It does NOT require COCO json.
    It evaluates directly from YOLO-format batch targets:
        targets: [batch_idx, cls, x, y, w, h], normalized to input image.
    """

    def __init__(
        self,
        num_classes: int,
        metrics_dir: str | Path,
        device: str | torch.device | None = None,
        conf_thres: float = 0.001,
        nms_iou_thres: float = 0.7,
        max_det: int = 300,
        iou_thresholds: Optional[Sequence[float]] = None,
        class_agnostic_nms: bool = False,
        save_csv: bool = True,
        show_progress: bool = True,
        class_names: Optional[Sequence[str]] = None,
        print_results: bool = True,
    ) -> None:
        self.num_classes = int(num_classes)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.conf_thres = float(conf_thres)
        self.nms_iou_thres = float(nms_iou_thres)
        self.max_det = int(max_det)
        self.class_agnostic_nms = bool(class_agnostic_nms)

        self.save_csv = bool(save_csv)
        self.show_progress = bool(show_progress)
        self.print_results = bool(print_results)

        if iou_thresholds is None:
            self.iouv = torch.linspace(0.5, 0.95, 10, device=self.device)
        else:
            self.iouv = torch.tensor(iou_thresholds, device=self.device).float()

        self.niou = int(self.iouv.numel())
        self.class_names = self._normalize_class_names(class_names)

    @torch.no_grad()
    def evaluate(
        self,
        model: Any,
        dataloader: Any,
        epoch: int,
        task_id: int,
        seen_classes: Optional[Sequence[int]] = None,
        csv_name: Optional[str] = None,
    ) -> DetectionEvalResult:
        self._set_eval(model)

        seen_classes = (
            [int(c) for c in seen_classes if 0 <= int(c) < self.num_classes]
            if seen_classes is not None
            else list(range(self.num_classes))
        )

        stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        total_images = 0
        total_labels = 0

        iterator = dataloader
        if self.show_progress:
            iterator = tqdm(
                dataloader,
                desc=f"YOLO eval task {task_id} epoch {epoch}",
                ncols=120,
            )

        for batch in iterator:
            imgs = batch["img"].to(self.device, non_blocking=True).float()
            if imgs.max() > 2.0:
                imgs = imgs / 255.0

            preds = self._predict_raw(model, imgs)

            preds = non_max_suppression(
                prediction=preds,
                conf_thres=self.conf_thres,
                iou_thres=self.nms_iou_thres,
                classes=seen_classes,
                agnostic=self.class_agnostic_nms,
                max_det=self.max_det,
                nc=self.num_classes,
            )

            targets = batch["targets"].to(self.device)
            height, width = imgs.shape[-2:]
            total_images += int(imgs.shape[0])
            total_labels += int(targets.shape[0])

            for image_idx, pred in enumerate(preds):
                labels = targets[targets[:, 0] == image_idx]

                if labels.numel() > 0:
                    tcls = labels[:, 1]
                    tbox = xywh2xyxy(labels[:, 2:6])
                    tbox[:, [0, 2]] *= width
                    tbox[:, [1, 3]] *= height
                else:
                    tcls = torch.zeros(0, device=self.device)
                    tbox = torch.zeros((0, 4), device=self.device)

                if pred is None or pred.numel() == 0:
                    if labels.numel() > 0:
                        stats.append(
                            (
                                torch.zeros((0, self.niou), dtype=torch.bool, device=self.device),
                                torch.zeros(0, device=self.device),
                                torch.zeros(0, device=self.device),
                                tcls,
                            )
                        )
                    continue

                pred = pred.to(self.device)
                correct = self._process_batch(pred, tbox, tcls)

                stats.append(
                    (
                        correct,
                        pred[:, 4],
                        pred[:, 5],
                        tcls,
                    )
                )

        result = self._compute_result(
            stats=stats,
            epoch=epoch,
            task_id=task_id,
            seen_classes=seen_classes,
            num_images=total_images,
            num_labels=total_labels,
        )

        if self.print_results:
            self.print_eval_result(result, seen_classes)

        if self.save_csv:
            if csv_name is None:
                csv_name = f"mAPs_task_{task_id}.csv"
            self.append_yolo_csv(result, self.metrics_dir / csv_name)

        return result

    def _process_batch(
        self,
        detections: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_cls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match predictions to GT over IoU thresholds.

        detections:
            [N, 6] = xyxy, conf, cls

        gt_boxes:
            [M, 4] = xyxy

        gt_cls:
            [M]
        """

        correct = torch.zeros(
            detections.shape[0],
            self.niou,
            dtype=torch.bool,
            device=detections.device,
        )

        if detections.numel() == 0 or gt_boxes.numel() == 0:
            return correct

        iou = box_iou(gt_boxes, detections[:, :4])
        correct_class = gt_cls[:, None] == detections[:, 5]

        for i, threshold in enumerate(self.iouv):
            matches = torch.where((iou >= threshold) & correct_class)

            if matches[0].numel() == 0:
                continue

            match_data = torch.cat(
                (
                    torch.stack(matches, dim=1).float(),
                    iou[matches[0], matches[1]][:, None],
                ),
                dim=1,
            )

            match_np = match_data.detach().cpu().numpy()

            if match_np.shape[0] > 1:
                # Sort by IoU descending.
                match_np = match_np[match_np[:, 2].argsort()[::-1]]

                # Unique prediction.
                match_np = match_np[np.unique(match_np[:, 1], return_index=True)[1]]

                # Sort again and unique GT.
                match_np = match_np[match_np[:, 2].argsort()[::-1]]
                match_np = match_np[np.unique(match_np[:, 0], return_index=True)[1]]

            det_indices = torch.as_tensor(
                match_np[:, 1].astype(np.int64),
                device=detections.device,
                dtype=torch.long,
            )

            correct[det_indices, i] = True

        return correct

    def _compute_result(
        self,
        stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        epoch: int,
        task_id: int,
        seen_classes: Sequence[int],
        num_images: int,
        num_labels: int,
    ) -> DetectionEvalResult:
        per_class_precision = [0.0 for _ in range(self.num_classes)]
        per_class_recall = [0.0 for _ in range(self.num_classes)]
        per_class_ap50 = [0.0 for _ in range(self.num_classes)]
        per_class_ap = [0.0 for _ in range(self.num_classes)]

        if not stats:
            return DetectionEvalResult(
                epoch=epoch,
                task_id=task_id,
                precision=0.0,
                recall=0.0,
                mAP50=0.0,
                mAP5095=0.0,
                per_class_precision=per_class_precision,
                per_class_recall=per_class_recall,
                per_class_ap50=per_class_ap50,
                per_class_ap=per_class_ap,
                num_images=num_images,
                num_labels=num_labels,
            )

        stats_np = [
            torch.cat(x, 0).detach().cpu().numpy()
            if len(x) and torch.cat(x, 0).numel()
            else np.array([])
            for x in zip(*stats)
        ]

        tp, conf, pred_cls, target_cls = stats_np

        if tp.size == 0 or target_cls.size == 0:
            return DetectionEvalResult(
                epoch=epoch,
                task_id=task_id,
                precision=0.0,
                recall=0.0,
                mAP50=0.0,
                mAP5095=0.0,
                per_class_precision=per_class_precision,
                per_class_recall=per_class_recall,
                per_class_ap50=per_class_ap50,
                per_class_ap=per_class_ap,
                num_images=num_images,
                num_labels=num_labels,
            )

        names = {i: self.class_names[i] for i in range(self.num_classes)}

        ap_out = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=False,
            save_dir=self.metrics_dir,
            names=names,
        )

        # Ultralytics versions may return:
        #   tp, fp, p, r, f1, ap, unique_classes, ...
        tp_out, fp_out, p, r, f1, ap, unique_classes = ap_out[:7]
        unique_classes = unique_classes.astype(int)

        for i, cls_id in enumerate(unique_classes):
            cls_id = int(cls_id)
            if not (0 <= cls_id < self.num_classes):
                continue

            per_class_precision[cls_id] = float(p[i]) if len(p) > i else 0.0
            per_class_recall[cls_id] = float(r[i]) if len(r) > i else 0.0
            per_class_ap50[cls_id] = float(ap[i, 0])
            per_class_ap[cls_id] = float(ap[i].mean())

        # YOLO official style averages over classes present in target.
        target_unique = sorted(set(int(x) for x in target_cls.tolist()))
        eval_classes = [c for c in target_unique if c in seen_classes]

        if not eval_classes:
            eval_classes = [int(c) for c in seen_classes if 0 <= int(c) < self.num_classes]

        precision = float(np.mean([per_class_precision[c] for c in eval_classes])) if eval_classes else 0.0
        recall = float(np.mean([per_class_recall[c] for c in eval_classes])) if eval_classes else 0.0
        mAP50 = float(np.mean([per_class_ap50[c] for c in eval_classes])) if eval_classes else 0.0
        mAP5095 = float(np.mean([per_class_ap[c] for c in eval_classes])) if eval_classes else 0.0

        return DetectionEvalResult(
            epoch=epoch,
            task_id=task_id,
            precision=precision,
            recall=recall,
            mAP50=mAP50,
            mAP5095=mAP5095,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_ap50=per_class_ap50,
            per_class_ap=per_class_ap,
            num_images=num_images,
            num_labels=num_labels,
        )

    def _predict_raw(self, model: Any, imgs: torch.Tensor) -> torch.Tensor:
        if hasattr(model, "predict_raw"):
            preds = model.predict_raw(imgs)
        else:
            preds = model(imgs)

        if hasattr(preds, "raw"):
            preds = preds.raw

        if isinstance(preds, dict):
            for key in ("raw", "preds", "pred", "outputs"):
                if key in preds:
                    preds = preds[key]
                    break

        if isinstance(preds, tuple):
            preds = preds[0]

        if isinstance(preds, list):
            if len(preds) == 2 and isinstance(preds[0], torch.Tensor):
                preds = preds[0]
            elif len(preds) > 0 and isinstance(preds[0], torch.Tensor):
                preds = preds[0]

        if not isinstance(preds, torch.Tensor):
            raise TypeError(f"Unsupported prediction type for evaluator: {type(preds)!r}")

        return preds

    def append_yolo_csv(
        self,
        result: DetectionEvalResult,
        path: str | Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = (
            [
                "epoch",
                "precision",
                "recall",
                "mAP50",
                "mAP50-95",
                "num_images",
                "num_labels",
            ]
            + [f"class_{i}_P" for i in range(self.num_classes)]
            + [f"class_{i}_R" for i in range(self.num_classes)]
            + [f"class_{i}_AP50" for i in range(self.num_classes)]
            + [f"class_{i}_AP" for i in range(self.num_classes)]
        )

        row = (
            [
                result.epoch,
                result.precision,
                result.recall,
                result.mAP50,
                result.mAP5095,
                result.num_images,
                result.num_labels,
            ]
            + result.per_class_precision
            + result.per_class_recall
            + result.per_class_ap50
            + result.per_class_ap
        )

        write_header = not path.exists()

        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    def print_eval_result(
        self,
        result: DetectionEvalResult,
        seen_classes: Optional[Sequence[int]] = None,
    ) -> None:
        if seen_classes is None:
            seen_classes = list(range(self.num_classes))

        seen_classes = [int(c) for c in seen_classes if 0 <= int(c) < self.num_classes]

        print("")
        print(
            f"{'Class':>20} {'Images':>10} {'Labels':>10} "
            f"{'P':>10} {'R':>10} {'mAP@.5':>10} {'mAP@.5:.95':>12}"
        )
        print("-" * 86)

        print(
            f"{'all':>20} "
            f"{result.num_images:>10} "
            f"{result.num_labels:>10} "
            f"{result.precision:>10.4f} "
            f"{result.recall:>10.4f} "
            f"{result.mAP50:>10.4f} "
            f"{result.mAP5095:>12.4f}"
        )

        for cls_id in seen_classes:
            print(
                f"{self.class_names[cls_id]:>20} "
                f"{'':>10} "
                f"{'':>10} "
                f"{result.per_class_precision[cls_id]:>10.4f} "
                f"{result.per_class_recall[cls_id]:>10.4f} "
                f"{result.per_class_ap50[cls_id]:>10.4f} "
                f"{result.per_class_ap[cls_id]:>12.4f}"
            )

        print("-" * 86)
        print(
            f"task={result.task_id}, epoch={result.epoch}, "
            f"P={result.precision:.4f}, "
            f"R={result.recall:.4f}, "
            f"mAP50={result.mAP50:.4f}, "
            f"mAP50-95={result.mAP5095:.4f}"
        )
        print("")

    def _normalize_class_names(
        self,
        class_names: Optional[Sequence[str]],
    ) -> list[str]:
        if class_names is None:
            return [f"class_{i}" for i in range(self.num_classes)]

        names = list(class_names)

        if len(names) < self.num_classes:
            names = names + [f"class_{i}" for i in range(len(names), self.num_classes)]

        return names[: self.num_classes]

    def _set_eval(self, model: Any) -> None:
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.eval()
        elif isinstance(model, torch.nn.Module):
            model.eval()
        elif hasattr(model, "eval"):
            model.eval()