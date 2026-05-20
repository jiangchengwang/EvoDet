from __future__ import annotations


class DetectionMetrics:
    """Placeholder for mAP/mAR integration."""

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, float]:
        raise NotImplementedError
