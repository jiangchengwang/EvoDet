from __future__ import annotations

from pathlib import Path


def read_yolo_label_file(path: str | Path) -> list[dict]:
    path = Path(path)
    labels: list[dict] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(
            {
                "class_id": int(float(parts[0])),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
        )
    return labels
