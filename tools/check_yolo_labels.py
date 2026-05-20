#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    images = Path(args.images)
    labels = Path(args.labels)
    image_files = [p for p in images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    found = 0
    missing = []
    for img in image_files:
        rel = img.relative_to(images).with_suffix(".txt")
        label = labels / rel
        if label.exists():
            found += 1
        else:
            missing.append(str(label))
    print({"images": len(image_files), "labels_found": found, "labels_missing": len(missing)})
    if missing:
        print("first_missing:")
        for x in missing[:20]:
            print(x)


if __name__ == "__main__":
    main()
