#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Move FoggyCityscapes images whose filename does not contain the requested beta.")
    parser.add_argument("--src", required=True, help="Image directory to clean")
    parser.add_argument("--dst", required=True, help="Where non-matching beta images are moved")
    parser.add_argument("--beta", default="0.02")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise FileNotFoundError(src)
    moved = 0
    kept = 0
    for path in src.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if args.beta in path.name:
            kept += 1
            continue
        target = dst / path.relative_to(src)
        print(f"MOVE {path} -> {target}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        moved += 1
    print({"kept": kept, "moved": moved, "dry_run": args.dry_run})


if __name__ == "__main__":
    main()
