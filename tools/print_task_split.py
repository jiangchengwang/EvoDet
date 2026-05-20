#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clod_framework.data.class_incremental_split import ClassIncrementalSplit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--initial", type=int, required=True)
    parser.add_argument("--increment", type=int, required=True)
    args = parser.parse_args()
    split = ClassIncrementalSplit(args.num_classes, args.initial, args.increment)
    for task in split.build():
        ids = list(task.class_ids)
        if ids:
            print(f"task {task.task_id}: {ids[0]}..{ids[-1]} ({len(ids)} classes), seen={len(task.seen_class_ids)}")
        else:
            print(f"task {task.task_id}: empty")


if __name__ == "__main__":
    main()
