from pathlib import Path
from collections import defaultdict

label_dir = Path("/datasets/MilTech/natural/labels/train")

task0 = {0, 1}
task1 = {2, 3}

total_images = 0
task0_images = 0
task1_images = 0
overlap_images = 0

class_image_count = defaultdict(int)

for label_path in label_dir.rglob("*.txt"):
    total_images += 1

    classes = set()

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        classes.add(cls)

    for c in classes:
        class_image_count[c] += 1

    has_task0 = bool(classes & task0)
    has_task1 = bool(classes & task1)

    if has_task0:
        task0_images += 1

    if has_task1:
        task1_images += 1

    if has_task0 and has_task1:
        overlap_images += 1

print(f"total label files/images: {total_images}")
print(f"task0 images [0,1]: {task0_images}")
print(f"task1 images [2,3]: {task1_images}")
print(f"overlap images task0 & task1: {overlap_images}")
print()
for c in range(4):
    print(f"class {c} image count: {class_image_count[c]}")