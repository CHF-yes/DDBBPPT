"""Export the fixed CHF 1600/400 RGB split to COCO without copying images.

Images are hard-linked on the same filesystem. The output is published only
after all annotations and links pass validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

from PIL import Image

COMPETITION_CLASS_NAMES = (
    "person", "boat", "animal", "seat", "sign", "bicycle", "car",
    "ball", "light", "garbage_can", "uav", "tricycle",
)


def image_for(visible: Path, stem: str) -> Path:
    matches = [visible / f"{stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png")]
    found = [path for path in matches if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one RGB image for {stem}: {found}")
    return found[0]


def annotations_for(label_path: Path, image_id: int, width: int, height: int,
                    next_id: int, counts: Counter[int]) -> tuple[list[dict], int]:
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    annotations = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label {label_path}:{line_no}: {line}")
        cls = int(parts[0])
        if cls < 0 or cls >= len(COMPETITION_CLASS_NAMES):
            raise ValueError(f"Invalid class {cls} in {label_path}:{line_no}")
        cx, cy, bw, bh = [max(0.0, min(1.0, float(v))) for v in parts[1:5]]
        x0 = max(0.0, (cx - bw / 2) * width)
        y0 = max(0.0, (cy - bh / 2) * height)
        x1 = min(float(width), (cx + bw / 2) * width)
        y1 = min(float(height), (cy + bh / 2) * height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Degenerate box in {label_path}:{line_no}")
        w, h = x1 - x0, y1 - y0
        annotations.append({
            "id": next_id, "image_id": image_id, "category_id": cls,
            "bbox": [x0, y0, w, h], "area": w * h, "iscrowd": 0,
        })
        counts[cls] += 1
        next_id += 1
    return annotations, next_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    split_bytes = args.split.read_bytes()
    split = json.loads(split_bytes)
    train, val = split["train"], split["val"]
    if len(train) != 1600 or len(val) != 400 or len(set(train + val)) != 2000:
        raise ValueError("Split must contain disjoint 1600 train + 400 val stems")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visible = args.root / "visible"
    categories = [{"id": i, "name": name, "supercategory": "object"}
                  for i, name in enumerate(COMPETITION_CLASS_NAMES)]
    with tempfile.TemporaryDirectory(prefix=f".{args.output.name}-", dir=args.output.parent) as scratch:
        temporary = Path(scratch)
        summary = {"split_sha256": hashlib.sha256(split_bytes).hexdigest(),
                   "source_rgb": str(visible), "source_labels": str(args.labels),
                   "classes": list(COMPETITION_CLASS_NAMES), "splits": {}}
        for split_name, stems in (("train", train), ("valid", val)):
            folder = temporary / split_name
            folder.mkdir()
            images: list[dict] = []
            annotations: list[dict] = []
            counts: Counter[int] = Counter()
            next_id = 1
            for image_id, stem in enumerate(stems, start=1):
                source = image_for(visible, stem)
                with Image.open(source) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid image size: {source}")
                os.link(source, folder / source.name)
                images.append({"id": image_id, "file_name": source.name,
                               "width": width, "height": height})
                labels, next_id = annotations_for(args.labels / f"{stem}.txt",
                                                  image_id, width, height, next_id, counts)
                annotations.extend(labels)
            payload = {"info": {"description": "CHF fixed RGB split"},
                       "licenses": [], "categories": categories,
                       "images": images, "annotations": annotations}
            (folder / "_annotations.coco.json").write_text(json.dumps(payload))
            summary["splits"][split_name] = {
                "images": len(images), "annotations": len(annotations),
                "per_class_boxes": {COMPETITION_CLASS_NAMES[i]: counts[i]
                                    for i in range(len(COMPETITION_CLASS_NAMES))},
            }
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (temporary / "export_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        os.rename(temporary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
