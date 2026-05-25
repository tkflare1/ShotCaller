"""Data loader for the ShotCaller Billiard Pool COCO dataset.

Loads the Roboflow "Billiard Pool" export, which ships three COCO splits
(train / valid / test). Each split is a directory of .jpg images plus an
``_annotations.coco.json`` file.

COCO annotation conventions used here:
  * bbox = [x, y, width, height], with (x, y) the top-left corner in pixels.
  * category_id 0 ("balls") is the COCO supercategory placeholder and is
    skipped. The cue ball is category "white ball"; numbered balls are
    "ball 1".."ball 9"; structural classes are "pool table" and "rack".

Example
-------
    from dataset import BilliardDataset

    train = BilliardDataset("data/Billiard Pool.v5i.coco", "train")
    test  = BilliardDataset("data/Billiard Pool.v5i.coco", "test")

    sample = train[0]
    image  = train.load_image(0)          # HxWx3 BGR uint8 (OpenCV order)
    for box in sample.boxes:
        print(box.label, box.xyxy)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "valid", "test")
ANNOTATION_FILE = "_annotations.coco.json"


@dataclass
class Box:
    """A single object annotation."""

    label: str           # category name, e.g. "white ball", "ball 8"
    category_id: int     # original COCO category id
    xywh: tuple[float, float, float, float]  # [x, y, w, h] top-left origin

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Corner format [x_min, y_min, x_max, y_max]."""
        x, y, w, h = self.xywh
        return (x, y, x + w, y + h)

    @property
    def center(self) -> tuple[float, float]:
        """Box center (cx, cy) — useful as a ball's position estimate."""
        x, y, w, h = self.xywh
        return (x + w / 2.0, y + h / 2.0)


@dataclass
class Sample:
    """One image and all of its boxes."""

    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    boxes: list[Box]


class BilliardDataset:
    """Indexable view over one COCO split of the Billiard Pool dataset."""

    def __init__(self, root: str | Path, split: str, skip_supercategory: bool = True):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        ann_path = self.split_dir / ANNOTATION_FILE
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            coco = json.load(f)

        # id -> category name
        self.categories: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}
        # categories to drop (the COCO supercategory placeholder, id 0 "balls")
        self._skip_ids = {0} if skip_supercategory else set()

        # Group annotations by image id.
        boxes_by_image: dict[int, list[Box]] = {}
        for ann in coco["annotations"]:
            cid = ann["category_id"]
            if cid in self._skip_ids:
                continue
            box = Box(
                label=self.categories.get(cid, str(cid)),
                category_id=cid,
                xywh=tuple(ann["bbox"]),
            )
            boxes_by_image.setdefault(ann["image_id"], []).append(box)

        # Build the ordered list of samples.
        self.samples: list[Sample] = []
        for img in coco["images"]:
            self.samples.append(
                Sample(
                    image_id=img["id"],
                    file_name=img["file_name"],
                    path=self.split_dir / img["file_name"],
                    width=img["width"],
                    height=img["height"],
                    boxes=boxes_by_image.get(img["id"], []),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    def load_image(self, idx: int) -> np.ndarray:
        """Read the image for sample ``idx`` as an HxWx3 BGR uint8 array."""
        sample = self.samples[idx]
        img = cv2.imread(str(sample.path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {sample.path}")
        return img

    def class_counts(self) -> dict[str, int]:
        """Total number of boxes per label across the split."""
        counts: dict[str, int] = {}
        for sample in self.samples:
            for box in sample.boxes:
                counts[box.label] = counts.get(box.label, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def draw(self, idx: int) -> np.ndarray:
        """Return a copy of image ``idx`` with all boxes drawn (for debugging)."""
        img = self.load_image(idx)
        for box in self.samples[idx].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, box.label, (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
            )
        return img


def load_splits(root: str | Path) -> dict[str, BilliardDataset]:
    """Convenience helper: load train / valid / test in one call."""
    return {split: BilliardDataset(root, split) for split in SPLITS}


if __name__ == "__main__":
    # Quick smoke test: load every split, print stats, and dump one annotated
    # sample image to disk so you can eyeball the parsing.
    DATA_ROOT = "data/Billiard Pool.v5i.coco"

    datasets = load_splits(DATA_ROOT)
    for split, ds in datasets.items():
        print(f"[{split}] {len(ds)} images, {sum(len(s.boxes) for s in ds)} boxes")

    print("\nTrain class distribution:")
    for label, count in datasets["train"].class_counts().items():
        print(f"  {label:12s} {count}")

    train = datasets["train"]
    sample = train[0]
    print(f"\nSample 0: {sample.file_name} ({sample.width}x{sample.height})")
    for box in sample.boxes:
        cx, cy = box.center
        print(f"  {box.label:12s} center=({cx:.0f},{cy:.0f}) xywh={box.xywh}")

    out = "sample_annotated.jpg"
    cv2.imwrite(out, train.draw(0))
    print(f"\nWrote annotated preview to {out}")
