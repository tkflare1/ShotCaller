"""Batch demo: run the ShotCaller pipeline over a set of images.

Picks a handful of images from the Roboflow dataset (plus any sample photos),
runs the full pipeline, and writes camera-view and rectified overlays into
``results/``. Useful for the milestone visualizations and qualitative review.

Usage:
    python src/demo.py [--n 6] [--split train]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from pipeline import run

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "Billiard Pool.v5i.coco"
RESULTS = ROOT / "results"


def _dataset_images(split: str, n: int) -> list[Path]:
    split_dir = DATASET / split
    if not split_dir.exists():
        return []
    imgs = sorted(p for p in split_dir.glob("*.jpg"))
    # Spread the picks across the split rather than taking the first n.
    if len(imgs) <= n:
        return imgs
    step = len(imgs) // n
    return [imgs[i * step] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="number of dataset images")
    ap.add_argument("--split", default="train", choices=["train", "valid", "test"])
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    images = _dataset_images(args.split, args.n)

    # Include any standalone sample photos in data/ as well.
    images += sorted((ROOT / "data").glob("sample_pool_*.jpg"))

    if not images:
        print("No images found. Is the dataset in data/?")
        return

    summary = []
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue
        res = run(img)
        stem = f"{args.split}_{i:02d}"
        if not res.table.found:
            summary.append((path.name, "no table"))
            continue

        cv2.imwrite(str(RESULTS / f"{stem}_camera.png"), res.overlay_camera)
        cv2.imwrite(str(RESULTS / f"{stem}_rectified.png"), res.overlay_rectified)

        traj = res.trajectory
        summary.append((
            path.name,
            f"{len(res.balls)} balls, cue={'Y' if res.cue else 'N'}, "
            f"bounces={traj.bounces if traj else '-'}, "
            f"collision={'Y' if traj and traj.collision else 'N'}",
        ))

    print("\n=== ShotCaller batch demo ===")
    for name, info in summary:
        print(f"  {name[:42]:42s}  {info}")
    print(f"\nOverlays written to {RESULTS}/")


if __name__ == "__main__":
    main()
