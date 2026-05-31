"""Quantitative evaluation of Stages 1-2 against the Roboflow COCO labels.

Metrics
-------
Table detection
  * found rate: fraction of images where a table quadrilateral is returned.
  * bbox IoU: IoU between the detected quad's bounding box and the GT
    "pool table" box (when both exist).

Ball detection (evaluated in the rectified, metric frame)
  * precision / recall / F1 against GT ball boxes whose centers project inside
    the rectified table.
  * mean positional error (cm) over matched pairs.

GT ball centers are projected through the Stage-1 homography into the rectified
frame, then greedily matched to detections by nearest center within one ball
diameter (5.7 cm).

Run:
    python src/evaluate_detection.py --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # for train.py (dataset loader)
sys.path.insert(0, str(ROOT / "src"))  # for stage modules

from train import BilliardDataset  # noqa: E402

from ball_detection import detect_balls  # noqa: E402
from table_detection import detect_table  # noqa: E402

DATASET = ROOT / "data" / "Billiard Pool.v5i.coco"
BALL_LABELS = {"white ball", *(f"ball {i}" for i in range(1, 10))}
MATCH_DIST_CM = 5.7  # one ball diameter


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _project_to_cm(centers_px, H, px_per_cm):
    if not centers_px:
        return []
    pts = np.array([[[x, y]] for x, y in centers_px], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H)
    return [(float(x) / px_per_cm, float(y) / px_per_cm) for [[x, y]] in out]


def _match(gt_cm, det_cm, thresh):
    """Greedy nearest matching. Returns (tp, matched_errors)."""
    used = [False] * len(det_cm)
    tp, errors = 0, []
    for gx, gy in gt_cm:
        best_j, best_d = -1, thresh
        for j, (dx, dy) in enumerate(det_cm):
            if used[j]:
                continue
            d = ((gx - dx) ** 2 + (gy - dy) ** 2) ** 0.5
            if d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            tp += 1
            errors.append(best_d)
    return tp, errors


def evaluate(split: str, limit: int | None = None):
    ds = BilliardDataset(DATASET, split)

    n_images = 0
    n_table_found = 0
    ious = []
    total_tp = total_fp = total_fn = 0
    all_errors = []

    for i in range(len(ds)):
        if limit and n_images >= limit:
            break
        sample = ds[i]
        img = ds.load_image(i)
        n_images += 1

        table = detect_table(img)
        if not table.found:
            # All visible GT balls become misses.
            total_fn += sum(1 for b in sample.boxes if b.label in BALL_LABELS)
            continue
        n_table_found += 1

        # Table IoU vs GT "pool table" box.
        gt_table = next((b for b in sample.boxes if b.label == "pool table"), None)
        if gt_table is not None:
            qx = table.corners[:, 0]
            qy = table.corners[:, 1]
            quad_bbox = (qx.min(), qy.min(), qx.max(), qy.max())
            ious.append(_bbox_iou(quad_bbox, gt_table.xyxy))

        # Ball detection in the rectified cm frame.
        H, W = table.warped.shape[:2]
        out_w_cm, out_h_cm = W / table.px_per_cm, H / table.px_per_cm

        gt_centers_px = [b.center for b in sample.boxes if b.label in BALL_LABELS]
        gt_cm_all = _project_to_cm(gt_centers_px, table.homography, table.px_per_cm)
        # Only GT balls that project inside the rectified table are evaluable.
        gt_cm = [(x, y) for x, y in gt_cm_all if 0 <= x <= out_w_cm and 0 <= y <= out_h_cm]

        balls = detect_balls(table, img)
        det_cm = [(b.x_cm, b.y_cm) for b in balls]

        tp, errors = _match(gt_cm, det_cm, MATCH_DIST_CM)
        total_tp += tp
        total_fp += len(det_cm) - tp
        total_fn += len(gt_cm) - tp
        all_errors.extend(errors)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"=== Detection evaluation: split={split} ({n_images} images) ===")
    print(f"Table found rate : {n_table_found}/{n_images} = {n_table_found / n_images:.1%}")
    if ious:
        print(f"Table bbox IoU   : mean {np.mean(ious):.3f} (n={len(ious)})")
    print(f"Ball precision   : {precision:.3f}")
    print(f"Ball recall      : {recall:.3f}")
    print(f"Ball F1          : {f1:.3f}")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    if all_errors:
        print(f"Mean pos. error  : {np.mean(all_errors):.2f} cm "
              f"(median {np.median(all_errors):.2f} cm, n={len(all_errors)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--limit", type=int, default=None)
    main_args = ap.parse_args()
    evaluate(main_args.split, main_args.limit)


if __name__ == "__main__":
    main()
