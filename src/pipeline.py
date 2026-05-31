"""ShotCaller end-to-end pipeline.

Ties the three stages together:

    image -> Stage 1 (table + homography)
          -> Stage 2 (balls in rectified cm frame)
          -> Stage 3 (cue-ball trajectory in cm)
          -> overlays on both the rectified and the original camera view.

Run directly:

    python src/pipeline.py <image> [--aim DEG] [--out result.png]

If ``--aim`` is omitted, the cue is aimed at the nearest detected object ball.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np

from ball_detection import Ball, detect_balls, draw_balls
from table_detection import TableResult, detect_table
from trajectory import Trajectory, draw_trajectory, predict_trajectory


@dataclass
class ShotCallerResult:
    table: TableResult
    balls: list[Ball]
    cue: Ball | None
    trajectory: Trajectory | None
    overlay_rectified: np.ndarray | None
    overlay_camera: np.ndarray | None


def _pick_cue(balls: list[Ball]) -> Ball | None:
    """Choose the cue ball; if several are flagged, this still returns one."""
    cues = [b for b in balls if b.is_cue]
    if cues:
        return cues[0]
    return None


def _nearest_object(cue: Ball, balls: list[Ball]) -> Ball | None:
    others = [b for b in balls if b is not cue]
    if not others:
        return None
    return min(others, key=lambda b: (b.x_cm - cue.x_cm) ** 2 + (b.y_cm - cue.y_cm) ** 2)


def run(
    img: np.ndarray,
    aim_deg: float | None = None,
    felt_color: str | None = None,
    max_bounces: int = 4,
) -> ShotCallerResult:
    table = detect_table(img, felt_color=felt_color)
    if not table.found:
        return ShotCallerResult(table, [], None, None, None, None)

    balls = detect_balls(table, img)
    cue = _pick_cue(balls)

    px_per_cm = table.px_per_cm
    H, W = table.warped.shape[:2]
    table_cm = (W / px_per_cm, H / px_per_cm)

    traj = None
    if cue is not None:
        # Aim: explicit angle, else toward the nearest object ball.
        if aim_deg is None:
            target = _nearest_object(cue, balls)
            if target is not None:
                aim_deg = math.degrees(
                    math.atan2(target.y_cm - cue.y_cm, target.x_cm - cue.x_cm)
                )
            else:
                aim_deg = 0.0
        obstacles = [(b.x_cm, b.y_cm) for b in balls if b is not cue]
        traj = predict_trajectory(
            (cue.x_cm, cue.y_cm), aim_deg, table_cm, obstacles, max_bounces
        )

    overlay_rect = _render_rectified(table.warped, balls, traj, px_per_cm)
    overlay_cam = _render_camera(img, table, traj, px_per_cm)

    return ShotCallerResult(table, balls, cue, traj, overlay_rect, overlay_cam)


def _render_rectified(warped, balls, traj, px_per_cm):
    vis = draw_balls(warped, balls)
    if traj is not None:
        vis = draw_trajectory(vis, traj, px_per_cm)
    return vis


def _project_to_camera(points_cm, H_inv, px_per_cm):
    """Map cm points (rectified frame) back to original camera pixels."""
    if not points_cm:
        return []
    pts = np.array([[[p[0] * px_per_cm, p[1] * px_per_cm]] for p in points_cm],
                   dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H_inv)
    return [(int(round(x)), int(round(y))) for [[x, y]] in out]


def _render_camera(img, table: TableResult, traj, px_per_cm):
    """Draw the table outline and trajectory back on the original image."""
    vis = img.copy()
    cv2.polylines(vis, [table.corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    if traj is None:
        return vis

    H_inv = np.linalg.inv(table.homography)

    cue_px = _project_to_camera(traj.cue_path, H_inv, px_per_cm)
    for a, b in zip(cue_px, cue_px[1:]):
        cv2.line(vis, a, b, (0, 255, 255), 2, cv2.LINE_AA)
    for q in cue_px[1:-1]:
        cv2.circle(vis, q, 5, (0, 200, 255), -1)

    if traj.collision is not None:
        obj_px = _project_to_camera(traj.collision.object_path, H_inv, px_per_cm)
        for a, b in zip(obj_px, obj_px[1:]):
            cv2.line(vis, a, b, (0, 0, 255), 2, cv2.LINE_AA)
        contact = _project_to_camera([traj.collision.point], H_inv, px_per_cm)[0]
        cv2.circle(vis, contact, 6, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description="ShotCaller pool trajectory predictor")
    ap.add_argument("image")
    ap.add_argument("--aim", type=float, default=None,
                    help="aim angle in degrees (default: nearest object ball)")
    ap.add_argument("--felt", choices=["blue", "green"], default=None)
    ap.add_argument("--bounces", type=int, default=4)
    ap.add_argument("--out", default="shotcaller_result.png")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    res = run(img, aim_deg=args.aim, felt_color=args.felt, max_bounces=args.bounces)
    if not res.table.found:
        print("Table not detected.")
        raise SystemExit(1)

    print(f"Table felt: {res.table.felt_color}")
    print(f"Balls detected: {len(res.balls)} (cue: {'yes' if res.cue else 'no'})")
    if res.trajectory is not None:
        print(f"Cue path points: {len(res.trajectory.cue_path)}, "
              f"bounces: {res.trajectory.bounces}, "
              f"collision: {'yes' if res.trajectory.collision else 'no'}")

    cam_out = args.out
    rect_out = args.out.replace(".png", "_rectified.png")
    cv2.imwrite(cam_out, res.overlay_camera)
    cv2.imwrite(rect_out, res.overlay_rectified)
    print(f"Wrote {cam_out} and {rect_out}")


if __name__ == "__main__":
    main()
