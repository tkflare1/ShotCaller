"""Stage 3: Trajectory prediction and visualization.

All physics is computed in centimeters on the rectified table surface, where
the geometry is a simple axis-aligned rectangle. The model covers:

  * Rail bounces via the law of reflection (angle in = angle out), with the
    ball center kept one radius from each cushion.
  * The first ball-to-ball collision along the cue's path, using the standard
    "90-degree rule" of pool: the object ball departs along the line of centers
    and the cue ball deflects roughly perpendicular to it.

The output is a list of polyline segments (in cm) plus the collision event, all
renderable on the rectified image and projectable back to the camera view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

BALL_DIAMETER_CM = 5.7
BALL_RADIUS_CM = BALL_DIAMETER_CM / 2.0


@dataclass
class Collision:
    """Records the first cue/object-ball contact along a predicted shot."""

    point: tuple[float, float]            # contact point of cue center, cm
    object_ball: tuple[float, float]      # struck ball center, cm
    object_path: list[tuple[float, float]]  # predicted object-ball polyline, cm


@dataclass
class Trajectory:
    """Predicted cue-ball path and any resulting collision."""

    cue_path: list[tuple[float, float]] = field(default_factory=list)
    collision: Collision | None = None
    bounces: int = 0


def _ray_rect_hit(p, d, bounds):
    """First intersection of ray p + t d (t>0) with the inset rectangle.

    bounds = (xmin, ymin, xmax, ymax) for the *ball center*. Returns
    (t, point, axis) where axis is 'x' or 'y' indicating which rail was hit.
    """
    xmin, ymin, xmax, ymax = bounds
    px, py = p
    dx, dy = d
    best_t, best_axis = math.inf, None

    if dx > 1e-9:
        t = (xmax - px) / dx
        if 0 < t < best_t:
            best_t, best_axis = t, "x"
    elif dx < -1e-9:
        t = (xmin - px) / dx
        if 0 < t < best_t:
            best_t, best_axis = t, "x"
    if dy > 1e-9:
        t = (ymax - py) / dy
        if 0 < t < best_t:
            best_t, best_axis = t, "y"
    elif dy < -1e-9:
        t = (ymin - py) / dy
        if 0 < t < best_t:
            best_t, best_axis = t, "y"

    if best_axis is None:
        return None
    hit = (px + best_t * dx, py + best_t * dy)
    return best_t, hit, best_axis


def _first_ball_hit(p, d, obstacles, contact_dist, max_t):
    """Closest obstacle center whose ball the cue contacts before max_t.

    A contact occurs when the cue center passes within ``contact_dist`` (the
    sum of radii) of an obstacle center. Returns (t, center) or None.
    """
    px, py = p
    dx, dy = d
    best = None
    for (ox, oy) in obstacles:
        # Project obstacle center onto the ray.
        t = (ox - px) * dx + (oy - py) * dy
        if t <= 1e-6 or t > max_t:
            continue
        cx, cy = px + t * dx, py + t * dy
        perp = math.hypot(ox - cx, oy - cy)
        if perp <= contact_dist:
            # Back off to the moment of first contact (cue center position).
            back = math.sqrt(max(contact_dist ** 2 - perp ** 2, 0.0))
            t_contact = t - back
            if t_contact > 1e-6 and (best is None or t_contact < best[0]):
                best = (t_contact, (ox, oy))
    return best


def predict_trajectory(
    cue_cm: tuple[float, float],
    direction_deg: float,
    table_cm: tuple[float, float],
    obstacles_cm: list[tuple[float, float]] | None = None,
    max_bounces: int = 4,
    ball_radius_cm: float = BALL_RADIUS_CM,
) -> Trajectory:
    """Predict the cue-ball path from a position and aim direction.

    Parameters
    ----------
    cue_cm : (x, y) cue-ball center in cm.
    direction_deg : aim angle in degrees (0 = +x, CCW positive).
    table_cm : (length_x, width_y) of the playing surface in cm.
    obstacles_cm : other ball centers in cm.
    max_bounces : how many rail reflections to simulate.
    """
    L, W = table_cm
    r = ball_radius_cm
    bounds = (r, r, L - r, W - r)
    obstacles = list(obstacles_cm or [])
    contact_dist = 2.0 * r

    ang = math.radians(direction_deg)
    d = (math.cos(ang), math.sin(ang))
    p = (float(np.clip(cue_cm[0], r, L - r)), float(np.clip(cue_cm[1], r, W - r)))

    traj = Trajectory(cue_path=[p])

    for _ in range(max_bounces + 1):
        rail = _ray_rect_hit(p, d, bounds)
        if rail is None:
            break
        t_rail, rail_pt, axis = rail

        ball = _first_ball_hit(p, d, obstacles, contact_dist, t_rail)
        if ball is not None:
            t_c, obj_center = ball
            contact_pt = (p[0] + t_c * d[0], p[1] + t_c * d[1])
            traj.cue_path.append(contact_pt)

            # Object ball departs along the line of centers (contact -> object).
            ldir = (obj_center[0] - contact_pt[0], obj_center[1] - contact_pt[1])
            n = math.hypot(*ldir) or 1.0
            ldir = (ldir[0] / n, ldir[1] / n)
            obj_path = _roll_to_rail(obj_center, ldir, bounds)
            traj.collision = Collision(contact_pt, obj_center, obj_path)
            break

        # Otherwise reflect off the rail and continue.
        traj.cue_path.append(rail_pt)
        traj.bounces += 1
        if axis == "x":
            d = (-d[0], d[1])
        else:
            d = (d[0], -d[1])
        p = rail_pt

    return traj


def _roll_to_rail(start, d, bounds):
    """Straight object-ball path from ``start`` until the first rail."""
    hit = _ray_rect_hit(start, d, bounds)
    if hit is None:
        return [start]
    _, pt, _ = hit
    return [start, pt]


def _cm_to_px(pt, px_per_cm):
    return (int(round(pt[0] * px_per_cm)), int(round(pt[1] * px_per_cm)))


def draw_trajectory(warped, traj: Trajectory, px_per_cm: float) -> np.ndarray:
    """Render a trajectory onto the rectified table image."""
    vis = warped.copy()

    pts = [_cm_to_px(p, px_per_cm) for p in traj.cue_path]
    for a, b in zip(pts, pts[1:]):
        cv2.line(vis, a, b, (0, 255, 255), 2, cv2.LINE_AA)
    for q in pts[1:-1]:
        cv2.circle(vis, q, 5, (0, 200, 255), -1)  # rail-bounce markers

    if traj.collision is not None:
        c = traj.collision
        cv2.circle(vis, _cm_to_px(c.point, px_per_cm), 6, (0, 0, 255), 2, cv2.LINE_AA)
        opath = [_cm_to_px(p, px_per_cm) for p in c.object_path]
        for a, b in zip(opath, opath[1:]):
            cv2.line(vis, a, b, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    # Self-contained demo on a synthetic table (no image needed).
    table = (254.0, 127.0)
    cue = (60.0, 60.0)
    obstacles = [(180.0, 95.0)]

    t = predict_trajectory(cue, 25.0, table, obstacles, max_bounces=4)
    print(f"cue path ({t.bounces} bounces):")
    for p in t.cue_path:
        print(f"  ({p[0]:6.1f}, {p[1]:6.1f})")
    if t.collision:
        print("collision with object ball at", t.collision.object_ball)

    canvas = np.full((int(127 * 3), int(254 * 3), 3), (120, 60, 0), np.uint8)
    for ox, oy in obstacles:
        cv2.circle(canvas, (int(ox * 3), int(oy * 3)), int(BALL_RADIUS_CM * 3),
                   (0, 215, 255), -1)
    cv2.circle(canvas, (int(cue[0] * 3), int(cue[1] * 3)), int(BALL_RADIUS_CM * 3),
               (255, 255, 255), -1)
    cv2.imwrite("trajectory_demo.png", draw_trajectory(canvas, t, 3.0))
    print("wrote trajectory_demo.png")
