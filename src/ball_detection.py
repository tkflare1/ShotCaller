"""Stage 2: Ball detection and classification.

Operates on the *rectified* (bird's-eye) table image produced by Stage 1, where
balls appear as roughly constant-size circles and metric scale is known.

Pipeline
--------
  1. Hough circle transform locates ball candidates. The expected radius is
     derived from the known ball diameter (5.7 cm) and the rectified scale.
  2. Each candidate is filtered against the felt: a real ball patch is *not*
     dominated by felt-colored pixels.
  3. Each ball is classified as cue (white), 8-ball (black), or a colored ball,
     and colored balls are further split into solid vs. stripe by how much
     white rim is present.

The returned coordinates are given both in rectified pixels and in centimeters
on the table surface, so Stage 3 can reason about physics directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from table_detection import TABLE_LENGTH_CM, TABLE_WIDTH_CM

BALL_DIAMETER_CM = 5.7  # regulation pool ball

# Hue centers (OpenCV H in [0,180]) for the standard ball colors. Used only for
# a human-readable color name; trajectory logic only needs cue vs. the rest.
COLOR_HUES = {
    "yellow": 25,
    "orange": 12,
    "red": 0,
    "purple": 145,
    "blue": 110,
    "green": 65,
    "maroon": 5,
}


@dataclass
class Ball:
    """A detected ball on the rectified table."""

    x: float            # rectified-pixel center
    y: float
    r: float            # rectified-pixel radius
    x_cm: float         # table-surface center, centimeters
    y_cm: float
    kind: str           # "cue", "eight", "solid", "stripe"
    color: str          # human-readable color name ("white", "black", ...)

    @property
    def is_cue(self) -> bool:
        return self.kind == "cue"


def _circle_mask(patch_shape, r_frac=0.6):
    """Boolean mask selecting the inner disk of a square patch."""
    h, w = patch_shape[:2]
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr = min(h, w) / 2.0 * r_frac
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= rr * rr


def _classify_patch(patch: np.ndarray) -> tuple[str, str]:
    """Classify a square BGR ball patch -> (kind, color_name)."""
    mask = _circle_mask(patch.shape)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][mask].astype(np.float32)
    s = hsv[..., 1][mask].astype(np.float32)
    v = hsv[..., 2][mask].astype(np.float32)
    if v.size == 0:
        return "unknown", "unknown"

    # White: bright, low saturation. Black: dark.
    white_frac = float(np.mean((s < 60) & (v > 150)))
    black_frac = float(np.mean(v < 60))

    if white_frac > 0.55:
        return "cue", "white"
    if black_frac > 0.5:
        return "eight", "black"

    # Colored ball: name it from the median hue of saturated pixels, excluding
    # the felt-blue band so cloth leaking into the patch doesn't dominate.
    colored = (s > 70) & (v > 60)
    not_felt = (h < 86) | (h > 120)
    sel = colored & not_felt
    if not np.any(sel):
        sel = colored if np.any(colored) else np.ones_like(colored)
    hue_med = float(np.median(h[sel]))
    color = min(COLOR_HUES, key=lambda c: min(
        abs(hue_med - COLOR_HUES[c]), 180 - abs(hue_med - COLOR_HUES[c])
    ))

    # Stripe vs solid: stripes carry a substantial white band alongside color.
    kind = "stripe" if white_frac > 0.18 else "solid"
    return kind, color


def detect_balls(
    table,
    img: np.ndarray | None = None,
    hough_param2: int = 14,
) -> list[Ball]:
    """Detect and classify balls on a detected table's rectified image.

    Working in the rectified frame means every ball has the same expected pixel
    radius (set by the known 5.7 cm diameter and the rectified scale), which
    makes the Hough radius window tight and reliable.

    Parameters
    ----------
    table : TableResult from :func:`table_detection.detect_table`.
    img   : unused; accepted so callers may pass the source frame.
    hough_param2 : Hough accumulator threshold (lower = more, noisier circles).
    """
    if not table.found or table.warped is None:
        return []

    warped = table.warped
    px_per_cm = table.px_per_cm
    felt_color = table.felt_color or "blue"

    expected_r = 0.5 * BALL_DIAMETER_CM * px_per_cm
    r_min = max(4, int(expected_r * 0.6))
    r_max = int(expected_r * 1.5)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=int(expected_r * 1.2),
        param1=100, param2=hough_param2, minRadius=r_min, maxRadius=r_max,
    )
    if circles is None:
        return []

    lo, hi = _felt_hsv(felt_color)
    hsv_full = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    H, W = warped.shape[:2]

    balls: list[Ball] = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(W, cx + r), min(H, cy + r)
        patch = warped[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        # Reject candidates dominated by felt (false circles on the cloth).
        patch_hsv = hsv_full[y0:y1, x0:x1]
        felt_bin = cv2.inRange(patch_hsv, np.array(lo), np.array(hi)) > 0
        if float(np.mean(felt_bin)) > 0.55:
            continue
        # Also reject if the ball *core* is mostly felt (circle on bare cloth).
        core = _circle_mask(patch.shape, r_frac=0.5)
        if core.shape == felt_bin.shape and float(np.mean(felt_bin[core])) > 0.5:
            continue

        kind, color = _classify_patch(patch)
        balls.append(
            Ball(
                x=float(cx), y=float(cy), r=float(r),
                x_cm=cx / px_per_cm, y_cm=cy / px_per_cm,
                kind=kind, color=color,
            )
        )

    return _dedupe(balls, px_per_cm)


def _dedupe(balls: list[Ball], px_per_cm: float) -> list[Ball]:
    """Drop near-duplicate detections (Hough often fires twice per ball)."""
    min_sep = BALL_DIAMETER_CM * 0.7 * px_per_cm
    kept: list[Ball] = []
    for b in balls:
        if all((b.x - k.x) ** 2 + (b.y - k.y) ** 2 > min_sep ** 2 for k in kept):
            kept.append(b)
    return kept


def _felt_hsv(felt_color: str):
    from table_detection import FELT_HSV_RANGES
    return FELT_HSV_RANGES.get(felt_color, FELT_HSV_RANGES["blue"])


# Per-kind overlay colors (BGR).
_DRAW_COLORS = {
    "cue": (255, 255, 255),
    "eight": (40, 40, 40),
    "solid": (0, 215, 255),
    "stripe": (0, 140, 255),
    "unknown": (200, 200, 200),
}


def draw_balls(warped: np.ndarray, balls: list[Ball]) -> np.ndarray:
    """Overlay detected balls with kind/color labels on the rectified image."""
    vis = warped.copy()
    for b in balls:
        c = _DRAW_COLORS.get(b.kind, (200, 200, 200))
        cv2.circle(vis, (int(b.x), int(b.y)), int(b.r), c, 2, cv2.LINE_AA)
        cv2.circle(vis, (int(b.x), int(b.y)), 2, c, -1)
        label = b.kind if b.kind in ("cue", "eight") else f"{b.kind[:1]}:{b.color}"
        cv2.putText(vis, label, (int(b.x - b.r), int(b.y - b.r - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from table_detection import detect_table

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("usage: python ball_detection.py <image_path>")
        raise SystemExit(1)

    img = cv2.imread(path)
    table = detect_table(img)
    if not table.found:
        print("table not found")
        raise SystemExit(1)

    balls = detect_balls(table, img)
    print(f"detected {len(balls)} balls:")
    for b in balls:
        print(f"  {b.kind:6s} {b.color:7s} at ({b.x_cm:5.1f}, {b.y_cm:5.1f}) cm")
    cv2.imwrite("balls_overlay.png", draw_balls(table.warped, balls))
    print("wrote balls_overlay.png")
