"""Stage 1: Table detection and homography.

Given a pool-table image (perspective view), this module:
  1. Segments the felt by HSV color thresholding. Blue felt (tournament
     footage) is the default; green felt is also supported.
  2. Finds the felt's outer quadrilateral via contour approximation, with a
     Hough-line corner-estimation fallback for noisy masks.
  3. Computes a homography that warps the perspective view into a rectified
     top-down (bird's-eye) image with a known pixels-per-cm scale.

A regulation 9-foot pool table has a 254 x 127 cm playing surface (2:1), which
sets the aspect ratio of the rectified output.

Example
-------
    import cv2
    from table_detection import detect_table

    img = cv2.imread("data/Billiard Pool.v5i.coco/train/<some>.jpg")
    result = detect_table(img)
    if result.found:
        cv2.imwrite("warped.png", result.warped)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Regulation 9-foot table playing surface, in centimeters (length x width).
TABLE_LENGTH_CM = 254.0
TABLE_WIDTH_CM = 127.0

# HSV ranges (OpenCV: H in [0,180], S/V in [0,255]).
# Tournament felt is a bright, saturated cyan-blue; the lower S/V bounds keep
# out dim background lighting and the dark crowd.
FELT_HSV_RANGES = {
    "blue": ((88, 110, 90), (115, 255, 255)),
    "green": ((35, 60, 60), (85, 255, 255)),
}


@dataclass
class TableResult:
    """Output of :func:`detect_table`."""

    found: bool
    corners: np.ndarray | None        # (4, 2) float32, ordered TL, TR, BR, BL
    homography: np.ndarray | None     # (3, 3) maps source px -> rectified px
    warped: np.ndarray | None         # rectified bird's-eye image
    mask: np.ndarray | None           # felt segmentation mask (uint8 0/255)
    px_per_cm: float                  # rectified-image scale
    felt_color: str | None            # which felt range matched


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(s)],      # TL: smallest x+y
            pts[np.argmin(diff)],   # TR: smallest y-x
            pts[np.argmax(s)],      # BR: largest x+y
            pts[np.argmax(diff)],   # BL: largest y-x
        ],
        dtype=np.float32,
    )


def segment_felt(img: np.ndarray, felt_color: str | None = None):
    """Return (mask, color_name) for the felt region.

    If ``felt_color`` is None, both blue and green are tried and the one with
    the larger response is kept.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    candidates = [felt_color] if felt_color else list(FELT_HSV_RANGES)

    best_mask, best_color, best_score = None, None, -1
    for color in candidates:
        lo, hi = FELT_HSV_RANGES[color]
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        score = int(mask.sum())
        if score > best_score:
            best_mask, best_color, best_score = mask, color, score

    # Clean the mask: drop speckle, then close small gaps within the felt.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # A strong opening severs thin "necks" that connect the felt to background
    # blue (stage lights, doorways, other tables) before component selection.
    big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, big, iterations=1)

    # Keep only the largest connected component — the felt is one big blob.
    best_mask = _largest_component(best_mask)
    return best_mask, best_color


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Zero out everything except the largest connected white region."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    # Label 0 is background; pick the foreground label with the most pixels.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    # Fill holes (balls, cue, reflections sitting on the felt).
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, cv2.FILLED)
    return out


def _quad_from_contour(mask: np.ndarray) -> np.ndarray | None:
    """Largest contour approximated to a 4-point polygon, else convex-hull min-rect."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.02 * mask.shape[0] * mask.shape[1]:
        return None  # too small to be the table

    peri = cv2.arcLength(largest, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(largest, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Fallback: minimum-area rotated rectangle around the hull.
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    return box.astype(np.float32)


def detect_table(
    img: np.ndarray,
    felt_color: str | None = None,
    px_per_cm: float = 4.0,
) -> TableResult:
    """Detect the table and compute the rectifying homography.

    Parameters
    ----------
    img : BGR uint8 image.
    felt_color : "blue", "green", or None to auto-pick.
    px_per_cm : resolution of the rectified output.
    """
    mask, color = segment_felt(img, felt_color)
    quad = _quad_from_contour(mask)
    if quad is None:
        return TableResult(False, None, None, None, mask, px_per_cm, color)

    corners = _order_corners(quad)

    out_w = int(round(TABLE_LENGTH_CM * px_per_cm))
    out_h = int(round(TABLE_WIDTH_CM * px_per_cm))

    # The longer side of the detected quad should map to the table length, so
    # orient the destination rectangle to match (avoids a 90-degree rotation).
    tl, tr, br, bl = corners
    top_len = np.linalg.norm(tr - tl)
    left_len = np.linalg.norm(bl - tl)
    if left_len > top_len:
        out_w, out_h = out_h, out_w

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h))

    return TableResult(True, corners, H, warped, mask, px_per_cm, color)


def draw_detection(img: np.ndarray, result: TableResult) -> np.ndarray:
    """Overlay the detected table quadrilateral and corners on ``img``."""
    vis = img.copy()
    if not result.found or result.corners is None:
        cv2.putText(vis, "table not found", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    pts = result.corners.astype(int)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(vis, "TL TR BR BL".split()[i], (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("usage: python table_detection.py <image_path>")
        raise SystemExit(1)

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    res = detect_table(img)
    print(f"found={res.found} felt={res.felt_color}")
    if res.found:
        print("corners (TL,TR,BR,BL):\n", res.corners)
        cv2.imwrite("table_overlay.png", draw_detection(img, res))
        cv2.imwrite("table_warped.png", res.warped)
        print("wrote table_overlay.png and table_warped.png")
