"""Quantitative evaluation of the Stage-3 collision model.

Uses the Billiards Sports Analytics recordings: every strike has the cue ball's
path and the struck object ball's path, both in centimeters. We test the core
physics of the predictor — the "cut" / 90-degree rule that sets which way the
object ball departs.

For each strike:
  * approach direction u  = unit(cue contact point - cue start),
  * object center O       = first recorded object position,
  * predicted object dir  = line-of-centers direction at the modeled contact
                            (cue center one ball-diameter from O, along u),
  * actual object dir     = unit(first significant object displacement).

Metric: angular error between predicted and actual object directions. We also
report a naive baseline that assumes the object continues in the cue's approach
direction (a "follow" assumption), to show the geometric model adds value.

Angles are invariant to the per-video translation/scale of the Kinovea frame,
so cross-recording calibration differences do not affect the metric.

Run:
    python src/evaluate_trajectory.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from trajectory_data import iter_strikes  # noqa: E402

BALL_DIAMETER_CM = 5.7
TRAJ_ROOT = ROOT / "data" / "data_trajectories"
RESULTS = ROOT / "results"


def _unit(vx, vy):
    n = math.hypot(vx, vy)
    return (vx / n, vy / n) if n > 1e-9 else (0.0, 0.0)


def _angle_between(a, b) -> float:
    """Angle in degrees between two unit vectors."""
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(dot))


def _significant_dir(path, min_disp=2.0):
    """Direction of the first displacement exceeding ``min_disp`` cm."""
    x0, y0 = path[0]
    for x, y in path[1:]:
        if math.hypot(x - x0, y - y0) >= min_disp:
            return _unit(x - x0, y - y0)
    return _unit(path[-1][0] - x0, path[-1][1] - y0)


def _contact_index(cue_path, O, max_d):
    """Index of the cue point closest to the object center (within max_d)."""
    best_i, best_d = -1, max_d
    for i, (x, y) in enumerate(cue_path):
        d = math.hypot(x - O[0], y - O[1])
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _approach_dir(cue_path, ci):
    """Cue heading from rest position to the contact point.

    The long start->contact baseline is far more stable than an instantaneous
    finite-difference near contact, which is dominated by tracking jitter over
    the few-centimeter pre-contact window.
    """
    return _unit(cue_path[ci][0] - cue_path[0][0], cue_path[ci][1] - cue_path[0][1])


def _predicted_object_dir(cue_start, u, O, D):
    """Line-of-centers direction under the cut rule; also return miss distance.

    Returns (obj_dir, perp) where perp is the perpendicular distance from the
    cue's approach line to O (perp > D means the recorded approach would not
    actually contact the ball within a diameter).
    """
    wx, wy = O[0] - cue_start[0], O[1] - cue_start[1]
    t = wx * u[0] + wy * u[1]
    fx, fy = cue_start[0] + t * u[0], cue_start[1] + t * u[1]
    perp = math.hypot(O[0] - fx, O[1] - fy)
    perp_eff = min(perp, D * 0.999)
    back = math.sqrt(max(D * D - perp_eff * perp_eff, 0.0))
    contact = (fx - u[0] * back, fy - u[1] * back)
    return _unit(O[0] - contact[0], O[1] - contact[1]), perp


CUT_BINS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 90), (90, 181)]


def evaluate(limit: int | None = None, miss_tol: float = 1.5, plot: bool = True):
    D = BALL_DIAMETER_CM
    model_errs, base_errs, cut_mags = [], [], []
    n_total = n_used = 0

    for strike in iter_strikes(TRAJ_ROOT):
        if limit and n_total >= limit:
            break
        n_total += 1

        cue, obj = strike.cue_path, strike.object_path
        O = obj[0]

        ci = _contact_index(cue, O, max_d=D * 4)
        if ci <= 0:
            continue  # cue never approaches this object's start position
        u = _approach_dir(cue, ci)
        if u == (0.0, 0.0):
            continue

        pred_dir, perp = _predicted_object_dir(cue[0], u, O, D)
        # Skip strikes where the recorded approach clearly does not strike this
        # ball directly (combos, banks, or tracking noise): perp >> diameter.
        if perp > D * miss_tol:
            continue

        actual_dir = _significant_dir(obj)
        if actual_dir == (0.0, 0.0):
            continue

        n_used += 1
        model_errs.append(_angle_between(pred_dir, actual_dir))
        base_errs.append(_angle_between(u, actual_dir))   # "follow" baseline
        cut_mags.append(_angle_between(u, actual_dir))     # actual cut magnitude

    print("=== Trajectory (cut-rule) evaluation ===")
    print(f"Strikes scanned        : {n_total}")
    print(f"Direct-hit strikes used: {n_used}")
    if not model_errs:
        return

    me = np.array(model_errs)
    be = np.array(base_errs)
    cm = np.array(cut_mags)

    print("\nObject-ball departure angle error (degrees):")
    print(f"  cut model : mean {me.mean():5.1f}  median {np.median(me):5.1f}")
    print(f"  baseline  : mean {be.mean():5.1f}  median {np.median(be):5.1f}  (assume object follows cue)")
    print(f"\n  within 10 deg: model {np.mean(me <= 10):.1%}  vs baseline {np.mean(be <= 10):.1%}")
    print(f"  within 20 deg: model {np.mean(me <= 20):.1%}  vs baseline {np.mean(be <= 20):.1%}")

    print("\nError by shot difficulty (cut magnitude):")
    print(f"  {'cut range':>12} {'n':>5} {'model':>8} {'baseline':>9}")
    bin_centers, bin_model, bin_base = [], [], []
    for lo, hi in CUT_BINS:
        sel = (cm >= lo) & (cm < hi)
        if not np.any(sel):
            continue
        mm, bb = me[sel].mean(), be[sel].mean()
        print(f"  {f'{lo}-{hi} deg':>12} {int(sel.sum()):>5} {mm:>7.1f}  {bb:>8.1f}")
        if hi <= 90 and sel.sum() >= 5:  # skip degenerate/tiny bins in the figure
            bin_centers.append((lo + hi) / 2)
            bin_model.append(mm)
            bin_base.append(bb)

    if plot and bin_centers:
        _save_plot(cm, me, be, bin_centers, bin_model, bin_base)


def _save_plot(cm, me, be, bin_centers, bin_model, bin_base):
    RESULTS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    keep = cm <= 90
    ax1.scatter(cm[keep], me[keep], s=10, alpha=0.35, label="cut model", color="#1f77b4")
    ax1.plot([0, 90], [0, 90], "--", color="gray", lw=1, label="baseline (follow cue)")
    ax1.set_xlabel("actual cut magnitude (deg)")
    ax1.set_ylabel("object-ball direction error (deg)")
    ax1.set_title("Per-shot error vs. cut magnitude")
    ax1.set_xlim(0, 90)
    ax1.set_ylim(0, 90)
    ax1.legend(loc="upper left", fontsize=8)

    width = 3.5
    x = np.array(bin_centers)
    ax2.bar(x - width, bin_base, width * 2, label="baseline", color="#d62728", alpha=0.7)
    ax2.bar(x + width, bin_model, width * 2, label="cut model", color="#1f77b4", alpha=0.8)
    ax2.set_xlabel("cut magnitude bin (deg)")
    ax2.set_ylabel("mean direction error (deg)")
    ax2.set_title("Mean error by shot difficulty")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = RESULTS / "trajectory_cut_analysis.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--miss-tol", type=float, default=1.5,
                    help="reject strikes whose approach misses the ball by more "
                         "than miss_tol * ball diameter")
    a = ap.parse_args()
    evaluate(a.limit, a.miss_tol, plot=True)


if __name__ == "__main__":
    main()
