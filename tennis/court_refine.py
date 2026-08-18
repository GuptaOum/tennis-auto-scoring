"""Snap a fitted court onto the lines actually painted on the ground.

Why this exists
---------------
The keypoint model regresses 14 points from a squashed low-resolution crop, and
it mislocates them *consistently*: measured across two clips, the corners sat a
median 14 px and 68 px away from the nearest painted line while the reprojection
error held steady at 0.77 px and every frame reported ``is_reliable``.

That is not a contradiction. Reprojection asks only whether a single homography
explains the model's own 14 points. A model wrong in a mutually consistent -
that is, projective - way scores perfectly on it. So reprojection cannot see
this class of error at all, and neither can anything built on top of it.

The court's own painted lines are the missing reference. They are in every
frame, they need no labels, and they are the thing downstream geometry actually
cares about: ``line_margin`` for in/out calls is 0.10 m, roughly 5-10 px here,
so a court fit 15-70 px out makes those calls arbitrary.

Method
------
Take the keypoint fit as a starting guess, then move the four court corners
until the *whole* projected line network - baselines, sidelines, service lines,
centre line - lies on top of detected line pixels. Ten lines constrain eight
degrees of freedom, so the fit is over-determined and cannot simply collapse
onto one strong edge.
"""

from __future__ import annotations

import cv2
import numpy as np

from tennis.court import COURT_LINES, COURT_MODEL, CourtCalibration

# Court lines are thin and brighter than the surface they are painted on. A
# top-hat larger than the line width keeps exactly that and discards broad
# bright regions - the umpire's chair, a white shirt, sunlit stands - which is
# what makes this work on grass, clay and hard court without a per-surface
# threshold.
_TOPHAT_KERNEL = 13
_TOPHAT_FLOOR = 18

# Samples per court line. Dense enough that a line cannot slip between samples,
# cheap enough that a fit is a few milliseconds.
_SAMPLES_PER_LINE = 40

# Robust-cost saturation, in pixels, applied coarse-then-fine. The first pass
# has to reach an initialisation that may be 70 px out; the second stops a
# distant blob of unrelated white from dragging the fit.
_SCALES = (60.0, 25.0, 8.0)

# A refinement is only accepted if it improves alignment by at least this
# fraction. Prevents a marginal, noise-driven change replacing a decent fit.
_MIN_IMPROVEMENT = 0.05


def line_mask(frame: np.ndarray) -> np.ndarray:
    """Pixels that look like painted court lines: thin, bright, unsaturated."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_TOPHAT_KERNEL, _TOPHAT_KERNEL)
    )
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    unsaturated = hsv[..., 1] < 90

    mask = (tophat > _TOPHAT_FLOOR) & unsaturated
    return mask.astype(np.uint8)


def _distance_field(mask: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the nearest line pixel."""
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def _sample_points(lines=COURT_LINES) -> np.ndarray:
    """Points spread along the court model's lines, in court metres."""
    out = []
    t = np.linspace(0.0, 1.0, _SAMPLES_PER_LINE)[:, None]
    for a, b in lines:
        out.append(COURT_MODEL[a] * (1 - t) + COURT_MODEL[b] * t)
    return np.vstack(out)


def _alignment_cost(
    corners: np.ndarray, court_pts: np.ndarray, field: np.ndarray, scale: float
) -> float:
    """Mean saturating distance from projected line samples to real line pixels.

    Saturating rather than raw distance: a raw mean lets one badly-lit corner
    or an occluded baseline dominate and pull the whole court off the lines it
    *can* see. The saturation makes far samples cost a bounded amount, so the
    fit is decided by the lines it fits well.
    """
    try:
        H_inv = cv2.getPerspectiveTransform(
            COURT_MODEL[:4].astype(np.float32), corners.reshape(4, 2).astype(np.float32)
        )
    except cv2.error:
        return 1.0
    if H_inv is None or not np.isfinite(H_inv).all():
        return 1.0

    pts = cv2.perspectiveTransform(
        court_pts.reshape(-1, 1, 2).astype(np.float64), H_inv
    ).reshape(-1, 2)

    h, w = field.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    # A mask with no line pixels at all makes the distance transform enormous;
    # squaring it overflows float32 and turns the cost into NaN.
    d = np.minimum(field[y, x].astype(np.float64), 1e4)

    # Samples projected off-frame carry the full penalty rather than the
    # distance at the clamped pixel, else shoving the court off-screen would
    # look cheap.
    off = (pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)
    cost = (d * d) / (d * d + scale * scale)
    cost = np.where(off, 1.0, cost)
    return float(np.mean(cost))


def refine(
    frame: np.ndarray,
    calibration: CourtCalibration,
    fit_lines=COURT_LINES,
) -> tuple[CourtCalibration, dict]:
    """Nudge a calibration until its lines sit on the painted ones.

    Returns the refined calibration and a report. The original is returned
    unchanged when refinement does not clearly improve alignment, so this can
    never make a good fit worse.

    ``fit_lines`` exists for honest evaluation: fitting on a subset lets the
    held-out lines measure the result, which a metric computed over the lines
    that were optimised cannot do.
    """
    from scipy.optimize import minimize

    field = _distance_field(line_mask(frame))
    court_pts = _sample_points(fit_lines)
    start = calibration.to_image(COURT_MODEL[:4]).astype(np.float64).ravel()

    best = start.copy()
    for scale in _SCALES:
        result = minimize(
            _alignment_cost, best, args=(court_pts, field, scale),
            method="Powell",
            options={"xtol": 0.05, "ftol": 1e-4, "maxiter": 4000},
        )
        if np.isfinite(result.fun):
            best = result.x

    fine = _SCALES[-1]
    before = _alignment_cost(start, court_pts, field, fine)
    after = _alignment_cost(best, court_pts, field, fine)

    report = {
        "cost_before": round(before, 4),
        "cost_after": round(after, 4),
        "corner_shift_px": round(
            float(np.max(np.linalg.norm(
                (best - start).reshape(4, 2), axis=1))), 1
        ),
        "applied": False,
    }
    if not (after < before * (1.0 - _MIN_IMPROVEMENT)):
        return calibration, report

    H_inv = cv2.getPerspectiveTransform(
        COURT_MODEL[:4].astype(np.float32), best.reshape(4, 2).astype(np.float32)
    ).astype(np.float64)
    if abs(np.linalg.det(H_inv)) < 1e-12:
        return calibration, report

    report["applied"] = True
    return (
        CourtCalibration(
            H=np.linalg.inv(H_inv),
            H_inv=H_inv,
            reprojection_error=calibration.reprojection_error,
            inlier_count=calibration.inlier_count,
            frame_index=calibration.frame_index,
        ),
        report,
    )
