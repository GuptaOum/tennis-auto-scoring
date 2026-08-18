"""Snapping a fitted court onto the painted lines.

The bug this addresses is invisible to reprojection error, so these tests check
against synthesised line pixels rather than against keypoints.
"""

import cv2
import numpy as np
import pytest

from tennis.court import COURT_LINES, COURT_MODEL, CourtCalibration, calibrate
from tennis.court_refine import line_mask, refine


WIDTH, HEIGHT = 1280, 720
TRUE_CORNERS = np.float32([[380, 220], [900, 220], [140, 640], [1140, 640]])


def _truth_H():
    return cv2.getPerspectiveTransform(COURT_MODEL[:4].astype(np.float32), TRUE_CORNERS)


def _painted_court():
    """A frame with the court lines drawn where they really are."""
    frame = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)   # dark surface
    pts = cv2.perspectiveTransform(
        COURT_MODEL.reshape(-1, 1, 2).astype(np.float64), _truth_H().astype(np.float64)
    ).reshape(-1, 2)
    for a, b in COURT_LINES:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def _calibration_from(corners):
    H_inv = cv2.getPerspectiveTransform(
        COURT_MODEL[:4].astype(np.float32), np.float32(corners)
    ).astype(np.float64)
    return CourtCalibration(
        H=np.linalg.inv(H_inv), H_inv=H_inv,
        reprojection_error=0.3, inlier_count=14, frame_index=0,
    )


def _worst_corner_error(calibration):
    got = calibration.to_image(COURT_MODEL[:4])
    return float(np.max(np.linalg.norm(got - TRUE_CORNERS, axis=1)))


def test_line_mask_finds_the_painted_lines_and_not_the_surface():
    mask = line_mask(_painted_court())
    assert mask.sum() > 500
    # The middle of a court quadrant carries no line.
    assert mask[520, 640] == 0


def test_line_mask_ignores_a_large_bright_object():
    """An umpire's chair or a white shirt must not read as a court line."""
    frame = _painted_court()
    cv2.rectangle(frame, (1150, 80), (1260, 260), (255, 255, 255), -1)
    mask = line_mask(frame)
    assert mask[170, 1205] == 0, "a solid bright block was taken for a line"


def test_it_pulls_a_displaced_court_back_onto_the_lines():
    """The real failure: a consistently mislocated court, low reprojection."""
    off = TRUE_CORNERS + np.float32([[25, 12], [-30, 9], [-18, -20], [22, -14]])
    start = _calibration_from(off)
    assert _worst_corner_error(start) > 15

    refined, report = refine(_painted_court(), start)
    assert report["applied"]
    # Not sub-pixel, and it cannot be: the paint is 3 px wide, so the cost is
    # exactly zero anywhere within the stripe and the fit is free to slide
    # inside it. Line width is the precision floor. What matters is the order
    # of magnitude - 31 px of displacement is removed.
    assert _worst_corner_error(refined) < 8.0
    assert _worst_corner_error(refined) < _worst_corner_error(start) / 3


def test_it_leaves_an_already_correct_fit_alone():
    exact = _calibration_from(TRUE_CORNERS)
    refined, report = refine(_painted_court(), exact)
    assert _worst_corner_error(refined) < 3.0


def test_a_frame_with_no_lines_does_not_wreck_the_calibration():
    """No evidence must mean no change, not an arbitrary new answer."""
    blank = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)
    start = _calibration_from(TRUE_CORNERS)
    refined, report = refine(blank, start)
    assert not report["applied"]
    assert refined is start


def test_the_reprojection_error_is_not_overwritten():
    """The two metrics measure different things; conflating them hid the bug."""
    off = TRUE_CORNERS + np.float32([[20, 10], [-20, 10], [-15, -15], [15, -15]])
    start = _calibration_from(off)
    refined, _ = refine(_painted_court(), start)
    assert refined.reprojection_error == pytest.approx(start.reprojection_error)
