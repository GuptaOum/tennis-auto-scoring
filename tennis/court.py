"""Court geometry: image pixels <-> real-world court coordinates.

The baseline project mapped players onto its mini-court by finding the nearest
court keypoint and scaling by the player's pixel height, using hardcoded real
heights for the two specific pros in its sample video. That is not a projection,
it is an approximation that breaks for any other player and cannot localise a
ball in the air at all.

This module does the real thing: a homography from the 14 detected court
keypoints to a metric model of a tennis court. Once that matrix exists, any
image point on the court plane converts to metres, which is what bounce
localisation and in/out judgement need.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ITF court dimensions, in metres.
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
COURT_LENGTH = 23.77
SERVICE_LINE_FROM_NET = 6.40
ALLEY_WIDTH = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2  # 1.37

# The 14 keypoints the ResNet50 detector predicts, in the order it emits them,
# expressed in court coordinates: origin at the far-left doubles corner, x to
# the right along the baseline, y down the court towards the near baseline.
#
#   0 ---------------- 1      far baseline      (y = 0)
#   |  8 --- 12 --- 9  |      far service line  (y = 6.485)
#   |                  |      net               (y = 11.885)
#   | 10 --- 13 --- 11 |      near service line (y = 17.285)
#   2 ---------------- 3      near baseline     (y = 23.77)
#
# 4/5 and 6/7 are the singles sideline intersections with the two baselines.
_FAR, _NEAR = 0.0, COURT_LENGTH
_FAR_SERVICE = COURT_LENGTH / 2 - SERVICE_LINE_FROM_NET   # 5.485
_NEAR_SERVICE = COURT_LENGTH / 2 + SERVICE_LINE_FROM_NET  # 18.285
_LEFT_S, _RIGHT_S = ALLEY_WIDTH, ALLEY_WIDTH + SINGLES_WIDTH
_CENTRE = DOUBLES_WIDTH / 2

COURT_MODEL = np.array(
    [
        [0.0, _FAR],                  # 0  far  doubles corner, left
        [DOUBLES_WIDTH, _FAR],        # 1  far  doubles corner, right
        [0.0, _NEAR],                 # 2  near doubles corner, left
        [DOUBLES_WIDTH, _NEAR],       # 3  near doubles corner, right
        [_LEFT_S, _FAR],              # 4  far  singles corner, left
        [_LEFT_S, _NEAR],             # 5  near singles corner, left
        [_RIGHT_S, _FAR],             # 6  far  singles corner, right
        [_RIGHT_S, _NEAR],            # 7  near singles corner, right
        [_LEFT_S, _FAR_SERVICE],      # 8  far  service line, left
        [_RIGHT_S, _FAR_SERVICE],     # 9  far  service line, right
        [_LEFT_S, _NEAR_SERVICE],     # 10 near service line, left
        [_RIGHT_S, _NEAR_SERVICE],    # 11 near service line, right
        [_CENTRE, _FAR_SERVICE],      # 12 far  centre service mark
        [_CENTRE, _NEAR_SERVICE],     # 13 near centre service mark
    ],
    dtype=np.float64,
)

NET_Y = COURT_LENGTH / 2


@dataclass
class CourtCalibration:
    """A fitted image <-> court mapping, with the evidence for trusting it."""

    H: np.ndarray            # image -> court (metres)
    H_inv: np.ndarray        # court -> image
    reprojection_error: float  # mean px error, keypoints round-tripped
    inlier_count: int
    frame_index: int

    @property
    def is_reliable(self) -> bool:
        """Whether downstream geometry should be trusted.

        4 px mean reprojection on a 1080p broadcast frame is roughly 5 cm on
        court, which is well inside what rally-outcome scoring needs. Anything
        worse usually means the keypoint detector lost the court.
        """
        return self.reprojection_error < 4.0 and self.inlier_count >= 8

    def to_court(self, points: np.ndarray) -> np.ndarray:
        """Image pixels -> court metres. Accepts (2,) or (N, 2)."""
        return _apply(self.H, points)

    def to_image(self, points: np.ndarray) -> np.ndarray:
        """Court metres -> image pixels. Accepts (2,) or (N, 2)."""
        return _apply(self.H_inv, points)


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = pts.reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return out[0] if single else out


def calibrate(keypoints: np.ndarray, frame_index: int = 0) -> CourtCalibration:
    """Fit the homography from one frame's 14 keypoints.

    ``keypoints`` is the flat 28-vector the court detector emits (x0, y0, x1,
    y1, ...) or an already-shaped (14, 2) array.

    RANSAC rather than a plain least-squares fit: the keypoint regressor has no
    confidence output, so a badly-placed point has to be rejected by geometry
    or it drags the whole mapping with it.

    The fit runs court -> image, not image -> court, and is then inverted. That
    direction matters: RANSAC's threshold is expressed in the units of the
    destination points, and the measurement noise being rejected is the
    keypoint detector's, which is in pixels. Fitting the other way would make
    the threshold a distance in metres and quietly admit every outlier.
    """
    image_pts = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if image_pts.shape[0] != len(COURT_MODEL):
        raise ValueError(
            f"expected {len(COURT_MODEL)} keypoints, got {image_pts.shape[0]}"
        )

    H_inv, mask = cv2.findHomography(
        COURT_MODEL, image_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if H_inv is None or abs(np.linalg.det(H_inv)) < 1e-12:
        raise ValueError("homography fit failed - court keypoints are degenerate")

    H = np.linalg.inv(H_inv)

    # Error is measured in pixels, not metres, so the number stays comparable
    # across camera distances: project the known court model into the image and
    # compare against where the detector said the keypoints were.
    projected = _apply(H_inv, COURT_MODEL)
    residuals = np.linalg.norm(projected - image_pts, axis=1)

    # Averaged over inliers only. Including a rejected keypoint would let one
    # bad detection mark an otherwise sound calibration as unreliable, which is
    # exactly the failure RANSAC was used to avoid.
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(14, bool)
    error = float(np.mean(residuals[inliers])) if inliers.any() else float("inf")

    return CourtCalibration(
        H=H,
        H_inv=H_inv,
        reprojection_error=error,
        inlier_count=int(inliers.sum()),
        frame_index=frame_index,
    )


def is_inside_singles(court_pt: np.ndarray, margin: float = 0.0) -> bool:
    """Whether a court-space point is inside the singles court."""
    x, y = float(court_pt[0]), float(court_pt[1])
    return (_LEFT_S - margin <= x <= _RIGHT_S + margin) and (
        _FAR - margin <= y <= _NEAR + margin
    )


def side_of_net(court_pt: np.ndarray) -> str:
    """Which half a court-space point lies in: 'far' or 'near'."""
    return "far" if float(court_pt[1]) < NET_Y else "near"


COURT_LINES = [
    (0, 1), (2, 3), (0, 2), (1, 3),   # doubles perimeter
    (4, 5), (6, 7),                    # singles sidelines
    (8, 9), (10, 11),                  # service lines
    (12, 13),                          # centre service line
]
