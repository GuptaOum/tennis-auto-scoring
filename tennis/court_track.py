"""Keep the court locked to the image while the camera moves.

The rest of this project fitted the court once and reused the matrix, which is
correct only for a locked-off camera. Measured on real footage that assumption
does not hold: one clip drifts ~5 px/second and contains hard cuts of over 1000
px. A single fit is stale within a second.

So the court is *tracked*, the same way the ball and the players are:

* every ``redetect_every`` frames, find it from scratch - keypoint model, then
  ``court_refine`` to snap it onto the painted lines;
* on every frame in between, carry it forward by measuring how the whole scene
  moved, which is far cheaper than re-detecting and has no per-frame jitter.

Scene motion comes from sparse optical flow over corner features, condensed
into one homography by RANSAC. That is the right model for a camera that pans,
tilts and zooms about a mostly-planar scene, and RANSAC is what stops a moving
player's features from dragging the court along with them.

A hard cut is not motion, and tracking through one is meaningless: when flow
fails or the implied jump is absurd, the tracker reports that it has lost the
court and waits for the next detection rather than emitting a confident wrong
answer.
"""

from __future__ import annotations

import cv2
import numpy as np

from tennis.court import COURT_MODEL, CourtCalibration, calibrate
from tennis.court_refine import refine

# Features to follow between frames. A few hundred is plenty for a homography
# and keeps the per-frame cost in single-digit milliseconds.
_MAX_FEATURES = 400
_FEATURE_QUALITY = 0.01
_FEATURE_MIN_DISTANCE = 8

# Re-seed once tracking attrition drops the set this low, otherwise the
# homography slowly loses its constraint as points leave the frame.
_MIN_FEATURES = 40

# A frame-to-frame corner jump larger than this is a cut, not camera motion.
_MAX_CORNER_JUMP_PX = 120.0

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def calibration_from_corners(
    corners: np.ndarray, frame_index: int, reprojection_error: float, inliers: int
) -> CourtCalibration | None:
    """Build a calibration from the four doubles corners in image pixels."""
    try:
        H_inv = cv2.getPerspectiveTransform(
            COURT_MODEL[:4].astype(np.float32), corners.reshape(4, 2).astype(np.float32)
        ).astype(np.float64)
    except cv2.error:
        return None
    if H_inv is None or abs(np.linalg.det(H_inv)) < 1e-12:
        return None
    return CourtCalibration(
        H=np.linalg.inv(H_inv),
        H_inv=H_inv,
        reprojection_error=reprojection_error,
        inlier_count=inliers,
        frame_index=frame_index,
    )


class CourtTracker:
    """Per-frame court corners: detect periodically, follow in between."""

    def __init__(
        self,
        court_model,
        redetect_every: int = 30,
        use_refine: bool = True,
    ) -> None:
        self.court_model = court_model
        self.redetect_every = max(1, int(redetect_every))
        self.use_refine = use_refine

        self.corners: np.ndarray | None = None   # (4, 2) image pixels
        self.calibration: CourtCalibration | None = None
        self._prev_grey: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._since_detect = 0
        self.stats = {"detections": 0, "tracked": 0, "lost": 0, "recovered": 0}

    def update(self, frame: np.ndarray, index: int) -> CourtCalibration | None:
        """Court for this frame, or None if it is not currently held."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        due = self.corners is None or self._since_detect >= self.redetect_every
        if due:
            found = self._detect(frame, index)
            if found:
                self._seed(grey)
                self._since_detect = 0
                self._prev_grey = grey
                return self.calibration
            # Detection failed. Fall through and try to carry the old court
            # forward rather than dropping it for one bad frame.

        moved = self._follow(grey, index)
        self._prev_grey = grey
        self._since_detect += 1
        return moved

    def _detect(self, frame: np.ndarray, index: int) -> bool:
        try:
            candidate = calibrate(self.court_model.detect(frame), frame_index=index)
        except (ValueError, cv2.error):
            return False
        if self.use_refine:
            candidate, _ = refine(frame, candidate)
        if not candidate.is_reliable:
            return False

        self.corners = candidate.to_image(COURT_MODEL[:4]).astype(np.float64)
        self.calibration = candidate
        self.stats["detections"] += 1
        if self.stats["lost"]:
            self.stats["recovered"] += 1
        return True

    def _seed(self, grey: np.ndarray) -> None:
        """Pick features to follow, over the whole frame.

        Deliberately not restricted to the court: the stands and the surrounding
        markings are rigid too, and they stay in view when a player walks across
        the baseline. Non-rigid movers are rejected by RANSAC, not by masking.
        """
        self._features = cv2.goodFeaturesToTrack(
            grey, maxCorners=_MAX_FEATURES, qualityLevel=_FEATURE_QUALITY,
            minDistance=_FEATURE_MIN_DISTANCE,
        )

    def _follow(self, grey: np.ndarray, index: int) -> CourtCalibration | None:
        if (
            self.corners is None
            or self._prev_grey is None
            or self._features is None
            or len(self._features) < _MIN_FEATURES
        ):
            return self._lose(grey)

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_grey, grey, self._features.astype(np.float32), None, **_LK
        )
        if nxt is None or status is None:
            return self._lose(grey)

        keep = status.ravel() == 1
        before = self._features.reshape(-1, 2)[keep]
        after = nxt.reshape(-1, 2)[keep]
        if len(before) < _MIN_FEATURES:
            return self._lose(grey)

        delta, mask = cv2.findHomography(before, after, cv2.RANSAC, 3.0)
        if delta is None or not np.isfinite(delta).all():
            return self._lose(grey)

        moved = cv2.perspectiveTransform(
            self.corners.reshape(-1, 1, 2), delta
        ).reshape(4, 2)

        # A cut moves everything at once; camera motion does not.
        if float(np.max(np.linalg.norm(moved - self.corners, axis=1))) > _MAX_CORNER_JUMP_PX:
            return self._lose(grey)

        calibration = calibration_from_corners(
            moved, index,
            self.calibration.reprojection_error if self.calibration else 0.0,
            self.calibration.inlier_count if self.calibration else 4,
        )
        if calibration is None:
            return self._lose(grey)

        self.corners = moved
        self.calibration = calibration
        self._features = after.reshape(-1, 1, 2)
        if len(self._features) < _MIN_FEATURES:
            self._seed(grey)
        self.stats["tracked"] += 1
        return calibration

    def _lose(self, grey: np.ndarray) -> None:
        """Drop the court and force a fresh detection next frame."""
        self.stats["lost"] += 1
        self.corners = None
        self.calibration = None
        self._features = None
        self._since_detect = self.redetect_every
        return None
