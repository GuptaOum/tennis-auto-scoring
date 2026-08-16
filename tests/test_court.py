"""Homography tests.

Built on a synthetic camera: take the known court model, project it through a
made-up perspective matrix to get "detected" pixel keypoints, then check that
calibrate() recovers the mapping. This tests the geometry without needing a
video or the keypoint model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tennis.court import (
    COURT_LENGTH,
    COURT_MODEL,
    DOUBLES_WIDTH,
    calibrate,
    is_inside_singles,
    side_of_net,
)


def synthetic_camera() -> np.ndarray:
    """A plausible broadcast view: court -> image, elevated behind a baseline."""
    src = np.array(
        [[0, 0], [DOUBLES_WIDTH, 0], [DOUBLES_WIDTH, COURT_LENGTH], [0, COURT_LENGTH]],
        dtype=np.float32,
    )
    # Far baseline appears narrow and high, near baseline wide and low.
    dst = np.array(
        [[780, 260], [1140, 260], [1560, 940], [360, 940]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(src, dst)


def project(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(pts, np.float64).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)


@pytest.fixture
def detected_keypoints() -> np.ndarray:
    return project(synthetic_camera(), COURT_MODEL)


class TestCalibration:
    def test_recovers_court_coordinates(self, detected_keypoints):
        calib = calibrate(detected_keypoints)
        recovered = calib.to_court(detected_keypoints)
        # 1 mm. RANSAC refines iteratively, so an exact float match is not the
        # bar - being correct well below the precision anything downstream
        # needs is.
        assert np.allclose(recovered, COURT_MODEL, atol=1e-3)

    def test_perfect_input_is_reliable(self, detected_keypoints):
        calib = calibrate(detected_keypoints)
        assert calib.reprojection_error < 0.01  # sub-pixel, by a wide margin
        assert calib.inlier_count == 14
        assert calib.is_reliable

    def test_round_trip_image_to_court_and_back(self, detected_keypoints):
        calib = calibrate(detected_keypoints)
        point = np.array([960.0, 600.0])
        assert np.allclose(calib.to_image(calib.to_court(point)), point, atol=1e-6)

    def test_accepts_flat_28_vector(self, detected_keypoints):
        flat = detected_keypoints.flatten()
        assert np.allclose(calibrate(flat).H, calibrate(detected_keypoints).H)

    def test_survives_one_badly_placed_keypoint(self, detected_keypoints):
        """RANSAC should reject an outlier rather than let it skew the fit.

        This is the case that matters in practice: the keypoint regressor emits
        no confidence, so a single bad point has to be caught by geometry.
        """
        corrupted = detected_keypoints.copy()
        corrupted[7] += np.array([220.0, -160.0])
        calib = calibrate(corrupted)

        good = [i for i in range(14) if i != 7]
        recovered = calib.to_court(detected_keypoints[good])
        assert np.allclose(recovered, COURT_MODEL[good], atol=0.05)  # 5 cm

    def test_rejects_wrong_keypoint_count(self):
        with pytest.raises(ValueError, match="expected 14"):
            calibrate(np.zeros((10, 2)))

    def test_rejects_degenerate_keypoints(self):
        with pytest.raises(ValueError):
            calibrate(np.zeros((14, 2)))

    def test_measures_real_distance(self, detected_keypoints):
        """The point of all this: pixel distances become metres."""
        calib = calibrate(detected_keypoints)
        baseline_px = detected_keypoints[[0, 1]]
        far, near = calib.to_court(baseline_px)
        assert np.linalg.norm(far - near) == pytest.approx(DOUBLES_WIDTH, abs=1e-6)


class TestCourtRegions:
    def test_inside_and_outside_singles(self):
        assert is_inside_singles(np.array([DOUBLES_WIDTH / 2, COURT_LENGTH / 4]))
        assert not is_inside_singles(np.array([0.5, COURT_LENGTH / 4]))  # in alley
        assert not is_inside_singles(np.array([DOUBLES_WIDTH / 2, -1.0]))  # long

    def test_margin_allows_line_calls_to_be_generous(self):
        just_wide = np.array([1.30, 10.0])  # 7 cm outside the singles sideline
        assert not is_inside_singles(just_wide)
        assert is_inside_singles(just_wide, margin=0.10)

    def test_side_of_net(self):
        assert side_of_net(np.array([5.0, 2.0])) == "far"
        assert side_of_net(np.array([5.0, 20.0])) == "near"
