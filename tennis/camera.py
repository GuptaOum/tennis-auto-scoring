"""Recover the camera's 3-D position from the court homography.

The homography tells us where the court plane is. It does not, on its own, tell
us where the ball is - a ball two metres in the air projects onto the plane
metres from its true position, which is the artefact that shows up as court y
of -7.1 m on real footage.

That artefact is recoverable, because its geometry is exact. If the camera
centre is C and the ball is at P = (x, y, h), then the ray C -> P meets the
court plane at

    Q = C + s (P - C),    s = Cz / (Cz - h)

so the plane-projected point Q relates to the true position by

    Q_xy = C_xy + (P_xy - C_xy) * Cz / (Cz - h)          (forward)
    P_xy = Q_xy - (h / Cz) (Q_xy - C_xy)                 (inverse)

Everything needed is C. With one plane-to-image homography and the usual
assumptions - square pixels, principal point at the image centre - the focal
length follows in closed form from the orthogonality of the rotation columns,
and from there the full pose. This is Zhang's single-plane calibration reduced
to the one unknown that matters here.

The assumptions are worth stating plainly, because they are what limits the
accuracy: square pixels, no lens distortion, principal point centred. Broadcast
cameras are close enough to all three that the residual error is small next to
the ball's own detection noise, but a fisheye or a heavily cropped frame would
break it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81


@dataclass
class Camera:
    """A calibrated view of the court, in court coordinates (metres)."""

    centre: np.ndarray        # (3,) camera position, z up from the court plane
    focal_length: float       # pixels
    principal_point: tuple[float, float]
    residual: float           # how badly the two rotation constraints disagree

    @property
    def height(self) -> float:
        """Camera height above the court plane, in metres."""
        return float(self.centre[2])

    @property
    def is_plausible(self) -> bool:
        """Whether this pose could describe a real tennis broadcast camera.

        A camera below the court, or 60 m above it, means the recovery failed -
        usually because the keypoints were poor enough that the homography is
        only approximately a perspective transform.
        """
        return 2.0 <= self.height <= 60.0 and self.residual < 0.35

    def lift(self, plane_xy: np.ndarray, height: float) -> np.ndarray:
        """Plane-projected point + known height -> true 3-D position.

        Undoes the overshoot: a ball seen projecting to ``plane_xy`` while it
        was actually ``height`` metres up was really here.
        """
        plane_xy = np.asarray(plane_xy, dtype=float)
        offset = plane_xy - self.centre[:2]
        true_xy = plane_xy - (height / self.height) * offset
        return np.array([true_xy[0], true_xy[1], height])

    def project_to_plane(self, point3d: np.ndarray) -> np.ndarray:
        """True 3-D position -> where it appears to land on the court plane.

        The forward model the trajectory fit is optimised against.
        """
        point3d = np.asarray(point3d, dtype=float)
        height = point3d[2]
        if height >= self.height:
            # At or above the camera the ray never reaches the plane.
            return np.array([np.nan, np.nan])
        scale = self.height / (self.height - height)
        return self.centre[:2] + (point3d[:2] - self.centre[:2]) * scale


def calibrate_camera(
    H_court_to_image: np.ndarray, image_width: int, image_height: int
) -> Camera:
    """Recover focal length and camera centre from the court homography.

    ``H_court_to_image`` maps court metres to image pixels - the ``H_inv`` of a
    :class:`~tennis.court.CourtCalibration`.
    """
    H = np.asarray(H_court_to_image, dtype=float)
    H = H / H[2, 2]

    cx, cy = image_width / 2.0, image_height / 2.0
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    # Orthogonality of the first two rotation columns, r1 . r2 = 0, with
    # K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], solves for f in closed form.
    a1, b1 = h1[0] - cx * h1[2], h1[1] - cy * h1[2]
    a2, b2 = h2[0] - cx * h2[2], h2[1] - cy * h2[2]
    denominator = h1[2] * h2[2]

    if abs(denominator) < 1e-12:
        # The court plane is parallel to the image plane: a true overhead shot.
        # There is no perspective foreshortening to solve for a focal length
        # from - and equally, an overhead camera barely displaces an airborne
        # ball, so the correction this class exists for is not needed.
        raise ValueError(
            "camera appears to be looking straight down - no perspective to "
            "calibrate from (and none needed)"
        )

    f_squared = -(a1 * a2 + b1 * b2) / denominator
    if f_squared <= 0:
        raise ValueError("focal length recovery failed - degenerate homography")
    focal = float(np.sqrt(f_squared))

    K_inv = np.array(
        [[1 / focal, 0, -cx / focal], [0, 1 / focal, -cy / focal], [0, 0, 1]]
    )
    b_1, b_2, b_3 = K_inv @ h1, K_inv @ h2, K_inv @ h3

    n1, n2 = np.linalg.norm(b_1), np.linalg.norm(b_2)
    if n1 < 1e-12 or n2 < 1e-12:
        raise ValueError("degenerate homography - zero-norm rotation column")

    # The second constraint, |r1| = |r2|, is not used to solve for f, so how
    # far it is from holding measures how well the model fits.
    residual = float(abs(n1 - n2) / max(n1, n2))

    scale = 2.0 / (n1 + n2)
    r1, r2 = b_1 * scale, b_2 * scale
    r3 = np.cross(r1, r2)
    t = b_3 * scale

    R = np.column_stack([r1, r2, r3])
    # Re-orthogonalise: measurement noise leaves R slightly off the rotation
    # manifold, and the nearest true rotation is the SVD projection.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt

    centre = -R.T @ t
    if centre[2] < 0:
        # Sign of the homography scale is ambiguous; the camera is above.
        centre = -centre

    return Camera(
        centre=centre,
        focal_length=focal,
        principal_point=(cx, cy),
        residual=residual,
    )
