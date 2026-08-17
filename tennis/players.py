"""Telling the players apart from everyone else on camera.

The person detector finds people, not players. On any real broadcast that also
means ball kids at the net posts, line judges along the tramlines, the chair
umpire, coaches, and photographers behind the baseline - a measured six tracks
on a Vienna ATP clip where two people were playing.

Left unfiltered this is not merely untidy, it is wrong. Every one of those
tracks flows into distance covered, average speed, net approaches and the
coverage heatmaps, so a ball kid jogging to the post is reported as a player
covering ground. It also breaks hit attribution, because a ball passing near a
line judge looks like a shot struck by one.

Two facts about singles tennis do the work, and neither is a tuned threshold:

1. **Players are on the court.** Everyone else is beside or behind it. A player
   ranges a few metres wide and a few metres behind the baseline, so distance
   from the court is a strong discriminator once positions are in metres -
   which the homography already gives us.
2. **There is exactly one player per side of the net.** That kills the
   remaining hard case, a ball kid standing behind the baseline near a real
   player: they compete only against that player, and lose on distance.

Court frame: x 0 to 10.97 across, y 0 to 23.77 from the far baseline, net at
11.885.
"""

from __future__ import annotations

import numpy as np

from tennis.court import COURT_LENGTH, DOUBLES_WIDTH, NET_Y, CourtCalibration
from tennis.detect import Detection

# How far outside the doubles court a player may still be. Generous on purpose:
# a wide serve is returned from well outside the tramline, and players run
# several metres behind the baseline to retrieve a deep ball. These bound who
# is *plausibly* playing; the per-side contest decides who actually is.
MAX_WIDE_M = 3.0
MAX_BEHIND_M = 5.0


def court_distance(court_point: np.ndarray) -> float:
    """Metres from the doubles court, 0 for a point inside it."""
    x, y = float(court_point[0]), float(court_point[1])
    dx = max(0.0, -x, x - DOUBLES_WIDTH)
    dy = max(0.0, -y, y - COURT_LENGTH)
    return float(np.hypot(dx, dy))


def is_plausible(court_point: np.ndarray) -> bool:
    """Could someone standing here be playing this point?"""
    x, y = float(court_point[0]), float(court_point[1])
    return (
        -MAX_WIDE_M <= x <= DOUBLES_WIDTH + MAX_WIDE_M
        and -MAX_BEHIND_M <= y <= COURT_LENGTH + MAX_BEHIND_M
    )


def select(
    detections: list[Detection], calibration: CourtCalibration | None
) -> list[Detection]:
    """The at-most-two people actually playing, one per side of the net.

    Without a calibration there is no court to measure against, so the two
    largest boxes are returned - players are nearer the camera than the crowd,
    and it is the best available guess. Every caller treats this as a guess:
    nothing downstream claims a metre until a calibration exists.
    """
    if calibration is None:
        return sorted(detections, key=_box_area, reverse=True)[:2]

    scored: list[tuple[Detection, np.ndarray, float]] = []
    for detection in detections:
        try:
            court = calibration.to_court(detection.feet)
        except Exception:  # noqa: BLE001 - a bad matrix must not drop everyone
            continue
        if not is_plausible(court):
            continue
        scored.append((detection, court, court_distance(court)))

    chosen: list[Detection] = []
    for near_side in (False, True):
        side = [
            item for item in scored
            if (float(item[1][1]) >= NET_Y) == near_side
        ]
        if not side:
            continue
        # Closest to the court wins; a larger box breaks a tie, which favours
        # the player over someone standing further from the camera behind them.
        best = min(side, key=lambda item: (round(item[2], 2), -_box_area(item[0])))
        chosen.append(best[0])
    return chosen


def _box_area(detection: Detection) -> float:
    x1, y1, x2, y2 = detection.bbox
    return abs((x2 - x1) * (y2 - y1))
