"""Bounce and hit detection.

The baseline had one event detector: a sign change in the ball's vertical
position, which it labelled a "shot". That conflates the two things that
actually happen to a tennis ball, because both reverse its direction - a racket
hit and a bounce off the court. Its own stats are wrong for that reason, and no
scoring is possible on top of it, since a rally is precisely a *sequence* of
alternating hits and bounces.

This module separates them, using two signals the baseline could not access
because it never had a real homography:

  vertical  - image-space y. The camera is above the court, so a ball falling
              towards the ground moves down the frame. A bounce is a local
              maximum in image y: descent reverses to ascent.
  depth     - court-space y, in metres. A racket hit reverses the ball's travel
              *along* the court, sending it back towards the other end. A
              bounce does not.

A turning point that reverses vertically while the ball is far from both
players is a bounce. One that reverses in depth near a player is a hit. Both
carry a confidence, so an ambiguous event degrades a point's confidence instead
of silently deciding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tennis.court import NET_Y, is_inside_singles, side_of_net
from tennis.trajectory import Trajectory


class EventType(Enum):
    BOUNCE = "bounce"
    HIT = "hit"


@dataclass
class BallEvent:
    type: EventType
    frame: int
    court: np.ndarray          # position in metres when it happened
    confidence: float
    side: str                  # 'far' or 'near' - which half of the court
    in_bounds: bool            # only meaningful for bounces
    by_player: int | None = None   # only meaningful for hits

    def __repr__(self) -> str:
        where = "in" if self.in_bounds else "out"
        return (
            f"<{self.type.value} f{self.frame} {self.side} {where} "
            f"({self.court[0]:.1f}, {self.court[1]:.1f})m c={self.confidence:.2f}>"
        )


def _turning_points(values: np.ndarray, min_prominence: float) -> list[int]:
    """Indices where a 1-D signal reverses direction, ignoring small wobbles.

    ``min_prominence`` is how far the signal must travel away from the turning
    point, on both sides, before the reversal counts. Without it, detector
    jitter of a pixel or two registers as dozens of events per second.
    """
    if len(values) < 3:
        return []

    out: list[int] = []
    for i in range(1, len(values) - 1):
        before, here, after = values[i - 1], values[i], values[i + 1]
        is_peak = here >= before and here >= after
        is_valley = here <= before and here <= after
        if not (is_peak or is_valley):
            continue

        # Walk outwards until the signal has moved far enough to be convincing.
        left = right = 0.0
        for j in range(i - 1, -1, -1):
            left = max(left, abs(values[j] - here))
            if left >= min_prominence:
                break
        for j in range(i + 1, len(values)):
            right = max(right, abs(values[j] - here))
            if right >= min_prominence:
                break

        if min(left, right) >= min_prominence:
            out.append(i)
    return out


def _dedupe(indices: list[int], min_separation: int) -> list[int]:
    """Collapse turning points that are too close together to be distinct."""
    kept: list[int] = []
    for index in indices:
        if not kept or index - kept[-1] >= min_separation:
            kept.append(index)
    return kept


def detect_events(
    trajectory: Trajectory,
    player_positions: dict[int, dict[int, np.ndarray]] | None = None,
    fps: float = 30.0,
    min_vertical_prominence: float = 6.0,
    hit_radius: float = 2.5,
    line_margin: float = 0.10,
) -> list[BallEvent]:
    """Find bounces and hits along a ball trajectory.

    ``player_positions`` maps ``{player_id: {frame: court_xy}}``. Without it,
    every event is classified as a bounce, since hits are only distinguishable
    by proximity to a player.

    ``hit_radius`` is how close (in metres) the ball must be to a player for a
    turning point to read as a racket strike. 2.5 m covers an arm plus a racket
    plus the error in a foot-position estimate.
    """
    events: list[BallEvent] = []
    # Events cannot be closer together than a ball can physically travel; at
    # 30 fps a fifth of a second is a safe floor for hit-bounce-hit.
    min_separation = max(2, int(fps / 6))

    for run in trajectory.segments(max_gap=1):
        if len(run) < 5:
            continue

        image_y = np.array([s.image[1] for s in run])
        court_y = np.array([s.court[1] for s in run])

        vertical = _dedupe(
            _turning_points(image_y, min_vertical_prominence), min_separation
        )
        # Depth prominence in metres: a shot crosses most of a half-court, so
        # 2 m of travel is a low bar that still rejects jitter.
        depth = _dedupe(_turning_points(court_y, 2.0), min_separation)

        for index in sorted(set(vertical) | set(depth)):
            sample = run[index]
            nearest_player, distance = _nearest_player(
                sample.court, sample.frame, player_positions
            )

            reversed_depth = index in depth
            near_a_player = distance is not None and distance <= hit_radius

            if reversed_depth and near_a_player:
                kind = EventType.HIT
                # A hit is most credible when both signals agree: the ball
                # turned around in depth *and* someone was there to do it.
                confidence = sample.confidence * (
                    1.0 - min(distance / hit_radius, 1.0) * 0.3
                )
            elif index in vertical and not near_a_player:
                kind = EventType.BOUNCE
                confidence = sample.confidence
            else:
                # Ambiguous: a vertical turn right beside a player, or a depth
                # turn in open court. Record it as the more likely of the two
                # but mark it down, so any point resting on it is flagged.
                kind = EventType.HIT if near_a_player else EventType.BOUNCE
                confidence = sample.confidence * 0.5

            if sample.interpolated:
                confidence *= 0.6

            events.append(
                BallEvent(
                    type=kind,
                    frame=sample.frame,
                    court=sample.court,
                    confidence=round(float(min(confidence, 1.0)), 3),
                    side=side_of_net(sample.court),
                    in_bounds=is_inside_singles(sample.court, margin=line_margin),
                    by_player=nearest_player if kind is EventType.HIT else None,
                )
            )

    return events


def _nearest_player(
    ball_court: np.ndarray,
    frame: int,
    player_positions: dict[int, dict[int, np.ndarray]] | None,
) -> tuple[int | None, float | None]:
    if not player_positions:
        return None, None
    best_id, best_distance = None, float("inf")
    for player_id, by_frame in player_positions.items():
        position = by_frame.get(frame)
        if position is None:
            continue
        distance = float(np.linalg.norm(np.asarray(position) - ball_court))
        if distance < best_distance:
            best_id, best_distance = player_id, distance
    if best_id is None:
        return None, None
    return best_id, best_distance


def crossed_net(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether a path between two court points passed over the net line."""
    return (before[1] - NET_Y) * (after[1] - NET_Y) < 0
