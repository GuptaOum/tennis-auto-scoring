"""Bounce and hit detection.

The baseline had one event detector: a sign change in the ball's vertical
position, which it labelled a "shot". That conflates the two things that
actually happen to a tennis ball - a racket hit and a bounce off the court -
and no scoring can be built on top of it, since a rally is precisely a
*sequence* of alternating hits and bounces.

Separating them needs to know when the ball is near the ground, and the trick
is that the homography already measures that, by way of its own error.

A homography maps the court *plane*. A ball in the air is not on that plane, so
the camera ray through it pierces the plane somewhere beyond the ball's true
position - always further from the camera, and further the higher the ball is.
So the projected court position of an airborne ball is wrong in a specific,
usable direction: **the projection error is a height measurement.**

Measured on real footage, one rally's turning points looked like this:

    frame   image y   projected court y
    f9        736         +20.5          ball on the ground, near end
    f40       311          +0.3          ball on the ground, far end
    f56       243          -4.7          apex of the arc, several metres up

The court is 0 to 23.77 m long, so -4.7 is nonsense as a position - and exactly
right as a signal. Projected court y falls as the ball rises and recovers as it
descends, which makes:

    local maximum in projected court y   ->  ball is at its lowest: ground
                                             contact, so a bounce or a hit
    local minimum                        ->  apex of the flight: not an event

Raw image y cannot do this job, because in a perspective view it confounds
height with depth - a ball high in the frame may be high in the air or simply
far away. Projected court y separates the two.

Ground-contact events are then split by asking whether a player was within
racket reach, measured **in image space**: an airborne ball's court coordinate
is metres from where it really is, so court-space proximity is meaningless,
while a ball beside a player in the image really is beside them. Distance is
normalised by the player's bounding-box height, so one threshold serves a near
player 400 px tall and a far player 90 px tall.

Every event carries a confidence, so an ambiguous one degrades a point's
confidence instead of silently deciding it.
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


def _turning_points(
    values: np.ndarray, min_prominence: float, kind: str = "both"
) -> list[int]:
    """Indices where a 1-D signal reverses direction, ignoring small wobbles.

    ``kind`` selects ``"max"`` (peaks), ``"min"`` (valleys) or ``"both"``.

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
        wanted = {"max": is_peak, "min": is_valley, "both": is_peak or is_valley}[kind]
        if not wanted:
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
    player_boxes: dict[int, dict[int, tuple[float, float, float, float]]]
    | None = None,
    fps: float = 30.0,
    min_ground_prominence: float = 1.5,
    hit_reach: float = 0.8,
    line_margin: float = 0.10,
) -> list[BallEvent]:
    """Find bounces and hits along a ball trajectory.

    ``player_boxes`` maps ``{track_id: {frame: (x1, y1, x2, y2)}}`` in image
    pixels. Without it, every event is classified as a bounce, since hits are
    only distinguishable by proximity to a player.

    ``min_ground_prominence`` is how far projected court y must swing, in
    metres, for a descent-then-recovery to count as ground contact rather than
    detector jitter.

    ``hit_reach`` is how close the ball must be to a player, in multiples of
    that player's bounding-box height. A player's box is roughly their real
    height, so 0.8 covers an outstretched arm plus a racket (~1.4 m on a 1.8 m
    player). Normalising by box height rather than using a fixed pixel radius
    is what lets one threshold serve a near player 400 px tall and a far player
    90 px tall.
    """
    events: list[BallEvent] = []
    # Events cannot be closer together than a ball can physically travel; at
    # 30 fps a fifth of a second is a safe floor for hit-bounce-hit.
    min_separation = max(2, int(fps / 6))

    for run in trajectory.segments(max_gap=1):
        if len(run) < 5:
            continue

        # Projected court y as a height proxy: it dips while the ball is up and
        # peaks when the ball comes back down to the plane. Maxima are ground
        # contact; minima are the apex of a flight and are not events at all.
        height_signal = np.array([s.court[1] for s in run])
        contacts = _dedupe(
            _turning_points(height_signal, min_ground_prominence, kind="max"),
            min_separation,
        )

        for index in contacts:
            sample = run[index]
            nearest_player, distance = _nearest_player(
                sample.image, sample.frame, player_boxes
            )
            near_a_player = distance is not None and distance <= hit_reach

            if near_a_player:
                kind = EventType.HIT
                # Closer to the player is a more convincing racket strike.
                confidence = sample.confidence * (
                    1.0 - min(distance / hit_reach, 1.0) * 0.3
                )
            else:
                kind = EventType.BOUNCE
                confidence = sample.confidence

            if sample.interpolated:
                confidence *= 0.6

            # A bounce lies on the court plane by definition, so its projected
            # position has to be near the court. Far outside means the ball was
            # still airborne and this reading cannot support an in/out call.
            if kind is EventType.BOUNCE and not _plausible_landing(sample.court):
                confidence *= 0.4

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
    ball_image: np.ndarray,
    frame: int,
    player_boxes: dict[int, dict[int, tuple[float, float, float, float]]] | None,
) -> tuple[int | None, float | None]:
    """Closest player to the ball in the image, in box-height units.

    Returns ``(track_id, distance)`` where 1.0 means "one player height away".
    Distance is to the nearest edge of the box rather than its centre, so a
    ball at the feet and a ball at the head both read as close.
    """
    if not player_boxes:
        return None, None

    best_id, best_distance = None, float("inf")
    for track_id, by_frame in player_boxes.items():
        box = by_frame.get(frame)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        height = max(y2 - y1, 1.0)
        dx = max(x1 - ball_image[0], 0.0, ball_image[0] - x2)
        dy = max(y1 - ball_image[1], 0.0, ball_image[1] - y2)
        distance = float(np.hypot(dx, dy) / height)
        if distance < best_distance:
            best_id, best_distance = track_id, distance

    if best_id is None:
        return None, None
    return best_id, best_distance


def _plausible_landing(court_pt: np.ndarray) -> bool:
    """Whether a court-space point could be a real landing spot.

    Generous - 4 m of run-off past every line - because this only needs to
    catch the airborne-projection artefact, which overshoots by far more.
    """
    x, y = float(court_pt[0]), float(court_pt[1])
    return -4.0 <= x <= 14.97 and -4.0 <= y <= 27.77


def crossed_net(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether a path between two court points passed over the net line."""
    return (before[1] - NET_Y) * (after[1] - NET_Y) < 0
