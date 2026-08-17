"""Serve and fault detection.

Without this the scoring is wrong in a specific, silent way: a fault looks
exactly like a lost rally. The ball is struck, it bounces somewhere it should
not, play stops - and the point gets awarded to the wrong player. A double
fault reads as one lost point instead of one, and a first serve out reads as a
point conceded that was never conceded at all.

A serve is identifiable without any new detection work:

  it is the first hit of a rally
  it is struck from behind, or close to, the server's baseline
  it must land in the service box diagonally opposite

Only the last is a rule. The first two identify *which* stroke to judge.

What is deliberately not attempted:

  lets      a serve clipping the net cord changes the ball's path by a few
            centimetres over one or two frames. At 30 fps with a ball detected
            to a few pixels, that signal is not there. Lets are invisible here
            and will read as ordinary serves.
  foot faults
            these need the server's feet resolved against the baseline to
            within a couple of centimetres, across motion blur, at the far end
            of the court. Not attainable from this footage.

Both are stated rather than approximated, because a wrong fault call costs a
whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tennis.bounce import BallEvent, EventType
from tennis.court import (
    ALLEY_WIDTH,
    COURT_LENGTH,
    NET_Y,
    SERVICE_LINE_FROM_NET,
    SINGLES_WIDTH,
)
from tennis.rally import Rally
from tennis.scoring import Player

CENTRE_X = ALLEY_WIDTH + SINGLES_WIDTH / 2
FAR_SERVICE_Y = NET_Y - SERVICE_LINE_FROM_NET    # 5.485
NEAR_SERVICE_Y = NET_Y + SERVICE_LINE_FROM_NET   # 18.285

# How close to their own baseline a player must be for a first hit to read as
# a serve, as a fraction of the net-to-baseline distance. Servers stand at or
# just behind the baseline; 0.8 leaves room for the position estimate.
SERVE_DEPTH_FRACTION = 0.8

# Line calls are generous by this margin, in metres. A serve judged out by two
# centimetres is not a call this system should be making.
SERVICE_LINE_MARGIN = 0.10


class ServeOutcome(Enum):
    IN = "in"
    FAULT = "fault"
    UNKNOWN = "unknown"      # no landing found; refuse to call it


@dataclass
class Serve:
    rally_index: int
    frame: int
    server: Player | None
    outcome: ServeOutcome
    box: str | None            # service box it landed in, if any
    expected_box: str | None   # box it should have landed in
    landing: np.ndarray | None
    confidence: float
    is_second_serve: bool = False

    def as_dict(self) -> dict:
        return {
            "rally_index": self.rally_index,
            "frame": self.frame,
            "server": self.server,
            "outcome": self.outcome.value,
            "box": self.box,
            "expected_box": self.expected_box,
            "second_serve": self.is_second_serve,
            "confidence": round(self.confidence, 3),
            "landing_m": (
                [round(float(self.landing[0]), 2), round(float(self.landing[1]), 2)]
                if self.landing is not None
                else None
            ),
        }


def service_box(court_pt: np.ndarray, margin: float = SERVICE_LINE_MARGIN) -> str | None:
    """Which service box a landing falls in, or None if it is outside them all.

    Boxes are named by the half they sit in and the side of the centre line, as
    seen from behind that half's baseline: ``far-deuce``, ``far-ad``,
    ``near-deuce``, ``near-ad``.
    """
    x, y = float(court_pt[0]), float(court_pt[1])

    if not (ALLEY_WIDTH - margin <= x <= ALLEY_WIDTH + SINGLES_WIDTH + margin):
        return None

    if FAR_SERVICE_Y - margin <= y <= NET_Y + margin:
        half = "far"
        # Seen from behind the far baseline, left and right are mirrored
        # relative to the global x axis.
        side = "deuce" if x < CENTRE_X else "ad"
    elif NET_Y - margin <= y <= NEAR_SERVICE_Y + margin:
        half = "near"
        side = "deuce" if x > CENTRE_X else "ad"
    else:
        return None

    return f"{half}-{side}"


def expected_box(server: Player, point_number: int) -> str:
    """Which box this serve should land in.

    Player 1 defends the near half and serves into the far one. Serving starts
    from the deuce court and alternates every point, so the parity of the
    point number within the game determines the target.
    """
    receiving_half = "far" if server == 1 else "near"
    side = "deuce" if point_number % 2 == 0 else "ad"
    return f"{receiving_half}-{side}"


def _serve_hit(rally: Rally) -> BallEvent | None:
    return next((e for e in rally.events if e.type is EventType.HIT), None)


def _first_bounce_after(rally: Rally, frame: int) -> BallEvent | None:
    return next(
        (
            e
            for e in rally.events
            if e.type is EventType.BOUNCE and e.frame > frame
        ),
        None,
    )


def _struck_from_baseline(
    position: np.ndarray | None, server_half: str
) -> bool:
    """Whether a court position is deep enough to be a serving stance."""
    if position is None:
        return False
    y = float(position[1])
    depth = (NET_Y - y) if server_half == "far" else (y - NET_Y)
    return depth >= SERVE_DEPTH_FRACTION * (COURT_LENGTH / 2)


def classify_serve(
    rally: Rally,
    rally_index: int,
    point_number: int,
    player_positions: dict[int, dict[int, np.ndarray]] | None = None,
    is_second_serve: bool = False,
) -> Serve | None:
    """Judge the opening stroke of a rally. None if it has no identifiable serve."""
    hit = _serve_hit(rally)
    if hit is None:
        return None

    server: Player | None = 1 if hit.side == "near" else 2
    landing_event = _first_bounce_after(rally, hit.frame)

    if landing_event is None:
        return Serve(
            rally_index=rally_index,
            frame=hit.frame,
            server=server,
            outcome=ServeOutcome.UNKNOWN,
            box=None,
            expected_box=expected_box(server, point_number),
            landing=None,
            confidence=0.0,
            is_second_serve=is_second_serve,
        )

    box = service_box(landing_event.court)
    target = expected_box(server, point_number)

    # A serve is in if it landed in a service box on the receiver's side. The
    # deuce/ad distinction is reported but not enforced: it depends on knowing
    # the point count exactly, and one missed point earlier in the clip would
    # otherwise turn every subsequent serve into a fault.
    receiving_half = target.split("-")[0]
    outcome = (
        ServeOutcome.IN
        if box is not None and box.startswith(receiving_half)
        else ServeOutcome.FAULT
    )

    confidence = landing_event.confidence
    # A serve struck from mid-court is more likely a misidentified rally start
    # than a real serve, so the call is worth less.
    position = None
    if player_positions and hit.by_player is not None:
        by_frame = player_positions.get(hit.by_player, {})
        position = by_frame.get(hit.frame)
    if not _struck_from_baseline(position, hit.side):
        confidence *= 0.6

    return Serve(
        rally_index=rally_index,
        frame=hit.frame,
        server=server,
        outcome=outcome,
        box=box,
        expected_box=target,
        landing=landing_event.court,
        confidence=confidence,
        is_second_serve=is_second_serve,
    )


def analyse_serves(
    rallies: list[Rally],
    player_positions: dict[int, dict[int, np.ndarray]] | None = None,
) -> tuple[list[Serve], list[int], list[tuple[int, Player]]]:
    """Classify every serve and resolve faults into their scoring consequences.

    Returns ``(serves, fault_rally_indices, double_faults)`` where

    - ``fault_rally_indices`` are rallies that were a first-serve fault and so
      must **not** be scored as a lost point - the biggest single scoring error
      this module removes
    - ``double_faults`` are ``(rally_index, winner)`` pairs where two faults in
      a row hand the point to the receiver
    """
    serves: list[Serve] = []
    faults: list[int] = []
    double_faults: list[tuple[int, Player]] = []

    consecutive_faults = 0
    point_number = 0

    for index, rally in enumerate(rallies):
        serve = classify_serve(
            rally,
            index,
            point_number,
            player_positions,
            is_second_serve=consecutive_faults == 1,
        )
        if serve is None:
            point_number += 1
            continue

        serves.append(serve)

        if serve.outcome is ServeOutcome.FAULT:
            consecutive_faults += 1
            if consecutive_faults >= 2:
                # Double fault: the receiver takes the point.
                receiver: Player = 2 if serve.server == 1 else 1
                double_faults.append((index, receiver))
                consecutive_faults = 0
                point_number += 1
            else:
                # A first-serve fault ends no point. Scoring must skip it.
                faults.append(index)
        else:
            consecutive_faults = 0
            point_number += 1

    return serves, faults, double_faults


def summarise(serves: list[Serve], double_faults: list[tuple[int, Player]]) -> dict:
    in_play = [s for s in serves if s.outcome is ServeOutcome.IN]
    faulted = [s for s in serves if s.outcome is ServeOutcome.FAULT]
    firsts = [s for s in serves if not s.is_second_serve]
    first_in = [s for s in firsts if s.outcome is ServeOutcome.IN]

    return {
        "serves_detected": len(serves),
        "in": len(in_play),
        "faults": len(faulted),
        "double_faults": len(double_faults),
        "unknown": sum(1 for s in serves if s.outcome is ServeOutcome.UNKNOWN),
        "first_serve_percentage": (
            round(len(first_in) / len(firsts), 3) if firsts else None
        ),
        "boxes": {
            box: sum(1 for s in in_play if s.box == box)
            for box in sorted({s.box for s in in_play if s.box})
        },
        "serves": [s.as_dict() for s in serves],
    }
