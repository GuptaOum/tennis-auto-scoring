"""Rally segmentation and point attribution.

This is where the project's central design decision lives: **the system never
judges a line call to decide a point.** A single camera cannot do that
reliably - Hawk-Eye needs ten calibrated ones - and a system that pretends
otherwise produces confident nonsense.

A point is decided instead by what *ended* the rally, a far coarser and
therefore far more robust question:

  double bounce   the ball bounced twice on one side -> the other player wins
  bounced out     it bounced outside the court -> whoever hit it loses
  into the net    it never reached the other side -> whoever hit it loses

Segmentation follows from the same idea. An earlier version split the event
stream wherever there was a two-second silence, on the assumption that the ball
goes untracked between points. That assumption died when ball detection went
from 47% to 98%: with the ball visible almost continuously, the silences
vanished and a whole clip collapsed into one rally.

The fix is to stop inferring boundaries from absence and read them from
structure. A rally ends at the event that ends it - the second bounce, the
bounce landing out, the ball that never crosses. Those are the same events that
decide the point, so segmentation and attribution are one pass, and a rally
boundary can never disagree with the reason a point was awarded.

A time gap is still honoured as a fallback, because a genuinely lost ball does
mean play stopped. It is now the exception rather than the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tennis.bounce import BallEvent, EventType
from tennis.scoring import Match, Player

# A rally is abandoned when the ball has been missing this long, in seconds.
# Only a fallback now: real rally ends are detected from their terminal event.
RALLY_GAP_SECONDS = 2.5

# Below this, a point is reported but flagged for review rather than trusted.
LOW_CONFIDENCE = 0.6

# A rally needs at least this many events before it is worth judging. Two
# stray detections during a changeover should not become a point.
MIN_EVENTS = 2


@dataclass
class Rally:
    start_frame: int
    end_frame: int
    events: list[BallEvent] = field(default_factory=list)
    winner: Player | None = None
    reason: str = ""
    confidence: float = 0.0

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def shot_count(self) -> int:
        return sum(1 for e in self.events if e.type is EventType.HIT)

    @property
    def bounce_count(self) -> int:
        return sum(1 for e in self.events if e.type is EventType.BOUNCE)

    @property
    def is_decided(self) -> bool:
        return self.winner is not None

    @property
    def serve_frame(self) -> int | None:
        """The first hit of the rally - the serve."""
        first = next((e for e in self.events if e.type is EventType.HIT), None)
        return first.frame if first else None


def _player_for_side(side: str) -> Player:
    """Player 1 defends the near half, player 2 the far half."""
    return 1 if side == "near" else 2


def _other(player: Player) -> Player:
    return 2 if player == 1 else 1


def _terminal_outcome(events: list[BallEvent]) -> tuple[Player, str, float] | None:
    """Does this event sequence end with a point? If so, to whom and why.

    Checked against the *tail* of the sequence, so a rally closes on the event
    that ended it rather than at some later silence.
    """
    if len(events) < 2:
        return None

    last = events[-1]

    # 1. Two bounces on the same side with no hit between them: not returned.
    #    Checked first, being the least ambiguous ending in tennis.
    if last.type is EventType.BOUNCE:
        previous_bounce = None
        for event in reversed(events[:-1]):
            if event.type is EventType.HIT:
                break
            if event.type is EventType.BOUNCE:
                previous_bounce = event
                break
        if previous_bounce is not None and previous_bounce.side == last.side:
            loser = _player_for_side(last.side)
            return (
                _other(loser),
                f"double bounce on the {last.side} side",
                round(min(previous_bounce.confidence, last.confidence), 3),
            )

    # 2. The ball bounced outside the singles court. Whoever hit it last put it
    #    out - so the ball must have been struck before it landed.
    if last.type is EventType.BOUNCE and not last.in_bounds:
        last_hit = next(
            (e for e in reversed(events[:-1]) if e.type is EventType.HIT), None
        )
        if last_hit is not None:
            loser = last_hit.by_player or _player_for_side(
                "far" if last.side == "near" else "near"
            )
            return (
                _other(loser),  # type: ignore[arg-type]
                "ball landed outside the court",
                round(min(last.confidence, last_hit.confidence), 3),
            )

    # 3. A hit, then a bounce on the hitter's own side: it never crossed.
    if last.type is EventType.BOUNCE:
        last_hit = next(
            (e for e in reversed(events[:-1]) if e.type is EventType.HIT), None
        )
        if last_hit is not None and last_hit.side == last.side:
            after_hit = [e for e in events if e.frame > last_hit.frame]
            if all(e.side == last_hit.side for e in after_hit):
                loser = last_hit.by_player or _player_for_side(last_hit.side)
                return (
                    _other(loser),  # type: ignore[arg-type]
                    "ball did not cross the net",
                    round(last_hit.confidence * 0.8, 3),
                )

    return None


def segment(events: list[BallEvent], fps: float = 30.0) -> list[Rally]:
    """Split the event stream into rallies, closing each at its ending.

    One pass: accumulate events, and whenever the tail of the accumulator forms
    a point-ending pattern, close the rally there and begin the next. Because
    the same tail supplies the winner and the reason, a rally boundary can
    never disagree with the point it produced.
    """
    if not events:
        return []

    gap_frames = int(RALLY_GAP_SECONDS * fps)
    ordered = sorted(events, key=lambda e: e.frame)

    rallies: list[Rally] = []
    current: list[BallEvent] = []

    def close(decided: tuple[Player, str, float] | None) -> None:
        if len(current) < MIN_EVENTS:
            current.clear()
            return
        rally = Rally(
            start_frame=current[0].frame,
            end_frame=current[-1].frame,
            events=list(current),
        )
        if decided is not None:
            rally.winner, rally.reason, rally.confidence = decided
        else:
            rally.reason = "rally ending could not be determined"
        rallies.append(rally)
        current.clear()

    for event in ordered:
        # A long silence means play stopped without a readable ending.
        if current and event.frame - current[-1].frame > gap_frames:
            close(_terminal_outcome(current))

        current.append(event)

        outcome = _terminal_outcome(current)
        if outcome is not None:
            close(outcome)

    if current:
        close(_terminal_outcome(current))

    return rallies


def attribute(rally: Rally) -> Rally:
    """Re-derive a rally's outcome from its events. Mutates and returns it.

    :func:`segment` already attributes as it goes; this exists so a rally built
    by hand - in a test, or from edited events - can be judged on its own.
    """
    outcome = _terminal_outcome(rally.events)
    if outcome is None:
        rally.winner = None
        rally.confidence = 0.0
        rally.reason = rally.reason or "rally ending could not be determined"
    else:
        rally.winner, rally.reason, rally.confidence = outcome
    return rally


def score_match(
    events: list[BallEvent], fps: float = 30.0, match: Match | None = None
) -> tuple[Match, list[Rally]]:
    """Run the full chain: events -> rallies -> attributed points -> score.

    Undecided rallies are skipped rather than assigned to a player at random.
    They are still returned, so the report can show how many points could not
    be called - the honest denominator for any accuracy claim.
    """
    match = match or Match()
    rallies = segment(events, fps=fps)

    for rally in rallies:
        if rally.winner is None or match.is_over:
            continue
        match.award_point(
            rally.winner,
            reason=rally.reason,
            confidence=rally.confidence,
            start_frame=rally.start_frame,
            end_frame=rally.end_frame,
        )
    return match, rallies


def summarise(rallies: list[Rally], fps: float = 30.0) -> dict:
    decided = [r for r in rallies if r.is_decided]
    uncertain = [r for r in decided if r.confidence < LOW_CONFIDENCE]
    return {
        "rallies_found": len(rallies),
        "points_decided": len(decided),
        "points_undecided": len(rallies) - len(decided),
        "low_confidence_points": len(uncertain),
        "mean_confidence": (
            round(float(np.mean([r.confidence for r in decided])), 3)
            if decided
            else 0.0
        ),
        "mean_shots_per_rally": (
            round(float(np.mean([r.shot_count for r in rallies])), 1)
            if rallies
            else 0.0
        ),
        "longest_rally_shots": max((r.shot_count for r in rallies), default=0),
        "mean_rally_seconds": (
            round(float(np.mean([r.duration_frames for r in rallies])) / fps, 1)
            if rallies
            else 0.0
        ),
        "reasons": {
            reason: sum(1 for r in decided if r.reason == reason)
            for reason in sorted({r.reason for r in decided})
        },
    }
