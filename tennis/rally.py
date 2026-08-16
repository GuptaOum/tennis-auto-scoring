"""Rally segmentation and point attribution.

This is where the project's central design decision lives: **the system never
judges a line call to decide a point.** A single camera cannot do that
reliably - Hawk-Eye needs ten calibrated ones - and a system that pretends
otherwise produces confident nonsense.

Instead a point is decided by what ended the rally, which is a far coarser and
therefore far more robust question:

  double bounce   the ball bounced twice on one side -> the other player wins
  bounced out     it bounced outside the court -> whoever hit it loses
  into the net    it never reached the other side -> whoever hit it loses

Each outcome carries a confidence derived from the events behind it. Points
the system is unsure about are reported as unsure rather than guessed, because
a measured 85% with the failures visible is worth more than an unfalsifiable
100%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tennis.bounce import BallEvent, EventType
from tennis.court import NET_Y
from tennis.scoring import Match, Player

# A rally is over when the ball has been missing this long, in seconds. Real
# rallies have sub-second gaps from missed detections; the walk back to the
# baseline between points takes several seconds.
RALLY_GAP_SECONDS = 2.0

# Below this, a point is reported but flagged for review rather than trusted.
LOW_CONFIDENCE = 0.6


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
    def is_decided(self) -> bool:
        return self.winner is not None


def _player_for_side(side: str) -> Player:
    """Player 1 defends the near half, player 2 the far half."""
    return 1 if side == "near" else 2


def _other(player: Player) -> Player:
    return 2 if player == 1 else 1


def segment(
    events: list[BallEvent], fps: float = 30.0, min_events: int = 3
) -> list[Rally]:
    """Group ball events into rallies, splitting on long silences.

    ``min_events`` drops fragments too short to be a real point - a stray pair
    of detections during a changeover should not become a rally.
    """
    if not events:
        return []

    gap_frames = int(RALLY_GAP_SECONDS * fps)
    ordered = sorted(events, key=lambda e: e.frame)

    groups: list[list[BallEvent]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if current.frame - previous.frame > gap_frames:
            groups.append([current])
        else:
            groups[-1].append(current)

    return [
        Rally(start_frame=g[0].frame, end_frame=g[-1].frame, events=g)
        for g in groups
        if len(g) >= min_events
    ]


def attribute(rally: Rally) -> Rally:
    """Decide who won a rally, and how confidently. Mutates and returns it."""
    events = rally.events
    if len(events) < 2:
        rally.reason = "too few events to judge"
        return rally

    bounces = [e for e in events if e.type is EventType.BOUNCE]
    last_hit = next(
        (e for e in reversed(events) if e.type is EventType.HIT), None
    )

    # 1. Two bounces on the same side with no hit between them: the ball was
    #    not returned. Checked first because it is the least ambiguous ending.
    for first, second in zip(bounces, bounces[1:]):
        if first.side != second.side:
            continue
        between = [
            e
            for e in events
            if first.frame < e.frame < second.frame and e.type is EventType.HIT
        ]
        if between:
            continue
        loser = _player_for_side(first.side)
        rally.winner = _other(loser)
        rally.reason = f"double bounce on the {first.side} side"
        rally.confidence = round(min(first.confidence, second.confidence), 3)
        return rally

    # 2. The ball's last bounce landed outside the singles court. Whoever hit
    #    it last put it out.
    last_bounce = bounces[-1] if bounces else None
    if last_bounce is not None and not last_bounce.in_bounds:
        if last_hit is not None and last_hit.frame < last_bounce.frame:
            loser = last_hit.by_player or _player_for_side(
                "far" if last_bounce.side == "near" else "near"
            )
            rally.winner = _other(loser)  # type: ignore[arg-type]
            rally.reason = "ball landed outside the court"
            rally.confidence = round(
                min(last_bounce.confidence, last_hit.confidence), 3
            )
            return rally

    # 3. The ball never crossed to the other side after the final hit: it went
    #    into the net.
    if last_hit is not None:
        after = [e for e in events if e.frame > last_hit.frame]
        hit_side = last_hit.side
        if after and all(e.side == hit_side for e in after):
            loser = last_hit.by_player or _player_for_side(hit_side)
            rally.winner = _other(loser)  # type: ignore[arg-type]
            rally.reason = "ball did not cross the net"
            rally.confidence = round(last_hit.confidence * 0.8, 3)
            return rally

    rally.reason = "rally ending could not be determined"
    rally.confidence = 0.0
    return rally


def score_match(
    events: list[BallEvent], fps: float = 30.0, match: Match | None = None
) -> tuple[Match, list[Rally]]:
    """Run the full chain: events -> rallies -> attributed points -> score.

    Undecided rallies are skipped rather than assigned to a player at random.
    They are still returned, so the report can show how many points the system
    could not call - which is the honest denominator for any accuracy claim.
    """
    match = match or Match()
    rallies = [attribute(r) for r in segment(events, fps=fps)]

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
