"""Choosing which detection is the ball, using the whole flight to decide.

The detector returns several boxes per frame. ``BallDetector`` keeps the
highest-confidence one, which is a reasonable rule for a large, distinct object
and a poor one for this object. A tennis ball on a broadcast wide shot is about
ten pixels across, motion-blurred to a smear whenever it is travelling fast,
and it competes for the detector's attention with a court covered in bright
white lines of a similar width. When the ball is hardest to see - which is
exactly when it matters - a line marking often scores higher than the ball.

Confidence alone cannot separate them. Motion can. A line marking is in a
different place every time it is picked up; the ball moves along a smooth path
at a bounded speed. So the right question is not "which box looks most like a
ball in this frame" but "which sequence of boxes looks most like a ball
flying", and that question can only be answered once the later frames exist.

That is the same argument the project already makes for rendering the annotated
video in a second pass: a bounce is a local maximum in projected court y and is
identifiable only from the frames after it. Ball identity has the same shape.
Here it is resolved with a Viterbi pass over the per-frame candidates.

Why Viterbi rather than a tracker
---------------------------------
A constant-velocity filter mispredicts precisely at hits and bounces, which are
the events the whole system is built to find, and once it locks onto a line
marking it drags its own gate along with it. A global shortest-path over all
candidates has no such state to corrupt: a large displacement at a hit is
charged once and then forgiven, whereas a false positive must pay a large
displacement to get in *and* another to get out, which is what makes it lose.

The lattice also carries an explicit "no ball this frame" state, so lowering
the detector's threshold to admit faint true positives does not force a
false one to be accepted whenever the ball is genuinely absent or occluded.
Recall comes from the low threshold; precision comes from the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pixels per frame beyond which a step is not a tennis ball, on a 1080p
# broadcast wide shot. A 200 km/h serve covers about 1.8 m per frame at 30 fps,
# and the court spans roughly 90 px per metre across the frame at that framing.
# Expressed per frame of separation, so a step across a one-frame dropout is
# judged on the same scale.
MAX_STEP_PX = 200.0

# Motion cost is quadratic in the step, not linear, and that shape is the whole
# point. A linear cost charges an ordinary rally ball moving 50 px per frame a
# quarter of what it charges an impossible 200 px jump - enough to make the
# path prefer declaring the ball missing over admitting it moved, which is
# exactly backwards. Squaring makes ordinary motion nearly free (a 50 px step
# costs 1/16 of a 200 px one) while leaving absurd jumps ruinous.
MOTION_WEIGHT = 2.0
MOTION_CAP = 3.0

# What one frame of "the ball was not visible" costs. It has to sit above the
# total cost of a plausible detection - even a faint one moving quickly - and
# below that of a physically absurd jump. That ordering is what makes the path
# accept low-confidence true positives and still refuse false ones.
MISSING_COST = 0.8

# Weight on the detector's own score. Deliberately small: appearance is the
# weaker signal for a ten-pixel motion-blurred object, so it breaks ties rather
# than driving the decision.
APPEARANCE_WEIGHT = 0.4


@dataclass
class Candidate:
    xy: np.ndarray
    confidence: float


@dataclass
class _State:
    """One node in the lattice: a candidate, or the missing-ball option."""

    xy: np.ndarray | None       # None for the missing state
    confidence: float
    cost: float = 0.0
    back: int = -1              # index of the best predecessor state
    anchor: np.ndarray | None = None   # last real position on this path


@dataclass
class Costs:
    """The four numbers that decide what counts as a ball."""

    missing: float = MISSING_COST
    appearance: float = APPEARANCE_WEIGHT
    motion: float = MOTION_WEIGHT
    max_step_px: float = MAX_STEP_PX
    cap: float = MOTION_CAP


def _emission(state_conf: float | None, costs: Costs) -> float:
    """Cost of asserting this state in this frame, from appearance alone."""
    if state_conf is None:
        return costs.missing
    # A confident box is cheap to accept, a faint one is not free - but the
    # weight is small, so a faint box on a clean path still beats a gap.
    return costs.appearance * (1.0 - state_conf)


def _motion(previous: np.ndarray | None, current: np.ndarray | None,
            span: int, costs: Costs) -> float:
    """Cost of moving between two consecutive states.

    ``span`` is the frame separation, so a step that bridges a dropout is not
    penalised for the frames it crossed. A transition into or out of the
    missing state is free of motion cost - the ball's absence says nothing
    about where it went - but the anchor is carried across so that reappearing
    somewhere impossible is still charged.
    """
    if previous is None or current is None:
        return 0.0
    step = float(np.linalg.norm(current - previous)) / max(span, 1)
    return costs.motion * min((step / costs.max_step_px) ** 2, costs.cap)


def resolve(
    per_frame: list[dict],
    conf_floor: float = 0.0,
    max_bridge: int = 30,
    costs: Costs | None = None,
) -> list[dict]:
    """Pick one ball position per frame, or none, from per-frame candidates.

    ``per_frame`` is ``[{"frame": int, "boxes": [{"conf": float,
    "xy": [x, y]}, ...]}, ...]``, ordered by frame, holding every box the
    detector emitted above its threshold.

    ``max_bridge`` bounds how far apart two observed frames may be and still be
    linked by motion. Beyond it the path restarts rather than charging a
    meaningless step across seconds of missing video - which is what would
    otherwise happen across a camera cut.

    Returns the accepted detections in the same shape the rest of the pipeline
    already consumes: ``{"frame", "image", "confidence"}``.
    """
    costs = costs or Costs()
    rows = [
        (
            int(row["frame"]),
            [
                Candidate(np.asarray(b["xy"], dtype=float), float(b["conf"]))
                for b in row["boxes"]
                if float(b["conf"]) >= conf_floor
            ],
        )
        for row in per_frame
    ]
    rows = [(frame, cands) for frame, cands in rows if cands]
    if not rows:
        return []

    # Forward pass. Each column holds the candidates for that frame plus the
    # missing state, and every state records the cheapest way to reach it.
    columns: list[list[_State]] = []
    previous_frame: int | None = None

    for frame, candidates in rows:
        column = [
            _State(xy=c.xy, confidence=c.confidence) for c in candidates
        ]
        column.append(_State(xy=None, confidence=0.0))

        if not columns:
            for state in column:
                state.cost = _emission(
                    state.confidence if state.xy is not None else None, costs
                )
                state.anchor = state.xy
            columns.append(column)
            previous_frame = frame
            continue

        span = frame - (previous_frame or frame)
        bridged = span <= max_bridge
        for state in column:
            emission = _emission(
                state.confidence if state.xy is not None else None, costs
            )
            best_cost, best_index, best_anchor = float("inf"), -1, None
            for index, prior in enumerate(columns[-1]):
                # Motion is measured against the last real position on that
                # path, so a gap of missing frames does not erase continuity.
                reference = prior.anchor if bridged else None
                cost = prior.cost + _motion(reference, state.xy, span, costs)
                if cost < best_cost:
                    best_cost = cost
                    best_index = index
                    best_anchor = (
                        state.xy if state.xy is not None else prior.anchor
                    )
            state.cost = best_cost + emission
            state.back = best_index
            state.anchor = best_anchor
        columns.append(column)
        previous_frame = frame

    # Backtrace from the cheapest final state.
    path: list[int] = []
    index = min(
        range(len(columns[-1])), key=lambda i: columns[-1][i].cost
    )
    for column in reversed(columns):
        path.append(index)
        index = column[index].back
    path.reverse()

    out: list[dict] = []
    for (frame, _), column, chosen in zip(rows, columns, path):
        state = column[chosen]
        if state.xy is None:
            continue
        out.append(
            {
                "frame": frame,
                "image": [float(state.xy[0]), float(state.xy[1])],
                "confidence": round(float(state.confidence), 3),
            }
        )
    return out
