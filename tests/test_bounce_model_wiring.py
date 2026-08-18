"""``detect_events`` with a trained bounce model in place of the threshold."""

import numpy as np

from tennis.bounce import EventType, detect_events
from tennis.trajectory import BallSample, Trajectory


def _trajectory(n=17, bounce_at=8):
    samples = []
    for i in range(n):
        drop = i if i <= bounce_at else bounce_at - (i - bounce_at)
        samples.append(
            BallSample(
                frame=i,
                image=np.array([100.0 + 4.0 * i, 200.0 + 6.0 * drop]),
                court=np.array([5.0, 2.0 + 0.9 * i]),
                confidence=1.0,
            )
        )
    return Trajectory(samples)


class _AcceptAll:
    def accept(self, candidates):
        return [(c, 1.0) for c in candidates]


class _AcceptNone:
    def accept(self, candidates):
        return []


class _Broken:
    def accept(self, candidates):
        raise RuntimeError("corrupt model file")


def test_the_model_decides_which_candidates_become_events():
    events = detect_events(_trajectory(), bounce_model=_AcceptAll())
    assert events
    assert all(e.type is EventType.BOUNCE for e in events)
    assert min(abs(e.frame - 8) for e in events) <= 2


def test_a_model_that_rejects_everything_yields_no_events():
    assert detect_events(_trajectory(), bounce_model=_AcceptNone()) == []


def test_a_broken_model_does_not_take_the_run_down():
    """A corrupt model must cost one segment, not the whole analysis."""
    assert detect_events(_trajectory(), bounce_model=_Broken()) == []


def test_without_a_model_the_geometric_rule_still_runs():
    """The fallback path must stay reachable - it is the documented default."""
    events = detect_events(_trajectory(), bounce_model=None)
    assert isinstance(events, list)


def test_a_rejected_candidate_near_a_player_is_still_a_hit():
    """The regression that produced 57 bounces and 8 hits on a real clip.

    The model is trained to reject racket strikes, so filtering candidates by
    it deletes every hit. A rally stripped of its hits reads as consecutive
    same-side bounces - which the double-bounce rule scores as a point.
    """
    trajectory = _trajectory()
    # A player standing right where the candidate is, in image pixels.
    boxes = {1: {f: (80.0, 180.0, 180.0, 380.0) for f in range(17)}}

    events = detect_events(
        trajectory, player_boxes=boxes, bounce_model=_AcceptNone()
    )
    assert events, "a hit was deleted by the bounce model"
    assert all(e.type is EventType.HIT for e in events)
