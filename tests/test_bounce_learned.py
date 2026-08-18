"""The proposal rule, which decides what the bounce classifier ever sees.

Recall here is a hard ceiling on the whole scoring pipeline: a bounce that is
never proposed cannot be classified, cannot end a rally, and cannot score a
point. The rule this replaced reached only 35% of real bounces on the TrackNet
dataset, so these tests pin the behaviour that fixed it.
"""

import numpy as np
import pytest

from tennis import bounce_learned
from tennis.bounce_learned import FEATURE_NAMES, _upward_kink, propose_run
from tennis.trajectory import BallSample


def _run(image_pts, court_pts, start=0):
    return [
        BallSample(
            frame=start + i,
            image=np.array(p, dtype=float),
            court=np.array(c, dtype=float),
            confidence=1.0,
        )
        for i, (p, c) in enumerate(zip(image_pts, court_pts))
    ]


def _bouncing_run(n=17, bounce_at=8):
    """A ball descending in the frame, then rebounding upward at ``bounce_at``.

    Court y climbs monotonically throughout - the ball is travelling away from
    the camera the whole time. That is the case the old local-maximum rule was
    structurally blind to, and it is the common case in real footage.
    """
    image, court = [], []
    for i in range(n):
        drop = i if i <= bounce_at else bounce_at - (i - bounce_at)
        image.append((100.0 + 4.0 * i, 200.0 + 6.0 * drop))
        court.append((5.0, 2.0 + 0.9 * i))
    return _run(image, court)


def test_kink_is_scale_free():
    """The same flight filmed at 720p and 1080p must propose identically."""
    image_y = np.array([200.0, 206.0, 212.0, 218.0, 212.0, 200.0, 188.0])
    assert _upward_kink(image_y * 1.5) == pytest.approx(_upward_kink(image_y))


def test_kink_ignores_a_steady_climb():
    """Constant velocity is not a corner, however fast the ball is moving."""
    assert _upward_kink(np.arange(10, dtype=float) * 37.0) == pytest.approx(
        np.zeros(10), abs=1e-9
    )


def test_it_proposes_a_bounce_that_is_not_a_local_maximum():
    """The regression that motivated the whole rewrite.

    Projected court y rises monotonically across this bounce, so the previous
    rule - local maximum in that signal - proposed nothing at all here.
    """
    run = _bouncing_run(bounce_at=8)
    court_y = np.array([s.court[1] for s in run])
    assert np.all(np.diff(court_y) > 0), "fixture must not contain a local max"

    frames = [c.frame for c in propose_run(run)]
    assert frames, "the bounce was not proposed"
    assert min(abs(f - 8) for f in frames) <= bounce_learned.MATCH_TOLERANCE


def test_a_smooth_flight_proposes_nothing():
    run = _run(
        [(100.0 + 4 * i, 200.0 - 5 * i) for i in range(15)],
        [(5.0, 2.0 + 0.7 * i) for i in range(15)],
    )
    assert propose_run(run) == []


def test_a_run_too_short_to_judge_is_declined():
    assert propose_run(_bouncing_run(n=4, bounce_at=2)) == []


def test_every_feature_is_finite_and_named():
    """A NaN here trains silently and predicts garbage at inference."""
    candidates = propose_run(_bouncing_run())
    assert candidates
    for candidate in candidates:
        assert len(candidate.features) == len(FEATURE_NAMES)
        assert np.all(np.isfinite(candidate.features))
