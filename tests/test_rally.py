"""Bounce detection, rally segmentation and point attribution.

Trajectories here are synthesised from physics rather than pulled from a video:
a ball launched with a known velocity, bouncing at known times, projected into
a known camera. That means the ground truth is exact, so a failure is
unambiguously a detector bug rather than a labelling argument.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis.bounce import BallEvent, EventType, detect_events
from tennis.court import COURT_LENGTH, DOUBLES_WIDTH
from tennis.rally import attribute, score_match, segment, summarise
from tennis.trajectory import BallSample, Trajectory, fill_gaps, smooth

NEAR_Y = COURT_LENGTH * 0.75
FAR_Y = COURT_LENGTH * 0.25
MID_X = DOUBLES_WIDTH / 2


def make_event(
    kind: EventType,
    frame: int,
    x: float = MID_X,
    y: float = FAR_Y,
    confidence: float = 0.9,
    in_bounds: bool = True,
    by_player: int | None = None,
) -> BallEvent:
    return BallEvent(
        type=kind,
        frame=frame,
        court=np.array([x, y]),
        confidence=confidence,
        side="far" if y < COURT_LENGTH / 2 else "near",
        in_bounds=in_bounds,
        by_player=by_player,
    )


def arc(start_frame: int, count: int, peak_height: float = 90.0) -> list[BallSample]:
    """One parabolic flight: image y dips (ball rises) then returns (falls)."""
    samples = []
    for i in range(count):
        t = i / max(count - 1, 1)
        image_y = 600.0 - peak_height * np.sin(np.pi * t)
        samples.append(
            BallSample(
                frame=start_frame + i,
                image=np.array([500.0 + 6 * i, image_y]),
                court=np.array([MID_X, FAR_Y + 0.4 * i]),
                confidence=0.9,
            )
        )
    return samples


class TestTrajectory:
    def test_short_gaps_are_filled(self):
        samples = [
            BallSample(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.9),
            BallSample(4, np.array([40.0, 8.0]), np.array([4.0, 2.0]), 0.9),
        ]
        traj = fill_gaps(samples, max_gap=8)
        assert len(traj) == 5
        middle = traj.get(2)
        assert middle is not None and middle.interpolated
        assert middle.image == pytest.approx([20.0, 4.0])

    def test_long_gaps_are_left_alone(self):
        samples = [
            BallSample(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.9),
            BallSample(50, np.array([10.0, 10.0]), np.array([1.0, 1.0]), 0.9),
        ]
        traj = fill_gaps(samples, max_gap=8)
        assert len(traj) == 2   # nothing invented across a 50-frame hole

    def test_interpolated_samples_are_discounted(self):
        samples = [
            BallSample(0, np.zeros(2), np.zeros(2), 1.0),
            BallSample(3, np.ones(2), np.ones(2), 1.0),
        ]
        traj = fill_gaps(samples)
        assert all(s.confidence < 1.0 for s in traj if s.interpolated)

    def test_smoothing_does_not_bridge_a_gap(self):
        """Averaging across a gap would blend two unrelated flights."""
        left = [BallSample(i, np.array([float(i), 0.0]), np.zeros(2), 0.9)
                for i in range(5)]
        right = [BallSample(i, np.array([1000.0, 0.0]), np.zeros(2), 0.9)
                 for i in range(60, 65)]
        traj = smooth(Trajectory(left + right), window=5)
        assert traj.get(4).image[0] < 10      # untouched by the far-away run
        assert traj.get(60).image[0] == pytest.approx(1000.0)

    def test_smoothing_is_centred_not_trailing(self):
        """A trailing window shifts every event later by half its width."""
        samples = [
            BallSample(i, np.array([0.0, float(i)]), np.zeros(2), 0.9)
            for i in range(11)
        ]
        traj = smooth(Trajectory(samples), window=5)
        # On a straight ramp a centred mean returns the original value.
        assert traj.get(5).image[1] == pytest.approx(5.0)

    def test_rejects_even_window(self):
        with pytest.raises(ValueError, match="odd"):
            smooth(Trajectory([]), window=4)


class TestEventDetection:
    def test_finds_a_bounce_at_the_bottom_of_a_flight(self):
        # Two arcs back to back: the join is the ball hitting the ground.
        samples = arc(0, 15) + arc(15, 15)
        events = detect_events(Trajectory(samples), fps=30.0)
        assert any(e.type is EventType.BOUNCE for e in events)
        bounce_frames = [e.frame for e in events if e.type is EventType.BOUNCE]
        assert any(12 <= f <= 18 for f in bounce_frames)

    def test_ignores_detector_jitter(self):
        """A pixel of noise per frame must not produce dozens of events."""
        rng = np.random.default_rng(0)
        samples = [
            BallSample(
                i,
                np.array([500.0 + i, 600.0 + rng.normal(0, 0.8)]),
                np.array([MID_X, FAR_Y + 0.01 * i]),
                0.9,
            )
            for i in range(120)
        ]
        events = detect_events(Trajectory(samples), fps=30.0)
        assert len(events) <= 2

    def test_turning_point_near_a_player_is_a_hit(self):
        samples = arc(0, 15) + arc(15, 15)
        # A player box straddling the point where the trajectory turns.
        turn = samples[14].image
        box = (turn[0] - 40, turn[1] - 90, turn[0] + 40, turn[1] + 90)
        players = {1: {s.frame: box for s in samples}}
        events = detect_events(Trajectory(samples), player_boxes=players)
        assert any(e.type is EventType.HIT for e in events)
        hit = next(e for e in events if e.type is EventType.HIT)
        assert hit.by_player == 1

    def test_same_turn_far_from_players_is_a_bounce(self):
        samples = arc(0, 15) + arc(15, 15)
        away = (20.0, 20.0, 100.0, 200.0)   # far corner of the frame
        players = {1: {s.frame: away for s in samples}}
        events = detect_events(Trajectory(samples), player_boxes=players)
        assert any(e.type is EventType.BOUNCE for e in events)

    def test_proximity_is_scale_invariant(self):
        """One threshold must serve a near player and a far player.

        The same ball-to-player offset, measured in box heights, has to
        classify the same way whether the player is 360 px tall in the
        foreground or 90 px tall at the far baseline.
        """
        samples = arc(0, 15) + arc(15, 15)
        turn = samples[14].image

        def box(height):
            width = height / 3
            return (turn[0] - width / 2, turn[1] - height / 2,
                    turn[0] + width / 2, turn[1] + height / 2)

        near = detect_events(Trajectory(samples),
                             player_boxes={1: {s.frame: box(360) for s in samples}})
        far = detect_events(Trajectory(samples),
                            player_boxes={1: {s.frame: box(90) for s in samples}})
        near_kinds = [e.type for e in near]
        far_kinds = [e.type for e in far]
        assert EventType.HIT in near_kinds
        assert near_kinds == far_kinds

    def test_airborne_projection_lowers_bounce_confidence(self):
        """A 'bounce' projected off the map was really a ball in the air.

        The homography maps the court plane, so an airborne ball's image ray
        lands far beyond it - real footage produced court y of -7.1 m. Such a
        reading cannot support an in/out call and must be marked down.
        """
        samples = arc(0, 15) + arc(15, 15)
        for s in samples:
            s.court = np.array([MID_X, -7.1])   # impossible landing spot
        events = detect_events(Trajectory(samples))
        bounces = [e for e in events if e.type is EventType.BOUNCE]
        assert bounces and all(e.confidence < 0.5 for e in bounces)

    def test_out_of_bounds_bounce_is_marked(self):
        samples = []
        for i in range(30):
            t = (i % 15) / 14
            samples.append(
                BallSample(
                    i,
                    np.array([500.0 + 6 * i, 600.0 - 90 * np.sin(np.pi * t)]),
                    np.array([DOUBLES_WIDTH + 2.0, FAR_Y]),  # well wide
                    0.9,
                )
            )
        events = detect_events(Trajectory(samples))
        assert events and all(not e.in_bounds for e in events)

    def test_interpolated_events_lose_confidence(self):
        samples = arc(0, 15) + arc(15, 15)
        for s in samples[12:18]:
            s.interpolated = True
        events = detect_events(Trajectory(samples))
        near_join = [e for e in events if 12 <= e.frame <= 18]
        assert near_join and all(e.confidence < 0.9 for e in near_join)


class TestSegmentation:
    def test_splits_on_a_long_silence(self):
        events = (
            [make_event(EventType.HIT, f) for f in (0, 10, 20)]
            + [make_event(EventType.HIT, f) for f in (200, 210, 220)]
        )
        assert len(segment(events, fps=30.0)) == 2

    def test_keeps_a_rally_together_across_small_gaps(self):
        events = [make_event(EventType.HIT, f) for f in (0, 25, 50, 75)]
        assert len(segment(events, fps=30.0)) == 1

    def test_drops_fragments_too_short_to_be_a_point(self):
        assert segment([make_event(EventType.HIT, 0)], fps=30.0) == []


class TestAttribution:
    def test_double_bounce_gives_the_point_to_the_other_player(self):
        rally = segment(
            [
                make_event(EventType.HIT, 0, y=NEAR_Y, by_player=1),
                make_event(EventType.BOUNCE, 20, y=FAR_Y),
                make_event(EventType.BOUNCE, 45, y=FAR_Y),
            ],
            fps=30.0,
        )[0]
        attribute(rally)
        # Player 2 defends the far side and let it bounce twice.
        assert rally.winner == 1
        assert "double bounce" in rally.reason

    def test_a_return_between_bounces_is_not_a_double_bounce(self):
        rally = segment(
            [
                make_event(EventType.BOUNCE, 0, y=FAR_Y),
                make_event(EventType.HIT, 10, y=FAR_Y, by_player=2),
                make_event(EventType.BOUNCE, 40, y=FAR_Y),
            ],
            fps=30.0,
        )[0]
        attribute(rally)
        assert "double bounce" not in rally.reason

    def test_ball_landing_out_loses_the_point_for_the_hitter(self):
        rally = segment(
            [
                make_event(EventType.HIT, 0, y=NEAR_Y, by_player=1),
                make_event(EventType.HIT, 20, y=FAR_Y, by_player=2),
                make_event(EventType.BOUNCE, 40, y=NEAR_Y, in_bounds=False),
            ],
            fps=30.0,
        )[0]
        attribute(rally)
        assert rally.winner == 1        # player 2 hit it out
        assert "outside the court" in rally.reason

    def test_ball_that_never_crosses_is_into_the_net(self):
        rally = segment(
            [
                make_event(EventType.BOUNCE, 0, y=NEAR_Y),
                make_event(EventType.HIT, 15, y=NEAR_Y, by_player=1),
                make_event(EventType.BOUNCE, 30, y=NEAR_Y),
            ],
            fps=30.0,
        )[0]
        attribute(rally)
        assert rally.winner == 2
        assert "cross the net" in rally.reason

    def test_confidence_follows_the_weakest_event(self):
        rally = segment(
            [
                make_event(EventType.HIT, 0, y=NEAR_Y, by_player=1),
                make_event(EventType.BOUNCE, 20, y=FAR_Y, confidence=0.9),
                make_event(EventType.BOUNCE, 45, y=FAR_Y, confidence=0.35),
            ],
            fps=30.0,
        )[0]
        attribute(rally)
        assert rally.confidence == pytest.approx(0.35)

    def test_unreadable_rally_is_left_undecided(self):
        """The system must decline rather than guess."""
        rally = segment(
            [make_event(EventType.HIT, f, y=NEAR_Y, by_player=1)
             for f in (0, 20, 40)],
            fps=30.0,
        )[0]
        attribute(rally)
        assert rally.winner is None
        assert rally.confidence == 0.0


class TestScoringIntegration:
    def _point_to(self, player, base_frame):
        """Events that read as a clean double-bounce win for ``player``."""
        loser_side_y = FAR_Y if player == 1 else NEAR_Y
        return [
            make_event(EventType.HIT, base_frame, y=NEAR_Y, by_player=1),
            make_event(EventType.BOUNCE, base_frame + 20, y=loser_side_y),
            make_event(EventType.BOUNCE, base_frame + 45, y=loser_side_y),
        ]

    def test_four_points_win_a_game(self):
        events = []
        for i in range(4):
            events += self._point_to(1, i * 300)
        match, rallies = score_match(events, fps=30.0)
        assert len(rallies) == 4
        assert match.scoreline() == "1-0 | 0-0"

    def test_undecided_rallies_do_not_award_points(self):
        events = self._point_to(1, 0) + [
            make_event(EventType.HIT, 300 + f, y=NEAR_Y, by_player=1)
            for f in (0, 20, 40)
        ]
        match, rallies = score_match(events, fps=30.0)
        assert len(rallies) == 2
        assert len(match.history) == 1      # only the decided one scored

    def test_summary_reports_what_could_not_be_called(self):
        events = self._point_to(1, 0) + [
            make_event(EventType.HIT, 300 + f, y=NEAR_Y, by_player=1)
            for f in (0, 20, 40)
        ]
        _, rallies = score_match(events, fps=30.0)
        report = summarise(rallies, fps=30.0)
        assert report["rallies_found"] == 2
        assert report["points_decided"] == 1
        assert report["points_undecided"] == 1
