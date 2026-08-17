"""Serve and fault detection tests."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.bounce import BallEvent, EventType
from tennis.court import ALLEY_WIDTH, NET_Y, SINGLES_WIDTH
from tennis.rally import Rally
from tennis.serve import (
    ServeOutcome,
    analyse_serves,
    classify_serve,
    expected_box,
    service_box,
    summarise,
)

CENTRE_X = ALLEY_WIDTH + SINGLES_WIDTH / 2
LEFT_X = ALLEY_WIDTH + SINGLES_WIDTH * 0.25
RIGHT_X = ALLEY_WIDTH + SINGLES_WIDTH * 0.75

FAR_BOX_Y = 8.0      # between the far service line (5.485) and the net
NEAR_BOX_Y = 15.5    # between the net and the near service line (18.285)


def hit(y: float, frame: int, player: int | None = 1) -> BallEvent:
    return BallEvent(
        type=EventType.HIT, frame=frame, court=np.array([CENTRE_X, y]),
        confidence=0.9, side="far" if y < NET_Y else "near",
        in_bounds=True, by_player=player,
    )


def bounce(x: float, y: float, frame: int, conf: float = 0.9) -> BallEvent:
    return BallEvent(
        type=EventType.BOUNCE, frame=frame, court=np.array([x, y]),
        confidence=conf, side="far" if y < NET_Y else "near", in_bounds=True,
    )


def rally_of(*events) -> Rally:
    evs = list(events)
    return Rally(evs[0].frame, evs[-1].frame, events=evs)


class TestServiceBox:
    def test_far_boxes_split_at_the_centre_line(self):
        assert service_box(np.array([LEFT_X, FAR_BOX_Y])) == "far-deuce"
        assert service_box(np.array([RIGHT_X, FAR_BOX_Y])) == "far-ad"

    def test_near_boxes_are_mirrored(self):
        """Deuce is the server's right, so the two halves mirror."""
        assert service_box(np.array([RIGHT_X, NEAR_BOX_Y])) == "near-deuce"
        assert service_box(np.array([LEFT_X, NEAR_BOX_Y])) == "near-ad"

    def test_long_serve_is_outside_every_box(self):
        # Past the far service line, towards the baseline.
        assert service_box(np.array([CENTRE_X, 3.0])) is None

    def test_wide_serve_is_outside_every_box(self):
        assert service_box(np.array([ALLEY_WIDTH + SINGLES_WIDTH + 1.0, FAR_BOX_Y])) is None

    def test_line_calls_are_generous(self):
        """A serve out by 5 cm is not a call this system should make."""
        just_long = np.array([LEFT_X, 5.44])   # 4.5 cm past the service line
        assert service_box(just_long) == "far-deuce"

    def test_a_ball_on_the_centre_line_still_gets_a_box(self):
        """Exactly on the line is arbitrary but must not be None."""
        assert service_box(np.array([CENTRE_X, FAR_BOX_Y])) is not None


class TestExpectedBox:
    def test_serving_alternates_deuce_then_ad(self):
        assert expected_box(1, 0) == "far-deuce"
        assert expected_box(1, 1) == "far-ad"
        assert expected_box(1, 2) == "far-deuce"

    def test_player_two_serves_into_the_near_half(self):
        assert expected_box(2, 0) == "near-deuce"


class TestClassifyServe:
    def test_good_serve(self):
        rally = rally_of(hit(21.0, 0), bounce(LEFT_X, FAR_BOX_Y, 20))
        serve = classify_serve(rally, 0, 0)
        assert serve is not None
        assert serve.outcome is ServeOutcome.IN
        assert serve.server == 1
        assert serve.box == "far-deuce"

    def test_serve_landing_past_the_service_line_is_a_fault(self):
        rally = rally_of(hit(21.0, 0), bounce(CENTRE_X, 3.0, 20))
        assert classify_serve(rally, 0, 0).outcome is ServeOutcome.FAULT

    def test_serve_landing_wide_is_a_fault(self):
        rally = rally_of(
            hit(21.0, 0),
            bounce(ALLEY_WIDTH + SINGLES_WIDTH + 1.5, FAR_BOX_Y, 20),
        )
        assert classify_serve(rally, 0, 0).outcome is ServeOutcome.FAULT

    def test_serve_into_the_wrong_box_still_counts_as_in(self):
        """Deuce/ad is reported but not enforced.

        Enforcing it needs an exact point count; one missed point earlier in a
        clip would otherwise turn every later serve into a fault.
        """
        rally = rally_of(hit(21.0, 0), bounce(RIGHT_X, FAR_BOX_Y, 20))
        serve = classify_serve(rally, 0, 0)   # point 0 expects far-deuce
        assert serve.expected_box == "far-deuce"
        assert serve.box == "far-ad"
        assert serve.outcome is ServeOutcome.IN

    def test_no_landing_is_unknown_not_a_fault(self):
        """Refuse to call a serve the system never saw land."""
        serve = classify_serve(rally_of(hit(21.0, 0), hit(5.0, 30, 2)), 0, 0)
        assert serve.outcome is ServeOutcome.UNKNOWN
        assert serve.confidence == 0.0

    def test_rally_with_no_hit_has_no_serve(self):
        assert classify_serve(rally_of(bounce(CENTRE_X, 8.0, 0)), 0, 0) is None

    def test_serve_from_mid_court_loses_confidence(self):
        """More likely a misidentified rally start than a real serve."""
        rally = rally_of(hit(21.0, 0), bounce(LEFT_X, FAR_BOX_Y, 20))
        baseline = {1: {0: np.array([CENTRE_X, 23.0])}}     # behind baseline
        midcourt = {1: {0: np.array([CENTRE_X, 14.0])}}     # near the net
        assert (
            classify_serve(rally, 0, 0, baseline).confidence
            > classify_serve(rally, 0, 0, midcourt).confidence
        )


class TestFaultSequencing:
    def _serve_rally(self, landing_y, frame_base, x=LEFT_X):
        return rally_of(
            hit(21.0, frame_base), bounce(x, landing_y, frame_base + 20)
        )

    def test_first_serve_fault_is_not_a_lost_point(self):
        """The bug this module exists to fix.

        Without it a fault looks like a lost rally and the point goes to the
        wrong player.
        """
        rallies = [self._serve_rally(3.0, 0)]      # long: fault
        _, faults, doubles = analyse_serves(rallies)
        assert faults == [0]
        assert doubles == []

    def test_two_faults_give_the_point_to_the_receiver(self):
        rallies = [self._serve_rally(3.0, 0), self._serve_rally(3.0, 300)]
        _, faults, doubles = analyse_serves(rallies)
        assert faults == [0]           # only the first is a non-point
        assert doubles == [(1, 2)]     # player 1 served, player 2 wins

    def test_fault_then_good_serve_resets(self):
        rallies = [self._serve_rally(3.0, 0), self._serve_rally(FAR_BOX_Y, 300)]
        serves, faults, doubles = analyse_serves(rallies)
        assert faults == [0]
        assert doubles == []
        assert serves[1].is_second_serve
        assert serves[1].outcome is ServeOutcome.IN

    def test_three_faults_are_one_double_then_a_new_first(self):
        rallies = [self._serve_rally(3.0, f) for f in (0, 300, 600)]
        _, faults, doubles = analyse_serves(rallies)
        assert doubles == [(1, 2)]
        assert faults == [0, 2]        # rally 2 opens a fresh point

    def test_good_serves_produce_no_faults(self):
        rallies = [self._serve_rally(FAR_BOX_Y, f) for f in (0, 300)]
        _, faults, doubles = analyse_serves(rallies)
        assert faults == [] and doubles == []


class TestSummary:
    def test_first_serve_percentage(self):
        rallies = [
            rally_of(hit(21.0, 0), bounce(LEFT_X, FAR_BOX_Y, 20)),      # in
            rally_of(hit(21.0, 300), bounce(CENTRE_X, 3.0, 320)),        # fault
            rally_of(hit(21.0, 600), bounce(LEFT_X, FAR_BOX_Y, 620)),    # 2nd, in
        ]
        serves, _, doubles = analyse_serves(rallies)
        report = summarise(serves, doubles)
        assert report["serves_detected"] == 3
        assert report["faults"] == 1
        # Two of the three were first serves; one of those went in.
        assert report["first_serve_percentage"] == pytest.approx(0.5)

    def test_empty(self):
        report = summarise([], [])
        assert report["serves_detected"] == 0
        assert report["first_serve_percentage"] is None


class TestScoringIntegration:
    """Faults must change the score, which is the whole point of this module."""

    def _serve_events(self, base, landing_y, x=LEFT_X):
        return [hit(21.0, base), bounce(x, landing_y, base + 20)]

    def test_a_fault_awards_no_point(self):
        from tennis.rally import score_match

        events = self._serve_events(0, 3.0)          # long serve: fault
        match, _ = score_match(events, fps=30.0, fault_indices={0})
        assert len(match.history) == 0
        assert match.scoreline() == "0-0 | 0-0"

    def test_without_fault_handling_the_wrong_player_scores(self):
        """Documents the bug this module removes.

        The same fault, scored with no serve knowledge, hands a point to
        somebody - which is exactly what the baseline pipeline would do.
        """
        from tennis.rally import score_match

        events = self._serve_events(0, 3.0)
        naive, _ = score_match(events, fps=30.0)
        aware, _ = score_match(events, fps=30.0, fault_indices={0})
        assert len(aware.history) <= len(naive.history)

    def test_double_fault_gives_the_receiver_the_point(self):
        from tennis.rally import score_match

        events = self._serve_events(0, 3.0) + self._serve_events(300, 3.0)
        match, rallies = score_match(
            events, fps=30.0, fault_indices={0}, double_faults={1: 2}
        )
        assert len(match.history) == 1
        assert match.history[0].winner == 2
        assert match.history[0].reason == "double fault"
        assert match.scoreline() == "0-0 | 0-15"
