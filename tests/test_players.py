"""Tests for separating players from everyone else on camera.

These are written against the failure that prompted the module: a Vienna ATP
clip produced six tracked "players" - two of them actually playing, the rest
ball kids, line judges and the chair umpire - whose movement was being reported
as player movement.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis import players
from tennis.court import DOUBLES_WIDTH, NET_Y, calibrate
from tennis.detect import Detection


def a_calibration():
    image_points = np.array(
        [
            [420, 300], [860, 300],
            [180, 660], [1100, 660],
            [475, 300], [265, 660],
            [805, 300], [1015, 660],
            [475, 390], [805, 390],
            [360, 520], [920, 520],
            [640, 390], [640, 520],
        ],
        dtype=np.float64,
    )
    return calibrate(image_points)


def at_court(calibration, x: float, y: float, height: float = 120.0,
             track_id: int = 1) -> Detection:
    """A detection whose feet land on a given court coordinate."""
    feet = calibration.to_image(np.array([[x, y]]))[0]
    return Detection(
        bbox=(float(feet[0]) - 20, float(feet[1]) - height,
              float(feet[0]) + 20, float(feet[1])),
        confidence=0.9,
        track_id=track_id,
    )


# --- geometry -------------------------------------------------------------


def test_a_point_inside_the_court_is_zero_distance():
    assert players.court_distance(np.array([5.0, 12.0])) == 0.0


def test_distance_grows_outside_the_court():
    assert players.court_distance(np.array([-2.0, 12.0])) == pytest.approx(2.0)
    assert players.court_distance(np.array([5.0, -3.0])) == pytest.approx(3.0)


def test_a_player_behind_the_baseline_is_still_plausible():
    # Players routinely retrieve deep balls several metres back.
    assert players.is_plausible(np.array([5.0, -3.0]))
    assert players.is_plausible(np.array([DOUBLES_WIDTH + 2.0, 12.0]))


def test_the_umpire_and_the_crowd_are_not_plausible():
    assert not players.is_plausible(np.array([14.6, 5.4]))    # chair umpire
    assert not players.is_plausible(np.array([-3.5, 11.3]))   # by the net post
    assert not players.is_plausible(np.array([14.7, -8.5]))   # behind the court


# --- selection ------------------------------------------------------------


def test_the_six_track_case_reduces_to_two():
    # The exact positions measured on the clip that prompted this module.
    calibration = a_calibration()
    detections = [
        at_court(calibration, 5.9, 25.0, track_id=1),    # near player
        at_court(calibration, 0.9, -2.7, track_id=2),    # far player
        at_court(calibration, 14.7, -8.5, track_id=3),   # off court
        at_court(calibration, 14.6, 5.4, track_id=4),    # chair umpire
        at_court(calibration, -3.4, -8.3, track_id=5),   # off court
        at_court(calibration, -3.5, 11.3, track_id=6),   # by the net post
    ]
    chosen = players.select(detections, calibration)
    assert {d.track_id for d in chosen} == {1, 2}


def test_one_player_per_side_of_the_net():
    calibration = a_calibration()
    chosen = players.select(
        [
            at_court(calibration, 5.0, 20.0, track_id=1),
            at_court(calibration, 5.0, 4.0, track_id=2),
        ],
        calibration,
    )
    sides = [
        float(calibration.to_court(d.feet)[1]) >= NET_Y for d in chosen
    ]
    assert sorted(sides) == [False, True]


def test_a_ball_kid_behind_a_player_loses_to_that_player():
    # The hard case: same side of the net, both plausible. The one on the court
    # wins on distance.
    calibration = a_calibration()
    player = at_court(calibration, 5.0, 22.0, track_id=1)
    ball_kid = at_court(calibration, 1.0, 27.5, height=70.0, track_id=9)
    chosen = players.select([ball_kid, player], calibration)
    assert [d.track_id for d in chosen] == [1]


def test_never_returns_more_than_two():
    calibration = a_calibration()
    detections = [
        at_court(calibration, 2.0 + i * 0.5, 20.0, track_id=i)
        for i in range(6)
    ]
    assert len(players.select(detections, calibration)) <= 2


def test_an_empty_frame_yields_nothing():
    assert players.select([], a_calibration()) == []


def test_only_one_side_occupied_returns_one():
    calibration = a_calibration()
    chosen = players.select(
        [at_court(calibration, 5.0, 20.0, track_id=1)], calibration
    )
    assert len(chosen) == 1


def test_without_a_calibration_it_falls_back_to_the_two_largest():
    # No court means no metres to measure against. Players are nearer the
    # camera than the crowd, so box size is the best guess available.
    small = Detection(bbox=(0, 0, 10, 20), confidence=0.9, track_id=1)
    large = Detection(bbox=(0, 0, 100, 220), confidence=0.9, track_id=2)
    medium = Detection(bbox=(0, 0, 50, 110), confidence=0.9, track_id=3)
    chosen = players.select([small, large, medium], None)
    assert [d.track_id for d in chosen] == [2, 3]
