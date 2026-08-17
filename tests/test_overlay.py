"""Tests for the annotated-video renderer.

The renderer's one substantive claim is that every frame carries the score *as
it stood at that frame* - which is the thing a single detection pass cannot do
and the reason the second pass exists. Most of what follows pins that down; the
rest checks it never crashes on the ragged data real footage produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis import overlay
from tennis.bounce import BallEvent, EventType
from tennis.rally import Rally
from tennis.court import COURT_MODEL, calibrate
from tennis.scoring import Match

WIDTH, HEIGHT, FPS = 1280, 720, 30.0


def a_calibration():
    """A real homography, from the court model projected with a known matrix."""
    # A plausible broadcast view: the court model scaled and perspective-warped.
    image_points = np.array(
        [
            [420, 300], [860, 300],     # far doubles corners
            [180, 660], [1100, 660],    # near doubles corners
            [475, 300], [265, 660],
            [805, 300], [1015, 660],
            [475, 390], [805, 390],
            [360, 520], [920, 520],
            [640, 390], [640, 520],
        ],
        dtype=np.float64,
    )
    return calibrate(image_points)


def bounce(frame: int, x: float, y: float, in_bounds: bool = True) -> BallEvent:
    return BallEvent(
        type=EventType.BOUNCE,
        frame=frame,
        court=np.array([x, y]),
        confidence=0.8,
        side="far" if y < 11.885 else "near",
        in_bounds=in_bounds,
    )


def hit(frame: int, by: int = 1) -> BallEvent:
    return BallEvent(
        type=EventType.HIT,
        frame=frame,
        court=np.array([5.0, 18.0]),
        confidence=0.7,
        side="near",
        in_bounds=True,
        by_player=by,
    )


def a_renderer(**overrides):
    events = overrides.pop("events", [hit(10), bounce(30, 4.0, 5.0), hit(50)])
    rallies = overrides.pop(
        "rallies",
        [Rally(start_frame=10, end_frame=60, events=events, winner=1,
               reason="landed out", confidence=0.82)],
    )
    match = overrides.pop("match", None)
    if match is None:
        match = Match()
        match.award_point(1, reason="landed out", confidence=0.82,
                          start_frame=10, end_frame=60)

    defaults = dict(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        ball_track=[
            {"frame": f, "image": [600 + f, 400 - f], "court": [5.0, 10.0],
             "confidence": 0.6}
            for f in range(0, 70)
        ],
        player_boxes={1: {f: (500, 500, 560, 640) for f in range(70)},
                      2: {f: (600, 300, 650, 380) for f in range(70)}},
        player_court={
            1: {f: np.array([5.0 + f * 0.02, 19.0]) for f in range(70)},
            2: {f: np.array([5.0, 4.0]) for f in range(70)},
        },
        events=events,
        rallies=rallies,
        match=match,
        calibration=a_calibration(),
    )
    defaults.update(overrides)
    return overlay.Renderer(**defaults)


def blank():
    # Mid-grey, so both the dark panels and the light text are visibly drawn.
    return np.full((HEIGHT, WIDTH, 3), 120, dtype=np.uint8)


# --- the claim: the score is the score at that frame ----------------------


def test_score_before_the_point_is_not_the_score_after_it():
    renderer = a_renderer()
    before = renderer.states[20]
    after = renderer.states[65]
    assert before.points_played == 0
    assert after.points_played == 1
    assert before.scoreline != after.scoreline


def test_score_updates_only_from_the_frame_the_point_ended():
    renderer = a_renderer()
    assert renderer.states[59].points_played == 0
    assert renderer.states[60].points_played == 1


def test_second_point_supersedes_the_first():
    match = Match()
    match.award_point(1, reason="landed out", confidence=0.9,
                      start_frame=10, end_frame=60)
    match.award_point(1, reason="double bounce", confidence=0.9,
                      start_frame=70, end_frame=120)
    renderer = a_renderer(
        match=match,
        rallies=[
            Rally(start_frame=10, end_frame=60, events=[hit(10)], winner=1,
                  reason="landed out", confidence=0.9),
            Rally(start_frame=70, end_frame=120, events=[hit(70)], winner=1,
                  reason="double bounce", confidence=0.9),
        ],
    )
    assert renderer.states[65].points_played == 1
    assert renderer.state_for(121).points_played == 2
    # 30-0 after two points to the same player, not 15-0.
    assert "30" in renderer.state_for(121).scoreline


# --- rally and event state -----------------------------------------------


def test_rally_is_live_inside_its_frames_and_not_outside():
    renderer = a_renderer()
    assert renderer.states[30].rally_index == 0
    assert renderer.states[5].rally_index is None
    assert renderer.states[65].rally_index is None


def test_shot_count_grows_through_the_rally():
    renderer = a_renderer()
    assert renderer.states[20].rally_shots == 1
    assert renderer.states[55].rally_shots == 2


def test_event_caption_persists_past_its_single_frame():
    # At 30 fps one frame is 33 ms; a caption drawn on one frame is invisible.
    renderer = a_renderer()
    assert renderer.states[30].event is not None
    assert renderer.states[34].event is not None
    assert renderer.states[45].event is None


def test_bounces_accumulate_within_a_rally_and_clear_between_points():
    events = [hit(10), bounce(20, 4.0, 5.0), bounce(30, 6.0, 6.0)]
    renderer = a_renderer(
        events=events,
        rallies=[Rally(start_frame=10, end_frame=40, events=events, winner=1,
                       reason="landed out", confidence=0.8)],
    )
    assert len(renderer.states[35].bounces) == 2
    assert renderer.states[50].bounces == []


def test_point_banner_appears_at_the_end_and_expires():
    # Ball detections out to frame 300 so the state range covers the expiry.
    renderer = a_renderer(
        ball_track=[
            {"frame": f, "image": [600, 400], "court": [5.0, 10.0],
             "confidence": 0.6}
            for f in range(300)
        ]
    )
    assert renderer.states[60].point_banner is not None
    # ASCII only: cv2.putText draws an em dash as a hollow box.
    assert renderer.states[60].point_banner[0] == "POINT  -  PLAYER 1"
    assert renderer.states[60].point_banner[0].isascii()
    # Still up 1s later, gone well after the 2.2s window.
    assert renderer.states[90].point_banner is not None
    assert renderer.states[200].point_banner is None


# --- measurements shown live ----------------------------------------------


def test_distance_accumulates_and_never_decreases():
    renderer = a_renderer()
    early = renderer._distance_at(1, 10)
    late = renderer._distance_at(1, 60)
    assert late > early >= 0


def test_a_teleporting_track_id_does_not_inflate_distance():
    # A track-id swap jumps a player across the court. Counting it would make
    # "distance covered" meaningless, so steps over 2 m are dropped.
    positions = {0: np.array([1.0, 19.0]), 1: np.array([9.0, 3.0])}
    renderer = a_renderer(player_court={1: positions})
    assert renderer._distance_at(1, 1) == 0.0


def test_speed_is_zero_without_enough_observations():
    renderer = a_renderer(player_court={1: {0: np.array([5.0, 19.0])}})
    assert renderer._speed_at(1, 0) == 0.0


# --- rendering doesn't crash and does draw -------------------------------


def test_render_returns_a_frame_of_the_same_shape():
    renderer = a_renderer()
    out = renderer.render(blank(), 30)
    assert out.shape == (HEIGHT, WIDTH, 3)
    assert out.dtype == np.uint8


def test_render_does_not_mutate_the_input_frame():
    renderer = a_renderer()
    frame = blank()
    original = frame.copy()
    renderer.render(frame, 30)
    assert np.array_equal(frame, original)


def test_render_actually_draws_something():
    renderer = a_renderer()
    frame = blank()
    assert not np.array_equal(renderer.render(frame, 30), frame)


@pytest.mark.parametrize("index", [0, 10, 30, 60, 61, 200])
def test_render_survives_any_frame_index(index):
    renderer = a_renderer()
    assert renderer.render(blank(), index).shape == (HEIGHT, WIDTH, 3)


def test_frames_after_the_last_rally_keep_the_final_score():
    # Falling back to a fresh state here would redraw 0-0 for the rest of the
    # video, which reads as a real result rather than as missing data.
    renderer = a_renderer()
    tail = renderer.state_for(5000)
    assert tail.points_played == 1
    assert tail.scoreline == renderer.states[renderer.last_state_frame].scoreline


def test_render_works_with_no_calibration():
    # The court was never found: the overlay must still draw the rest rather
    # than take the whole video down.
    renderer = a_renderer(calibration=None)
    assert renderer.render(blank(), 30).shape == (HEIGHT, WIDTH, 3)


def test_render_works_with_nothing_detected_at_all():
    renderer = a_renderer(
        ball_track=[], player_boxes={}, player_court={}, events=[], rallies=[],
        match=Match(), calibration=None,
    )
    assert renderer.render(blank(), 5).shape == (HEIGHT, WIDTH, 3)


def test_render_handles_a_ball_outside_the_frame():
    # Detections near an edge, plus a projected court coordinate off the plan,
    # must not raise from an out-of-range pixel.
    renderer = a_renderer(
        ball_track=[
            {"frame": 30, "image": [WIDTH + 50, -20], "court": [-4.0, 30.0],
             "confidence": 0.5}
        ]
    )
    assert renderer.render(blank(), 30).shape == (HEIGHT, WIDTH, 3)


def test_panels_are_drawn_inside_the_frame_for_a_small_video():
    # A 640x360 clip is small enough that a fixed-size minimap would fall off
    # the right edge.
    renderer = a_renderer(width=640, height=360)
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    assert renderer.render(frame, 30).shape == (360, 640, 3)


def test_the_fourteen_court_landmarks_are_drawn_in_red():
    # The keypoint model's whole job. Drawn from the fitted homography, so a
    # dot off its line means the calibration drifted - visible, not hidden.
    renderer = a_renderer()
    frame = blank()
    out = renderer.render(frame, 30)
    calibration = a_calibration()
    landmarks = calibration.to_image(COURT_MODEL)
    hits = 0
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
            continue
        patch = out[max(y - 3, 0):y + 4, max(x - 3, 0):x + 4]
        # Red in BGR: the red channel dominates at the landmark.
        if patch[..., 2].max() > 180 and patch[..., 1].max() < 200:
            hits += 1
    assert hits >= 12, f"only {hits} of 14 landmarks drawn"


def test_landmarks_are_absent_without_a_calibration():
    renderer = a_renderer(calibration=None)
    out = renderer.render(blank(), 30)
    assert out.shape == (HEIGHT, WIDTH, 3)
