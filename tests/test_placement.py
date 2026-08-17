"""Shot placement classification tests."""

from __future__ import annotations

import numpy as np
import pytest

from tennis.bounce import BallEvent, EventType
from tennis.court import ALLEY_WIDTH, COURT_LENGTH, NET_Y, SINGLES_WIDTH
from tennis.placement import (
    classify_landing,
    landings_from_rallies,
    placement_grid,
    render_grid,
    shot_direction,
    summarise,
)
from tennis.rally import Rally

CENTRE_X = ALLEY_WIDTH + SINGLES_WIDTH / 2
LEFT_X = ALLEY_WIDTH + SINGLES_WIDTH * 0.1
RIGHT_X = ALLEY_WIDTH + SINGLES_WIDTH * 0.9


def bounce(x: float, y: float, frame: int = 0, conf: float = 0.9) -> BallEvent:
    return BallEvent(
        type=EventType.BOUNCE,
        frame=frame,
        court=np.array([x, y]),
        confidence=conf,
        side="far" if y < NET_Y else "near",
        in_bounds=True,
    )


def hit(x: float, y: float, frame: int, player: int) -> BallEvent:
    return BallEvent(
        type=EventType.HIT,
        frame=frame,
        court=np.array([x, y]),
        confidence=0.9,
        side="far" if y < NET_Y else "near",
        in_bounds=True,
        by_player=player,
    )


class TestDepth:
    def test_deep_ball_near_the_baseline(self):
        landing = classify_landing(np.array([CENTRE_X, 1.0]))     # far baseline
        assert landing.side == "far"
        assert landing.depth_band == "deep"
        assert landing.depth_m == pytest.approx(NET_Y - 1.0)

    def test_short_ball_inside_the_service_line(self):
        # 3 m past the net is well inside the service box.
        assert classify_landing(np.array([CENTRE_X, NET_Y - 3.0])).depth_band == "short"

    def test_mid_court_ball(self):
        assert classify_landing(np.array([CENTRE_X, NET_Y - 8.0])).depth_band == "mid"

    def test_depth_is_measured_from_the_net_on_both_sides(self):
        """The two halves must be directly comparable."""
        far = classify_landing(np.array([CENTRE_X, NET_Y - 9.0]))
        near = classify_landing(np.array([CENTRE_X, NET_Y + 9.0]))
        assert far.side == "far" and near.side == "near"
        assert far.depth_m == pytest.approx(near.depth_m)
        assert far.depth_band == near.depth_band


class TestWidth:
    def test_centre_ball(self):
        assert classify_landing(np.array([CENTRE_X, 5.0])).width_band == "centre"

    def test_width_is_mirrored_across_the_net(self):
        """'left' must mean the same physical side to both players.

        Court x is a single global axis, so without the flip the same corner
        would be reported as left for one player and right for the other, and
        every cross-court statistic would be meaningless.
        """
        far = classify_landing(np.array([LEFT_X, 4.0]))       # far half
        near = classify_landing(np.array([LEFT_X, 20.0]))     # near half
        assert far.width_band != near.width_band

    def test_out_wide_is_flagged_out_of_bounds(self):
        wide = classify_landing(np.array([ALLEY_WIDTH + SINGLES_WIDTH + 1.0, 5.0]))
        assert not wide.in_bounds


class TestDirection:
    def test_down_the_line(self):
        assert shot_direction(np.array([LEFT_X, 20.0]),
                              np.array([LEFT_X + 0.3, 3.0])) == "down-the-line"

    def test_cross_court(self):
        assert shot_direction(np.array([LEFT_X, 20.0]),
                              np.array([RIGHT_X, 3.0])) == "cross-court"

    def test_direction_uses_lateral_travel_not_distance(self):
        """A long straight ball is down-the-line, however far it travelled."""
        assert shot_direction(np.array([CENTRE_X, 22.0]),
                              np.array([CENTRE_X, 1.0])) == "down-the-line"


class TestRallyExtraction:
    def test_bounce_is_attributed_to_the_last_hitter(self):
        rally = Rally(
            start_frame=0,
            end_frame=60,
            events=[hit(CENTRE_X, 20.0, 0, 7), bounce(CENTRE_X, 3.0, 30)],
        )
        positions = {7: {0: np.array([CENTRE_X, 20.0])}}
        landings = landings_from_rallies([rally], positions)
        assert len(landings) == 1
        assert landings[0].hit_by == 7
        assert landings[0].direction == "down-the-line"

    def test_direction_needs_the_hitter_position(self):
        """Without a known origin, a landing is recorded but not directed."""
        rally = Rally(0, 60, events=[hit(CENTRE_X, 20.0, 0, 7),
                                     bounce(CENTRE_X, 3.0, 30)])
        landings = landings_from_rallies([rally], player_positions=None)
        assert landings[0].direction is None
        assert landings[0].hit_by == 7      # attribution still works

    def test_position_lookup_tolerates_a_missing_frame(self):
        rally = Rally(0, 60, events=[hit(CENTRE_X, 20.0, 10, 7),
                                     bounce(CENTRE_X, 3.0, 40)])
        positions = {7: {8: np.array([CENTRE_X, 20.0])}}   # frame 10 missing
        assert landings_from_rallies([rally], positions)[0].direction is not None

    def test_bounce_before_any_hit_is_unattributed(self):
        rally = Rally(0, 30, events=[bounce(CENTRE_X, 3.0, 10),
                                     hit(CENTRE_X, 20.0, 20, 7)])
        assert landings_from_rallies([rally], {})[0].hit_by is None


class TestGrids:
    def test_grid_counts_only_the_requested_side(self):
        landings = [classify_landing(np.array([CENTRE_X, 3.0])),
                    classify_landing(np.array([CENTRE_X, 20.0]))]
        assert sum(map(sum, placement_grid(landings, "far"))) == 1
        assert sum(map(sum, placement_grid(landings, "near"))) == 1

    def test_grid_excludes_out_balls(self):
        wide = classify_landing(np.array([ALLEY_WIDTH + SINGLES_WIDTH + 2.0, 3.0]))
        assert sum(map(sum, placement_grid([wide], "far"))) == 0

    def test_render_marks_the_net_end_first(self):
        lines = render_grid([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        assert len(lines) == 3
        assert lines[0].strip("|").strip() == ""      # nothing landed short

    def test_empty_grid_says_so(self):
        assert render_grid([[0] * 3 for _ in range(3)]) == ["(no landings)"]


class TestSummary:
    def test_counts_in_and_out(self):
        rally = Rally(0, 90, events=[
            hit(CENTRE_X, 20.0, 0, 1),
            bounce(CENTRE_X, 3.0, 20),
            hit(CENTRE_X, 3.0, 40, 2),
            BallEvent(EventType.BOUNCE, 60, np.array([ALLEY_WIDTH + SINGLES_WIDTH + 2, 20.0]),
                      0.8, "near", False),
        ])
        report = summarise([rally], {})
        assert report["total_landings"] == 2
        assert report["in_bounds"] == 1
        assert report["out_of_bounds"] == 1

    def test_per_player_breakdown(self):
        rally = Rally(0, 120, events=[
            hit(CENTRE_X, 20.0, 0, 1), bounce(CENTRE_X, 2.0, 20),
            hit(CENTRE_X, 2.0, 40, 2), bounce(CENTRE_X, 21.0, 60),
            hit(CENTRE_X, 21.0, 80, 1), bounce(CENTRE_X, 2.5, 100),
        ])
        report = summarise([rally], {})
        assert report["by_player"]["1"]["landings"] == 2
        assert report["by_player"]["2"]["landings"] == 1

    def test_deep_share_is_a_fraction(self):
        rally = Rally(0, 60, events=[
            hit(CENTRE_X, 20.0, 0, 1),
            bounce(CENTRE_X, 1.0, 20),      # deep
            hit(CENTRE_X, 1.0, 30, 2),
            bounce(CENTRE_X, NET_Y + 2.0, 50),   # short
        ])
        share = summarise([rally], {})["overall"]["deep_share"]
        assert 0.0 <= share <= 1.0
        assert share == pytest.approx(0.5)

    def test_empty_input(self):
        report = summarise([], {})
        assert report["total_landings"] == 0
        assert report["overall"] == {"landings": 0}
