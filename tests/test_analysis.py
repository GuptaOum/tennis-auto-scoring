"""Movement analysis tests."""

from __future__ import annotations

import pytest

from tennis.analysis import (
    analyse_players,
    court_coverage,
    player_stats,
    render_coverage,
)


def walk(track_id: int, start_y: float, step: float, count: int, x: float = 5.0):
    """A player moving in a straight line at a constant rate."""
    return [
        {"frame": i, "track_id": track_id, "x_m": x, "y_m": start_y + step * i}
        for i in range(count)
    ]


class TestPlayerStats:
    def test_distance_and_speed(self):
        # 0.1 m per frame at 30 fps = 3 m/s = 10.8 km/h
        stats = player_stats(walk(1, 20.0, 0.1, 31), fps=30.0, track_id=1)
        assert stats is not None
        assert stats.distance_m == pytest.approx(3.0, abs=1e-6)
        assert stats.avg_speed_kmh == pytest.approx(10.8, abs=1e-6)

    def test_id_switch_teleport_is_discarded(self):
        """One tracker glitch must not inflate the distance total."""
        track = walk(1, 20.0, 0.1, 10)
        track.append({"frame": 10, "track_id": 1, "x_m": 5.0, "y_m": 0.5})  # jump
        track += [
            {"frame": 11 + i, "track_id": 1, "x_m": 5.0, "y_m": 0.5 + 0.1 * i}
            for i in range(10)
        ]
        stats = player_stats(track, fps=30.0, track_id=1)
        assert stats is not None
        # ~0.9 m before the jump and ~0.9 m after; the 18 m teleport is dropped.
        assert stats.distance_m < 3.0
        assert stats.top_speed_kmh < 45.0

    def test_side_is_taken_from_where_they_played(self):
        near = player_stats(walk(1, 20.0, 0.01, 20), fps=30.0, track_id=1)
        far = player_stats(walk(2, 3.0, 0.01, 20), fps=30.0, track_id=2)
        assert near is not None and near.side == "near"
        assert far is not None and far.side == "far"

    def test_net_approach_counts_transitions_not_frames(self):
        """Standing at the net for 100 frames is one approach, not 100."""
        track = walk(1, 20.0, -0.5, 30)     # baseline in towards the net
        stats = player_stats(track, fps=30.0, track_id=1)
        assert stats is not None
        assert stats.net_approaches == 1
        assert stats.time_at_net_s > 0

    def test_too_short_a_track_yields_nothing(self):
        assert player_stats([{"frame": 0, "x_m": 1, "y_m": 1}], 30.0, 1) is None


class TestPlayerSelection:
    def test_picks_the_two_most_tracked_identities(self):
        """Ball kids and crowd appear briefly; players persist.

        The baseline chose players from frame 0 alone and could never revise
        that choice. Ranking by tracked duration is decided by the whole video.
        """
        track = (
            walk(1, 20.0, 0.05, 200)      # player
            + walk(2, 3.0, 0.05, 180)     # player
            + walk(9, 12.0, 0.0, 12)      # someone passing through
        )
        stats = analyse_players(track, fps=30.0, top_n=2)
        assert {s["track_id"] for s in stats} == {1, 2}

    def test_ignores_rows_without_a_track_id(self):
        track = walk(1, 20.0, 0.05, 50) + [
            {"frame": i, "track_id": None, "x_m": 5.0, "y_m": 5.0} for i in range(50)
        ]
        stats = analyse_players(track, fps=30.0)
        assert len(stats) == 1


class TestCoverage:
    def test_grid_counts_land_where_the_player_was(self):
        grid = court_coverage(walk(1, 21.0, 0.0, 10), track_id=1, bins=6)
        assert sum(sum(row) for row in grid) == 10
        assert grid[-1][2] == 10     # near baseline, middle of the court

    def test_render_is_one_line_per_row(self):
        lines = render_coverage(court_coverage(walk(1, 21.0, 0.0, 10), 1, bins=6))
        assert len(lines) == 6

    def test_empty_coverage_says_so(self):
        assert render_coverage([[0] * 6 for _ in range(6)]) == ["(no positions tracked)"]
