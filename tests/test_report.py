"""Tests for the HTML report.

The report has one property that matters more than its looks: it must never
turn missing data into a confident-looking zero. Most of what follows checks
that a section with nothing to show says so.
"""

from __future__ import annotations

import json

import pytest

from tennis import report as report_module


def full_report() -> dict:
    """A report with every section populated."""
    return {
        "input": {
            "path": "input_videos/match.mp4",
            "fps": 30.0,
            "resolution": "1920x1080",
            "frames_total": 900,
            "frames_processed": 900,
        },
        "performance": {"device": "cuda", "seconds": 96.8, "fps": 9.3},
        "court_calibration": {
            "attempts": 30,
            "reliable": 30,
            "median_reprojection_error_px": 0.29,
            "samples": [],
        },
        "detection": {
            "ball_frames": 884,
            "ball_detection_rate": 0.982,
            "player_observations": 1800,
        },
        "events": {"bounces": 24, "hits": 31},
        "rallies": {
            "rallies_found": 6,
            "points_decided": 5,
            "points_undecided": 1,
            "low_confidence_points": 1,
            "mean_confidence": 0.726,
            "mean_shots_per_rally": 4.2,
            "longest_rally_shots": 9,
            "mean_rally_seconds": 6.4,
            "reasons": {"landed out": 3, "double bounce": 2},
        },
        "serves": {
            "serves_detected": 6,
            "in": 4,
            "faults": 2,
            "double_faults": 1,
            "unknown": 0,
            "first_serve_percentage": 0.667,
            "boxes": {"deuce": 3, "advantage": 1},
            "serves": [
                {
                    "rally_index": 0,
                    "frame": 12,
                    "server": 1,
                    "outcome": "in",
                    "box": "deuce",
                    "expected_box": "deuce",
                    "second_serve": False,
                    "confidence": 0.81,
                    "landing_m": [3.2, 8.1],
                },
                {
                    "rally_index": 1,
                    "frame": 240,
                    "server": 1,
                    "outcome": "fault",
                    "box": None,
                    "expected_box": "advantage",
                    "second_serve": True,
                    "confidence": 0.7,
                    "landing_m": [9.9, 4.4],
                },
            ],
        },
        "analysis": {
            "players": [
                {
                    "track_id": 1,
                    "side": "near",
                    "frames_tracked": 900,
                    "distance_covered_m": 141.2,
                    "average_speed_kmh": 5.6,
                    "top_speed_kmh": 19.4,
                    "average_position_m": [5.4, 19.8],
                    "net_approaches": 2,
                    "time_at_net_s": 3.1,
                    "coverage": ["  @@  "],
                    "coverage_grid": [[0, 1, 4], [2, 9, 3], [0, 0, 1]],
                },
                {
                    "track_id": 2,
                    "side": "far",
                    "frames_tracked": 880,
                    "distance_covered_m": 128.4,
                    "average_speed_kmh": 5.1,
                    "top_speed_kmh": 17.8,
                    "average_position_m": [5.1, 3.4],
                    "net_approaches": 0,
                    "time_at_net_s": 0.0,
                    "coverage": ["@@    "],
                    "coverage_grid": [[3, 2, 0], [0, 5, 1], [0, 0, 0]],
                },
            ],
            "ball": {"frames_detected": 884, "mean_confidence": 0.61},
        },
        "placement": {
            "total_landings": 3,
            "in_bounds": 2,
            "out_of_bounds": 1,
            "overall": {
                "landings": 2,
                "mean_depth_m": 7.4,
                "depth_bands": {"short": 0, "mid": 1, "deep": 1},
                "width_bands": {"left": 1, "centre": 0, "right": 1},
                "directions": {"cross-court": 2},
                "deep_share": 0.5,
            },
            "by_player": {
                "1": {
                    "landings": 2,
                    "mean_depth_m": 7.4,
                    "depth_bands": {"short": 0, "mid": 1, "deep": 1},
                    "width_bands": {"left": 1, "centre": 0, "right": 1},
                    "directions": {"cross-court": 2},
                    "deep_share": 0.5,
                }
            },
            "grids": {"far": [[0]], "near": [[0]]},
            "landings": [
                {
                    "frame": 100,
                    "x_m": 2.4,
                    "y_m": 4.1,
                    "side": "far",
                    "depth_m": 7.8,
                    "depth_band": "deep",
                    "width_band": "left",
                    "in_bounds": True,
                    "hit_by": 1,
                    "direction": "cross-court",
                    "confidence": 0.8,
                },
                {
                    "frame": 160,
                    "x_m": 8.1,
                    "y_m": 6.0,
                    "side": "far",
                    "depth_m": 5.9,
                    "depth_band": "mid",
                    "width_band": "right",
                    "in_bounds": True,
                    "hit_by": 1,
                    "direction": "cross-court",
                    "confidence": 0.7,
                },
                {
                    "frame": 220,
                    "x_m": 11.5,
                    "y_m": 2.0,
                    "side": "far",
                    "depth_m": 9.9,
                    "depth_band": "deep",
                    "width_band": "right",
                    "in_bounds": False,
                    "hit_by": 2,
                    "direction": "down-the-line",
                    "confidence": 0.6,
                },
            ],
        },
        "score": {
            "scoreline": "1-0 | 40-30",
            "sets": [{"1": 1, "2": 0}],
            "sets_won": {"1": 0, "2": 0},
            "current_game": "40-30",
            "server": 1,
            "points_played": 5,
            "winner": None,
            "low_confidence_points": 1,
        },
        "points": [
            {
                "winner": 1,
                "reason": "landed out",
                "confidence": 0.82,
                "start_frame": 10,
                "end_frame": 190,
                "shots": 5,
            },
            {
                "winner": 2,
                "reason": "double bounce",
                "confidence": 0.41,
                "start_frame": 200,
                "end_frame": 340,
                "shots": 3,
            },
        ],
    }


def empty_report() -> dict:
    """What the pipeline emits for a clip where nothing could be measured."""
    return {
        "input": {
            "path": "input_videos/short.mp4",
            "fps": 30.0,
            "resolution": "1920x1080",
            "frames_total": 12,
            "frames_processed": 12,
        },
        "performance": {"device": "cpu", "seconds": 78.7, "fps": 0.15},
        "court_calibration": {
            "attempts": 0,
            "reliable": 0,
            "median_reprojection_error_px": None,
            "samples": [],
        },
        "detection": {
            "ball_frames": 0,
            "ball_detection_rate": 0.0,
            "player_observations": 0,
        },
        "events": {"bounces": 0, "hits": 0},
        "rallies": {
            "rallies_found": 0,
            "points_decided": 0,
            "points_undecided": 0,
            "low_confidence_points": 0,
            "mean_confidence": 0.0,
            "mean_shots_per_rally": 0.0,
            "mean_rally_seconds": 0.0,
            "reasons": {},
        },
        "analysis": {"players": [], "ball": {"frames_detected": 0, "mean_confidence": 0.0}},
        "score": {
            "scoreline": "0-0 | 0-0",
            "sets": [{"1": 0, "2": 0}],
            "sets_won": {"1": 0, "2": 0},
            "current_game": "0-0",
            "server": 1,
            "points_played": 0,
            "winner": None,
            "low_confidence_points": 0,
        },
        "points": [],
    }


# --- structure -----------------------------------------------------------


def test_renders_a_complete_html_document():
    html = report_module.render(full_report())
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<body>") == 1 and html.count("</body>") == 1


def test_is_self_contained_no_network_references():
    # The report's whole value is that it opens anywhere, years later.
    html = report_module.render(full_report())
    for pattern in ("http://", "https://", "<script", "src="):
        assert pattern not in html


def test_renders_the_scoreline():
    html = report_module.render(full_report())
    assert "1-0 | 40-30" in html


def test_all_sections_present_when_data_is():
    html = report_module.render(full_report())
    for heading in ("Run summary", "Points", "Shot placement", "Serves", "Players"):
        assert heading in html


# --- the important property: no invented numbers -------------------------


def test_empty_report_says_no_completed_point_rather_than_zero_zero():
    html = report_module.render(empty_report())
    assert "no completed point" in html
    # Not a scoreline banner pretending a game is in progress.
    assert '<div class="scoreline">' not in html


def test_empty_report_flags_that_the_court_was_never_calibrated():
    html = report_module.render(empty_report())
    assert "never calibrated" in html
    assert "unavailable, not zero" in html


def test_missing_sections_do_not_raise():
    # Reports written before placement/serve analysis existed must still render.
    html = report_module.render(empty_report())
    assert "No serve was identified" in html
    assert "No ball landings" in html


def test_poor_detection_rate_raises_a_warning():
    data = full_report()
    data["detection"]["ball_detection_rate"] = 0.44
    html = report_module.render(data)
    assert "44% of frames" in html


def test_good_run_raises_no_flags():
    data = full_report()
    data["rallies"]["low_confidence_points"] = 0
    assert "No quality flags raised" in report_module.render(data)


def test_timeline_places_bars_by_frame_not_by_order():
    # Point 1 spans frames 10-190 of a 900-frame clip, so its bar must start
    # near the left edge and take roughly a fifth of the width.
    html = report_module.render(full_report())
    bar = html.split('<span class="tl-bar ')[1]
    left = float(bar.split("left:")[1].split("%")[0])
    width = float(bar.split("width:")[1].split("%")[0])
    assert left == pytest.approx(10 / 900 * 100, abs=0.1)
    assert width == pytest.approx(180 / 900 * 100, abs=0.1)


def test_timeline_colours_bars_by_winner_and_flags_low_confidence():
    html = report_module.render(full_report())
    assert "tl-bar p1" in html
    assert "tl-bar p2 low" in html  # point 2 was won at confidence 0.41


def test_timeline_is_omitted_when_there_are_no_points():
    # The class names live in the stylesheet regardless; what must be absent
    # is the markup.
    assert '<div class="tl-track">' not in report_module.render(empty_report())


def test_wide_tables_can_scroll_rather_than_overflow_the_page():
    html = report_module.render(full_report())
    assert '<div class="scroll"><table class="points"' in html


def test_low_confidence_point_is_marked_in_the_table():
    html = report_module.render(full_report())
    assert "low confidence" in html
    assert 'class="low"' in html


def test_unreliable_calibration_is_reported():
    data = full_report()
    data["court_calibration"]["reliable"] = 22
    html = report_module.render(data)
    assert "unreliable on 8 of 30" in html


# --- court plan ----------------------------------------------------------


def test_landings_are_plotted_in_and_out_distinctly():
    html = report_module.render(full_report())
    assert 'class="mark in"' in html
    assert 'class="mark out"' in html


def test_out_of_bounds_landing_is_drawn_outside_the_court_rectangle():
    # x = 11.5 m is beyond the 10.97 m doubles width, so its centre must fall
    # to the right of the court rect. This is the check that the plan is drawn
    # to scale rather than normalised into the box.
    html = report_module.render(full_report())
    right_edge = report_module.PAD + report_module.DOUBLES_WIDTH * report_module.SCALE
    marks = [
        float(chunk.split('cx="')[1].split('"')[0])
        for chunk in html.split('<circle class="mark ')[1:]
    ]
    assert max(marks) > right_edge


def test_coverage_grid_is_drawn_and_scales_with_occupancy():
    html = report_module.render(full_report())
    assert 'class="heat"' in html
    assert "9 frames" in html  # the peak cell's tooltip


def test_player_without_a_coverage_grid_degrades_gracefully():
    data = full_report()
    del data["analysis"]["players"][0]["coverage_grid"]
    html = report_module.render(data)
    assert "coverage grid not recorded" in html


# --- escaping and io -----------------------------------------------------


def test_untrusted_text_is_escaped():
    data = full_report()
    data["points"][0]["reason"] = "<script>alert(1)</script>"
    html = report_module.render(data)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_creates_the_file_and_parent_directory(tmp_path):
    target = tmp_path / "nested" / "report.html"
    written = report_module.write(full_report(), target)
    assert written == target
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_cli_renders_from_a_json_file(tmp_path, capsys):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(full_report()), encoding="utf-8")
    assert report_module.main([str(source)]) == 0
    assert (tmp_path / "report.html").exists()
    assert "wrote" in capsys.readouterr().out


def test_cli_honours_an_explicit_output_path(tmp_path):
    source = tmp_path / "report.json"
    source.write_text(json.dumps(empty_report()), encoding="utf-8")
    target = tmp_path / "custom.html"
    assert report_module.main([str(source), "-o", str(target)]) == 0
    assert target.exists()


@pytest.mark.parametrize("missing", ["analysis", "placement", "serves", "points",
                                     "rallies", "score", "detection"])
def test_render_survives_any_single_missing_section(missing):
    data = full_report()
    data.pop(missing, None)
    assert report_module.render(data).startswith("<!doctype html>")
