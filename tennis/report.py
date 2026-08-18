"""HTML match report.

The JSON report is the machine-readable artefact; this is the one a human
looks at. It is deliberately a single self-contained file - no CSS framework,
no charting library, no network fetch - because the whole point is that it can
be attached to an email, opened from a USB stick, or committed next to the
video and still render years later.

Everything it draws comes out of ``report.json``, so it can be regenerated
without re-running inference:

    python -m tennis.report output/report.json -o output/report.html

Two rendering choices are worth stating:

- **The court diagrams are drawn in court metres, not image pixels.** Landing
  positions arrive from the homography in real coordinates, so the diagram is
  a true scale plan of the court rather than a redrawing of the camera view.
  A bounce plotted 2 m inside the baseline really was 2 m inside it.
- **Uncertainty is shown, never smoothed away.** Low-confidence points are
  marked in the points table, an unreliable calibration raises a banner at the
  top, and sections with no data say so instead of rendering an empty chart
  that reads as a zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

from tennis.court import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NET_Y,
    SERVICE_LINE_FROM_NET,
    SINGLES_WIDTH,
)

ALLEY = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2
LOW_CONFIDENCE = 0.6

# Pixels per court metre in the SVG plans. 24 puts a full court at roughly
# 263 x 570 px, which fits a phone screen without scrolling sideways.
SCALE = 24.0
PAD = 18.0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _fmt(value, suffix: str = "", digits: int = 1) -> str:
    """Format a number for display, or an em dash when it is missing."""
    if value is None:
        return "&mdash;"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value}{suffix}"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return escape(str(value))


def _pct(value) -> str:
    return "&mdash;" if value is None else f"{value * 100:.1f}%"


def _tile(label: str, value: str, note: str = "", tone: str = "") -> str:
    cls = f"tile {tone}".strip()
    note_html = f'<div class="tile-note">{note}</div>' if note else ""
    return (
        f'<div class="{cls}"><div class="tile-value">{value}</div>'
        f'<div class="tile-label">{escape(label)}</div>{note_html}</div>'
    )


def _section(title: str, body: str, subtitle: str = "") -> str:
    if not body:
        return ""
    sub = f'<p class="sub">{escape(subtitle)}</p>' if subtitle else ""
    return f'<section><h2>{escape(title)}</h2>{sub}{body}</section>'


def _empty(message: str) -> str:
    return f'<p class="empty">{escape(message)}</p>'


# --------------------------------------------------------------------------
# court plan
# --------------------------------------------------------------------------


def _court_svg(marks: list[dict], title: str = "") -> str:
    """A scale plan of the court with markers plotted in court metres.

    Each mark is ``{x, y, kind, label}`` where x/y are court metres and kind
    selects the styling ('in', 'out', 'serve').
    """
    width = DOUBLES_WIDTH * SCALE + 2 * PAD
    height = COURT_LENGTH * SCALE + 2 * PAD

    def px(x: float) -> float:
        return PAD + x * SCALE

    def py(y: float) -> float:
        return PAD + y * SCALE

    lines: list[str] = []

    def line(x1: float, y1: float, x2: float, y2: float, cls: str = "l") -> None:
        lines.append(
            f'<line class="{cls}" x1="{px(x1):.1f}" y1="{py(y1):.1f}" '
            f'x2="{px(x2):.1f}" y2="{py(y2):.1f}"/>'
        )

    # doubles boundary
    lines.append(
        f'<rect class="surface" x="{px(0):.1f}" y="{py(0):.1f}" '
        f'width="{DOUBLES_WIDTH * SCALE:.1f}" height="{COURT_LENGTH * SCALE:.1f}"/>'
    )
    # singles tramlines
    line(ALLEY, 0, ALLEY, COURT_LENGTH)
    line(DOUBLES_WIDTH - ALLEY, 0, DOUBLES_WIDTH - ALLEY, COURT_LENGTH)
    # service lines and centre service line
    far_service = NET_Y - SERVICE_LINE_FROM_NET
    near_service = NET_Y + SERVICE_LINE_FROM_NET
    line(ALLEY, far_service, DOUBLES_WIDTH - ALLEY, far_service)
    line(ALLEY, near_service, DOUBLES_WIDTH - ALLEY, near_service)
    line(DOUBLES_WIDTH / 2, far_service, DOUBLES_WIDTH / 2, near_service)
    # net
    line(-0.6, NET_Y, DOUBLES_WIDTH + 0.6, NET_Y, "net")

    dots: list[str] = []
    for mark in marks:
        cx, cy = px(float(mark["x"])), py(float(mark["y"]))
        kind = mark.get("kind", "in")
        label = escape(str(mark.get("label", "")))
        radius = 5.5 if kind == "serve" else 4.5
        dots.append(
            f'<circle class="mark {kind}" cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}">'
            f"<title>{label}</title></circle>"
        )

    caption = f'<figcaption>{escape(title)}</figcaption>' if title else ""
    return (
        f'<figure class="court">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="{escape(title or "court plan")}">'
        f'{"".join(lines)}{"".join(dots)}'
        f'<text class="axis" x="{px(DOUBLES_WIDTH / 2):.1f}" y="{py(0) - 5:.1f}" '
        f'text-anchor="middle">far baseline</text>'
        f'<text class="axis" x="{px(DOUBLES_WIDTH / 2):.1f}" '
        f'y="{py(COURT_LENGTH) + 13:.1f}" text-anchor="middle">near baseline</text>'
        f"</svg>{caption}</figure>"
    )


def _coverage_svg(grid: list[list[int]], label: str) -> str:
    """A player's occupancy grid, drawn over the court outline."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return _empty("no positions tracked")
    peak = max(max(row) for row in grid) or 1

    cell_w = DOUBLES_WIDTH / cols
    cell_h = COURT_LENGTH / rows
    cells: list[str] = []
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if not value:
                continue
            opacity = 0.12 + 0.78 * (value / peak)
            cells.append(
                f'<rect class="heat" x="{PAD + c * cell_w * SCALE:.1f}" '
                f'y="{PAD + r * cell_h * SCALE:.1f}" '
                f'width="{cell_w * SCALE:.1f}" height="{cell_h * SCALE:.1f}" '
                f'opacity="{opacity:.2f}"><title>{value} frames</title></rect>'
            )

    width = DOUBLES_WIDTH * SCALE + 2 * PAD
    height = COURT_LENGTH * SCALE + 2 * PAD
    net_y = PAD + NET_Y * SCALE
    return (
        f'<figure class="court small">'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{escape(label)}">'
        f'<rect class="surface" x="{PAD}" y="{PAD}" '
        f'width="{DOUBLES_WIDTH * SCALE:.1f}" height="{COURT_LENGTH * SCALE:.1f}"/>'
        f'{"".join(cells)}'
        f'<line class="net" x1="{PAD - 14:.1f}" y1="{net_y:.1f}" '
        f'x2="{PAD + DOUBLES_WIDTH * SCALE + 14:.1f}" y2="{net_y:.1f}"/>'
        f"</svg><figcaption>{escape(label)}</figcaption></figure>"
    )


def _bar(counts: dict, order: list[str] | None = None) -> str:
    """A horizontal bar row for a small categorical breakdown."""
    keys = order or list(counts)
    keys = [k for k in keys if counts.get(k)]
    if not keys:
        return ""
    total = sum(counts.get(k, 0) for k in keys) or 1
    bars = "".join(
        f'<div class="bar-row"><span class="bar-label">{escape(k)}</span>'
        f'<span class="bar-track"><span class="bar-fill" '
        f'style="width:{counts[k] / total * 100:.1f}%"></span></span>'
        f'<span class="bar-value">{counts[k]}</span></div>'
        for k in keys
    )
    return f'<div class="bars">{bars}</div>'


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _header(report: dict) -> str:
    info = report.get("input", {})
    score = report.get("score", {})
    rallies = report.get("rallies", {})
    decided = rallies.get("points_decided", 0)

    if decided:
        scoreline = escape(str(score.get("scoreline", "")))
        winner = score.get("winner")
        note = (
            f"match won by player {winner}"
            if winner
            else f"{score.get('points_played', 0)} points played, "
            f"player {score.get('server')} serving"
        )
        banner = (
            f'<div class="scoreline">{scoreline}</div>'
            f'<div class="score-note">{escape(note)}</div>'
        )
    else:
        banner = (
            '<div class="scoreline muted">no completed point</div>'
            '<div class="score-note">the clip contains no rally that ends in a '
            "decidable outcome &mdash; no score is reported rather than a guessed "
            "one</div>"
        )

    source = escape(Path(str(info.get("path", "?"))).name)
    meta = " &middot; ".join(
        [
            source,
            f"{info.get('frames_processed', 0)} frames",
            escape(str(info.get("resolution", "?"))),
            f"{_fmt(info.get('fps'), ' fps', 0)}",
        ]
    )
    return (
        f"<header><h1>Tennis match report</h1>"
        f'<p class="sub">{meta}</p>'
        f'<div class="score-card">{banner}</div></header>'
    )


def _quality(report: dict) -> str:
    """Warnings that must be read before any number below is trusted."""
    warnings: list[str] = []
    court = report.get("court_calibration", {})
    attempts, reliable = court.get("attempts", 0), court.get("reliable", 0)
    if attempts and reliable < attempts:
        warnings.append(
            f"court calibration was unreliable on {attempts - reliable} of "
            f"{attempts} attempts &mdash; positions in those stretches carry the "
            "last good homography"
        )
    if not attempts:
        warnings.append(
            "the court was never calibrated: every distance and landing position "
            "below is unavailable, not zero"
        )
    rate = report.get("detection", {}).get("ball_detection_rate")
    if rate is not None and rate < 0.8:
        warnings.append(
            f"the ball was found in only {rate:.0%} of frames &mdash; bounce and "
            "hit detection degrade quickly below about 80%"
        )
    low = report.get("rallies", {}).get("low_confidence_points", 0)
    if low:
        warnings.append(
            f"{low} point(s) were attributed with low confidence and are flagged "
            "in the table below"
        )
    if not warnings:
        return '<div class="banner ok">No quality flags raised on this run.</div>'
    items = "".join(f"<li>{w}</li>" for w in warnings)
    return f'<div class="banner warn"><ul>{items}</ul></div>'


def _run_summary(report: dict) -> str:
    detection = report.get("detection", {})
    court = report.get("court_calibration", {})
    perf = report.get("performance", {})
    events = report.get("events", {})
    rallies = report.get("rallies", {})

    rate = detection.get("ball_detection_rate")
    error = court.get("median_reprojection_error_px")
    tiles = [
        _tile(
            "ball detection rate",
            _pct(rate),
            f"{detection.get('ball_frames', 0)} frames",
            "good" if (rate or 0) >= 0.8 else "poor",
        ),
        _tile(
            "court reprojection error",
            _fmt(error, " px", 2),
            f"{court.get('reliable', 0)}/{court.get('attempts', 0)} reliable",
            "good" if (error is not None and error < 3) else "poor",
        ),
        _tile(
            "points decided",
            str(rallies.get("points_decided", 0)),
            f"{rallies.get('rallies_found', 0)} rallies found",
        ),
        _tile(
            "mean confidence",
            _fmt(rallies.get("mean_confidence"), "", 3),
            "over decided points",
        ),
        _tile(
            "events",
            f"{events.get('bounces', 0)} / {events.get('hits', 0)}",
            "bounces / hits",
        ),
        _tile(
            "throughput",
            _fmt(perf.get("fps"), " fps", 2),
            f"{escape(str(perf.get('device', '?')))}, "
            f"{_fmt(perf.get('seconds'), ' s')}",
        ),
    ]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _timeline(report: dict) -> str:
    """Every rally laid out against the clip's real duration.

    This is the part that makes the report usable next to the video: each bar
    is positioned by frame, so a suspicious point can be found and watched
    rather than argued about. Gaps between bars are dead time - warm-up,
    walking, ball retrieval - and are as informative as the bars.
    """
    points = report.get("points") or []
    info = report.get("input", {})
    fps = info.get("fps") or 30.0
    total = info.get("frames_total") or info.get("frames_processed") or 0
    span = max([p.get("end_frame", 0) for p in points] + [total])
    if not points or not span:
        return ""

    bars: list[str] = []
    for index, point in enumerate(points, start=1):
        start, end = point.get("start_frame", 0), point.get("end_frame", 0)
        left = start / span * 100
        width = max((end - start) / span * 100, 0.8)
        winner = point.get("winner")
        confidence = point.get("confidence")
        low = confidence is not None and confidence < LOW_CONFIDENCE
        tone = "undecided" if winner is None else f"p{winner}"
        bars.append(
            f'<span class="tl-bar {tone}{" low" if low else ""}" '
            f'style="left:{left:.2f}%;width:{width:.2f}%" '
            f'title="point {index}: '
            f'{"player " + str(winner) if winner is not None else "undecided"}, '
            f'{escape(str(point.get("reason", "")))}, '
            f'{start / fps:.1f}-{end / fps:.1f} s"></span>'
        )

    ticks = "".join(
        f'<span class="tl-tick" style="left:{i / 4 * 100:.0f}%">'
        f"{span / fps * i / 4:.0f}s</span>"
        for i in range(5)
    )
    legend = (
        '<div class="tl-legend">'
        '<span class="key p1"></span>player 1'
        '<span class="key p2"></span>player 2'
        '<span class="key undecided"></span>undecided'
        '<span class="key low"></span>low confidence'
        "</div>"
    )
    return (
        f'<div class="timeline"><div class="tl-track">{"".join(bars)}</div>'
        f'<div class="tl-axis">{ticks}</div>{legend}</div>'
    )


def _points(report: dict) -> str:
    points = report.get("points", [])
    if not points:
        return _empty("No rally in this clip reached a decidable outcome.")

    fps = report.get("input", {}).get("fps") or 30.0
    rows: list[str] = []
    for index, point in enumerate(points, start=1):
        winner = point.get("winner")
        confidence = point.get("confidence")
        low = confidence is not None and confidence < LOW_CONFIDENCE
        flag = ' <span class="flag">low confidence</span>' if low else ""
        start, end = point.get("start_frame", 0), point.get("end_frame", 0)
        rows.append(
            f'<tr class="{"low" if low else ""}">'
            f"<td>{index}</td>"
            f"<td>{'player ' + str(winner) if winner is not None else '&mdash;'}</td>"
            f"<td>{escape(str(point.get('reason', '')))}{flag}</td>"
            f"<td>{point.get('shots', 0)}</td>"
            f"<td>{(end - start) / fps:.1f} s</td>"
            f'<td class="mono">{start}&ndash;{end}</td>'
            f"<td>{_fmt(confidence, '', 3)}</td>"
            f"</tr>"
        )

    reasons = report.get("rallies", {}).get("reasons", {})
    breakdown = _bar(reasons) if reasons else ""
    rally_stats = report.get("rallies", {})
    tiles = (
        '<div class="tiles compact">'
        + _tile("longest rally", f"{rally_stats.get('longest_rally_shots', 0)} shots")
        + _tile("mean rally", _fmt(rally_stats.get("mean_rally_seconds"), " s"))
        + _tile("shots per rally", _fmt(rally_stats.get("mean_shots_per_rally")))
        + _tile("undecided", str(rally_stats.get("points_undecided", 0)))
        + "</div>"
    )
    return (
        tiles
        + _timeline(report)
        + '<div class="scroll"><table class="points"><thead><tr>'
        "<th>#</th><th>winner</th><th>reason</th><th>shots</th><th>length</th>"
        "<th>frames</th><th>conf.</th>"
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>{breakdown}'
    )


def _placement(report: dict) -> str:
    placement = report.get("placement") or {}
    landings = placement.get("landings") or []
    if not landings:
        return _empty("No ball landings were located on the court plane.")

    marks = [
        {
            "x": l["x_m"],
            "y": l["y_m"],
            "kind": "in" if l.get("in_bounds") else "out",
            "label": (
                f"frame {l.get('frame')} &middot; {l.get('depth_band')} "
                f"{l.get('width_band')} &middot; "
                f"{'in' if l.get('in_bounds') else 'out'}"
            ),
        }
        for l in landings
    ]
    plan = _court_svg(marks, "ball landings, in court metres")

    overall = placement.get("overall", {})
    stats = (
        f'<div class="tiles compact">'
        + _tile("landings in", str(placement.get("in_bounds", 0)))
        + _tile("landings out", str(placement.get("out_of_bounds", 0)))
        + _tile("mean depth", _fmt(overall.get("mean_depth_m"), " m", 2), "from net")
        + _tile("deep share", _pct(overall.get("deep_share")))
        + "</div>"
    )

    breakdowns = ""
    if overall.get("landings"):
        breakdowns = (
            '<div class="grid-2">'
            f'<div><h3>depth</h3>{_bar(overall.get("depth_bands", {}), ["short", "mid", "deep"])}</div>'
            f'<div><h3>width</h3>{_bar(overall.get("width_bands", {}), ["left", "centre", "right"])}</div>'
            "</div>"
        )
        directions = overall.get("directions") or {}
        if directions:
            breakdowns += f"<h3>direction</h3>{_bar(directions)}"

    per_player = placement.get("by_player") or {}
    table = ""
    if per_player:
        rows = "".join(
            f"<tr><td>player {escape(str(track_id))}</td>"
            f"<td>{stat.get('landings', 0)}</td>"
            f"<td>{_fmt(stat.get('mean_depth_m'), ' m', 2)}</td>"
            f"<td>{stat.get('depth_bands', {}).get('deep', 0)}</td>"
            f"<td>{stat.get('depth_bands', {}).get('mid', 0)}</td>"
            f"<td>{stat.get('depth_bands', {}).get('short', 0)}</td>"
            f"<td>{escape(', '.join(f'{k} {v}' for k, v in (stat.get('directions') or {}).items()) or '—')}</td>"
            "</tr>"
            for track_id, stat in sorted(per_player.items())
        )
        table = (
            '<div class="scroll"><table><thead><tr><th>player</th><th>shots</th>'
            "<th>mean depth</th><th>deep</th><th>mid</th><th>short</th>"
            "<th>direction</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    return f'{stats}<div class="split">{plan}<div class="split-body">{breakdowns}{table}</div></div>'


def _players(report: dict) -> str:
    players = (report.get("analysis") or {}).get("players") or []
    if not players:
        return _empty("No player was tracked for long enough to measure.")

    cards: list[str] = []
    for player in players:
        grid = player.get("coverage_grid")
        heat = (
            _coverage_svg(grid, f"player {player.get('track_id')} court coverage")
            if grid
            else _empty("coverage grid not recorded in this report")
        )
        position = player.get("average_position_m") or [None, None]
        cards.append(
            f'<div class="card">'
            f"<h3>Player {escape(str(player.get('track_id')))} "
            f'<span class="side">{escape(str(player.get("side", "")))} side</span></h3>'
            f'<div class="split">{heat}<div class="split-body">'
            f'<dl class="stats">'
            f"<dt>shots hit</dt><dd>{player.get('shots_hit', 0)}</dd>"
            f"<dt>distance covered</dt><dd>{_fmt(player.get('distance_covered_m'), ' m')}</dd>"
            f"<dt>average speed</dt><dd>{_fmt(player.get('average_speed_kmh'), ' km/h')}</dd>"
            f"<dt>top speed</dt><dd>{_fmt(player.get('top_speed_kmh'), ' km/h')}</dd>"
            f"<dt>net approaches</dt><dd>{player.get('net_approaches', 0)}</dd>"
            f"<dt>time at net</dt><dd>{_fmt(player.get('time_at_net_s'), ' s')}</dd>"
            f"<dt>average position</dt>"
            f"<dd>{_fmt(position[0], '', 2)}, {_fmt(position[1], '', 2)} m</dd>"
            f"<dt>frames tracked</dt><dd>{player.get('frames_tracked', 0)}</dd>"
            f"</dl></div></div></div>"
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def _serves(report: dict) -> str:
    serves = report.get("serves") or {}
    if not serves.get("serves_detected"):
        return _empty("No serve was identified in this clip.")

    tiles = (
        '<div class="tiles compact">'
        + _tile("serves", str(serves.get("serves_detected", 0)))
        + _tile("in", str(serves.get("in", 0)), tone="good")
        + _tile("faults", str(serves.get("faults", 0)))
        + _tile("double faults", str(serves.get("double_faults", 0)), tone="poor")
        + _tile("first serve %", _pct(serves.get("first_serve_percentage")))
        + "</div>"
    )

    marks = [
        {
            "x": s["landing_m"][0],
            "y": s["landing_m"][1],
            "kind": "serve" if s.get("outcome") == "in" else "out",
            "label": (
                f"frame {s.get('frame')} &middot; player {s.get('server')} "
                f"&middot; {s.get('outcome')}"
                + (" &middot; 2nd" if s.get("second_serve") else "")
            ),
        }
        for s in serves.get("serves", [])
        if s.get("landing_m")
    ]
    plan = _court_svg(marks, "serve landings") if marks else ""
    boxes = _bar(serves.get("boxes") or {})
    body = f'<div>{"<h3>service box</h3>" + boxes if boxes else ""}</div>'
    return tiles + (f'<div class="split">{plan}<div class="split-body">{body}</div></div>' if plan else boxes)


def _method(report: dict) -> str:
    court = report.get("court_calibration", {})
    perf = report.get("performance", {})
    return (
        "<ul class=\"method\">"
        "<li><strong>Court geometry.</strong> 14 keypoints are regressed per frame "
        f"and fitted to a homography; the median reprojection error on this run was "
        f"{_fmt(court.get('median_reprojection_error_px'), ' px', 2)} over "
        f"{court.get('attempts', 0)} calibration attempts. Every distance on this "
        "page is measured on the court plane in metres, not in image pixels.</li>"
        "<li><strong>Ground contact.</strong> A homography maps the court plane, so "
        "an airborne ball projects beyond where it truly is. A local maximum in "
        "projected court y is therefore ground contact and a local minimum is the "
        "apex of the arc &mdash; which is why bounces are judged in court metres "
        "while player-ball proximity is judged in image pixels.</li>"
        "<li><strong>Scoring.</strong> Points are awarded from rally outcomes "
        "&mdash; double bounce, landed out, failed to cross the net &mdash; never "
        "from a line call. A single camera cannot adjudicate a line, so the system "
        "does not pretend to.</li>"
        f"<li><strong>This run.</strong> {escape(str(perf.get('device', '?')))} at "
        f"{_fmt(perf.get('fps'), ' fps', 2)}, "
        f"{_fmt(perf.get('seconds'), ' s')} wall clock.</li>"
        "</ul>"
    )


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

STYLE = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #16191d; --muted: #6b7280;
  --line: #e2e5ea; --accent: #1f7a4d; --warn: #b45309; --bad: #b91c1c;
  --court: #dfe8ea; --courtline: #8c9aa0; --in: #1f7a4d; --out: #b91c1c;
  --serve: #1d4ed8; --heat: #1f7a4d;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101317; --panel: #171b21; --ink: #e7eaee; --muted: #99a1ad;
    --line: #262c35; --accent: #4ade80; --warn: #fbbf24; --bad: #f87171;
    --court: #1f2a2c; --courtline: #5d6b71; --in: #4ade80; --out: #f87171;
    --serve: #7aa2ff; --heat: #4ade80;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--ink);
  font: 15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
main { max-width: 1060px; margin: 0 auto; }
h1 { font-size: 1.65rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.05rem; margin: 0 0 .85rem; text-transform: uppercase;
     letter-spacing: .08em; color: var(--muted); }
h3 { font-size: .95rem; margin: 1rem 0 .5rem; }
.sub { color: var(--muted); margin: 0 0 1rem; font-size: .9rem; }
section { background: var(--panel); border: 1px solid var(--line);
          border-radius: 12px; padding: 1.25rem; margin: 1rem 0; }
header { margin-bottom: 1.25rem; }
.score-card { background: var(--panel); border: 1px solid var(--line);
              border-radius: 12px; padding: 1.5rem; text-align: center; }
.scoreline { font-size: 2.4rem; font-weight: 650; letter-spacing: -.02em;
             font-variant-numeric: tabular-nums; }
.scoreline.muted { font-size: 1.4rem; color: var(--muted); font-weight: 500; }
.score-note { color: var(--muted); font-size: .88rem; margin-top: .4rem; }
.banner { border-radius: 10px; padding: .85rem 1.1rem; font-size: .9rem;
          border: 1px solid var(--line); }
.banner.ok { color: var(--accent); }
.banner.warn { color: var(--warn); border-color: var(--warn); }
.banner ul { margin: 0; padding-left: 1.1rem; }
.tiles { display: grid; gap: .75rem;
         grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.tiles.compact { grid-template-columns: repeat(auto-fit, minmax(115px, 1fr));
                 margin-bottom: 1rem; }
.tile { border: 1px solid var(--line); border-radius: 10px; padding: .8rem .9rem; }
.tile-value { font-size: 1.4rem; font-weight: 620;
              font-variant-numeric: tabular-nums; }
.tile-label { color: var(--muted); font-size: .78rem; text-transform: uppercase;
              letter-spacing: .05em; margin-top: .15rem; }
.tile-note { color: var(--muted); font-size: .78rem; margin-top: .3rem; }
.tile.good .tile-value { color: var(--accent); }
.tile.poor .tile-value { color: var(--bad); }
table { width: 100%; border-collapse: collapse; font-size: .9rem;
        min-width: 420px; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 550; font-size: .78rem;
     text-transform: uppercase; letter-spacing: .05em; }
tr.low td { color: var(--warn); }
.mono { font-variant-numeric: tabular-nums; color: var(--muted); }
.flag { font-size: .72rem; border: 1px solid var(--warn); color: var(--warn);
        border-radius: 999px; padding: .05rem .45rem; margin-left: .35rem; }
.timeline { margin: 0 0 1.25rem; }
.tl-track { position: relative; height: 26px; background: var(--line);
            border-radius: 6px; overflow: hidden; }
.tl-bar { position: absolute; top: 0; height: 100%; border-radius: 3px; }
.tl-bar.p1 { background: var(--accent); }
.tl-bar.p2 { background: var(--serve); }
.tl-bar.undecided { background: var(--muted); }
.tl-bar.low { outline: 2px dashed var(--warn); outline-offset: -2px; }
.tl-axis { position: relative; height: 1.1rem; margin-top: .25rem; }
.tl-tick { position: absolute; transform: translateX(-50%); color: var(--muted);
           font-size: .72rem; font-variant-numeric: tabular-nums; }
.tl-tick:first-child { transform: none; }
.tl-tick:last-child { transform: translateX(-100%); }
.tl-legend { display: flex; gap: .4rem 1rem; flex-wrap: wrap; align-items: center;
             color: var(--muted); font-size: .78rem; margin-top: .5rem; }
.key { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
       margin-right: -.6rem; }
.key.p1 { background: var(--accent); }
.key.p2 { background: var(--serve); }
.key.undecided { background: var(--muted); }
.key.low { border: 2px dashed var(--warn); background: none; }
.split { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: flex-start; }
.split-body { flex: 1 1 320px; min-width: 260px; }
.grid-2 { display: grid; gap: 1rem 1.5rem;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.court { margin: 0; flex: 0 0 auto; max-width: 100%; }
.court svg { width: 240px; max-width: 100%; height: auto; }
.court.small svg { width: 150px; }
.scroll { overflow-x: auto; }
figcaption { color: var(--muted); font-size: .78rem; text-align: center;
             margin-top: .4rem; }
.surface { fill: var(--court); stroke: var(--courtline); stroke-width: 1.5; }
line.l { stroke: var(--courtline); stroke-width: 1.2; }
line.net { stroke: var(--ink); stroke-width: 2; stroke-dasharray: 4 3; }
.mark { stroke: var(--panel); stroke-width: 1.2; }
.mark.in { fill: var(--in); }
.mark.out { fill: var(--out); }
.mark.serve { fill: var(--serve); }
.heat { fill: var(--heat); }
text.axis { fill: var(--muted); font-size: 10px; }
.bars { margin: .5rem 0 1rem; }
.bar-row { display: flex; align-items: center; gap: .5rem; margin: .3rem 0;
           font-size: .85rem; }
.bar-label { flex: 0 0 110px; color: var(--muted); }
.bar-track { flex: 1; height: 8px; background: var(--line); border-radius: 999px; }
.bar-fill { display: block; height: 100%; background: var(--accent);
            border-radius: 999px; }
.bar-value { flex: 0 0 2rem; text-align: right;
             font-variant-numeric: tabular-nums; }
.cards { display: grid; gap: 1rem;
         grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 1rem; }
.card h3 { margin-top: 0; }
.side { color: var(--muted); font-weight: 400; font-size: .82rem; }
dl.stats { display: grid; grid-template-columns: auto 1fr; gap: .3rem .8rem;
           margin: 0; font-size: .88rem; }
dl.stats dt { color: var(--muted); }
dl.stats dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.empty { color: var(--muted); font-size: .9rem; font-style: italic; margin: 0; }
ul.method { margin: 0; padding-left: 1.1rem; font-size: .88rem; color: var(--ink); }
ul.method li { margin-bottom: .5rem; }
footer { color: var(--muted); font-size: .8rem; text-align: center;
         margin-top: 1.5rem; }
"""


def render(report: dict) -> str:
    """Render a report dict as a complete, self-contained HTML page."""
    source = Path(str(report.get("input", {}).get("path", "match"))).name
    body = "".join(
        [
            _header(report),
            _quality(report),
            _section("Run summary", _run_summary(report)),
            _section(
                "Points",
                _points(report),
                "each point, how it was decided, and how sure the system is",
            ),
            _section(
                "Shot placement",
                _placement(report),
                "landings measured at ground contact, where the homography is exact",
            ),
            _section("Serves", _serves(report)),
            _section("Players", _players(report)),
            _section("How these numbers were produced", _method(report)),
            "<footer>Generated by tennis-auto-scoring &middot; "
            "no line calls were used to decide any point.</footer>",
        ]
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Tennis report &mdash; {escape(source)}</title>"
        f"<style>{STYLE}</style></head><body><main>{body}</main></body></html>\n"
    )


def write(report: dict, path: str | Path) -> Path:
    """Render the report and write it to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(report), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tennis.report",
        description="Render report.json as a self-contained HTML page",
    )
    parser.add_argument("report", help="path to report.json")
    parser.add_argument(
        "-o", "--out", default=None, help="output path (default: report.html beside it)"
    )
    args = parser.parse_args(argv)

    source = Path(args.report)
    data = json.loads(source.read_text(encoding="utf-8"))
    target = Path(args.out) if args.out else source.with_suffix(".html")
    write(data, target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
