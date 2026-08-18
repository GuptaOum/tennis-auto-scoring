"""Calibrate the bounce detector's prominence threshold against real footage.

Why not tune it on the synthetic fixture
----------------------------------------
The threshold used to be 1.5 m, chosen while the prominence test was measuring
an unbounded absolute excursion - under which it barely constrained anything.
Once prominence was measured correctly, 1.5 m rejected real contacts, and the
replacement value of 0.6 m came from a synthetic arc in the test suite. A
synthetic arc has one height scale and one travel rate; a real rally has many.
So the number has to be read off a real trajectory or it is guesswork twice
over.

What it is calibrated against
-----------------------------
The Wimbledon test clip carries its own answer key: the broadcast score bug is
on screen, so the frames at which points actually ended are known. Murray
serving to Federer at 4-5, 1 set down:

    point 1 ends between 17 s and 18 s   -> frame 510-540
    point 2 ends between 36 s and 37 s   -> frame 1080-1110
    point 3 is still in play at 48 s     -> no boundary expected

A threshold is good when rally segmentation puts a point boundary near each of
those and does not invent others. That is a stricter test than counting
bounces, because it scores the thing the system exists to produce.

Usage
-----
    python -m training.calibrate_bounce --run output/sample_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tennis import rally as rally_module
from tennis import serve as serve_module
from tennis.bounce import EventType, detect_events
from tennis.trajectory import from_detections

# Frame at which each completed point ends, read off the broadcast scoreboard,
# with the tolerance allowed when matching a detected boundary to it. 45 frames
# is 1.5 s at 30 fps - the graphic itself updates a beat after the point ends,
# so demanding better than that would be measuring the broadcast's latency.
GROUND_TRUTH_ENDS = [525, 1095]
TOLERANCE = 45


def load(run_dir: Path) -> tuple[list[dict], dict]:
    ball_track = json.loads((run_dir / "ball_track.json").read_text(encoding="utf-8"))
    tracks_path = run_dir / "tracks.json"
    if not tracks_path.exists():
        raise SystemExit(
            f"{tracks_path} not found - re-run the pipeline so it writes the "
            "player boxes this sweep needs"
        )
    raw = json.loads(tracks_path.read_text(encoding="utf-8"))
    player_boxes = {
        int(track_id): {int(frame): tuple(box) for frame, box in by_frame.items()}
        for track_id, by_frame in raw.items()
    }
    return ball_track, player_boxes


def score(prominence: float, ball_track: list[dict], player_boxes: dict,
          fps: float) -> dict:
    trajectory = from_detections(ball_track)
    events = detect_events(
        trajectory,
        player_boxes=player_boxes,
        fps=fps,
        min_ground_prominence=prominence,
    )
    provisional = rally_module.segment(events, fps=fps)
    _, faults, doubles = serve_module.analyse_serves(provisional, {})
    match, rallies = rally_module.score_match(
        events, fps=fps, fault_indices=set(faults), double_faults=dict(doubles)
    )

    ends = [r.end_frame for r in rallies]
    matched, errors = 0, []
    for truth in GROUND_TRUTH_ENDS:
        nearest = min(ends, key=lambda e: abs(e - truth)) if ends else None
        if nearest is not None and abs(nearest - truth) <= TOLERANCE:
            matched += 1
            errors.append(abs(nearest - truth))

    return {
        "prominence": prominence,
        "bounces": sum(1 for e in events if e.type is EventType.BOUNCE),
        "hits": sum(1 for e in events if e.type is EventType.HIT),
        "rallies": len(rallies),
        "decided": sum(1 for r in rallies if r.winner is not None),
        "truth_matched": matched,
        "mean_boundary_error": (
            round(float(np.mean(errors)), 1) if errors else None
        ),
        # Boundaries that match nothing in the answer key. A threshold that
        # finds both real points by also proposing fifteen others has not
        # solved anything.
        "spurious": max(0, len(rallies) - matched),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="calibrate_bounce")
    parser.add_argument("--run", required=True, help="a finished output/ run dir")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--values", type=float, nargs="+",
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5],
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ball_track, player_boxes = load(Path(args.run))
    print(f"{len(ball_track)} ball samples, {len(player_boxes)} tracks")
    print(f"answer key: points end at frames {GROUND_TRUTH_ENDS} (+/-{TOLERANCE})\n")

    header = (
        f"{'prom':>5} {'bounce':>7} {'hits':>5} {'rally':>6} {'decided':>8} "
        f"{'found/2':>8} {'err':>6} {'spurious':>9}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for value in args.values:
        row = score(value, ball_track, player_boxes, args.fps)
        rows.append(row)
        print(
            f"{row['prominence']:>5.2f} {row['bounces']:>7} {row['hits']:>5} "
            f"{row['rallies']:>6} {row['decided']:>8} {row['truth_matched']:>8} "
            f"{str(row['mean_boundary_error']):>6} {row['spurious']:>9}"
        )

    # Most real points found first; ties broken by fewer inventions, then by
    # tighter timing. Optimising the count of bounces instead would reward a
    # threshold that shreds one rally into many.
    best = max(
        rows,
        key=lambda r: (
            r["truth_matched"],
            -r["spurious"],
            -(r["mean_boundary_error"] or 999),
        ),
    )
    print(
        f"\nbest: prominence={best['prominence']} - "
        f"{best['truth_matched']}/{len(GROUND_TRUTH_ENDS)} real points found, "
        f"{best['spurious']} spurious, "
        f"mean boundary error {best['mean_boundary_error']} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
