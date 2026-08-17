"""Command-line entry point.

    python -m tennis.cli --input input_videos/match.mp4 --out output/

Two passes over the video. The first detects players, ball and court and turns
that into events, rallies and a score. The second re-decodes the video and
draws the annotated overlay, which can only happen once the first pass is done:
a bounce is identifiable only from the frames after it, so the score at a given
frame is not knowable while that frame is being detected.

Outputs land in --out: annotated.mp4, report.html, report.json, ball_track.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tennis import analysis as analysis_module
from tennis import placement as placement_module
from tennis import overlay
from tennis import rally as rally_module
from tennis import report as report_module
from tennis import serve as serve_module
from tennis import video
from tennis.bounce import EventType, detect_events
from tennis.court import calibrate
from tennis.detect import BallDetector, CourtDetector, PlayerDetector
from tennis.trajectory import from_detections

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tennis", description="Automatic tennis match analysis and scoring"
    )
    parser.add_argument("--input", required=True, help="path to the match video")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after N frames (for quick tests)"
    )
    parser.add_argument("--start", type=int, default=0, help="first frame to process")
    parser.add_argument(
        "--recalibrate-every",
        type=int,
        default=30,
        help="re-estimate court keypoints every N frames (0 = once, at the start)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto (default: use the GPU when one is present)",
    )
    parser.add_argument(
        "--ball-imgsz",
        type=int,
        default=960,
        help="inference resolution for the ball detector (default 960; 640 "
             "halves the detection rate on 1080p footage)",
    )
    parser.add_argument("--player-model", default="yolov8x.pt")
    parser.add_argument("--ball-model", default="models/ball_finetuned.pt")
    parser.add_argument("--court-model", default="models/keypoints_model.pth")
    parser.add_argument(
        "--no-video", action="store_true", help="write the JSON report only"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="skip the HTML report (report.html is written by default)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    info = video.probe(args.input)
    print(f"input: {info}", flush=True)
    codec = video.codec_of(args.input)
    if codec:
        print(f"codec: {codec}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    print("loading models...", flush=True)
    players_model = PlayerDetector(args.player_model, device=device)
    ball_model = BallDetector(args.ball_model, device=device,
                              imgsz=args.ball_imgsz)
    court_model = CourtDetector(args.court_model, device=device)

    # No writer during detection. The annotated video is rendered in a second
    # pass, because the things worth drawing - the score, the reason a rally
    # ended, whether a bounce was in - cannot be known while the frame is being
    # detected. A bounce is a local maximum in projected court y, identifiable
    # only once the frames after it exist. See tennis/overlay.py.

    calib = None
    keypoints = None
    calibrations: list[dict] = []
    ball_track: list[dict] = []
    player_track: list[dict] = []
    player_boxes: dict[int, dict[int, tuple]] = {}
    player_boxes_court: dict[int, dict[int, np.ndarray]] = {}
    processed = 0
    started = time.time()

    for index, frame in video.frames(args.input, args.start, args.limit):
        recalibrate = args.recalibrate_every and (
            processed % args.recalibrate_every == 0
        )
        if keypoints is None or recalibrate:
            keypoints = court_model.detect(frame)
            try:
                candidate = calibrate(keypoints, frame_index=index)
                # Keep the previous calibration if this frame's is worse -
                # a player crossing a line can briefly wreck the keypoints,
                # and a fixed camera means the old matrix is still valid.
                if calib is None or candidate.is_reliable:
                    calib = candidate
                calibrations.append(
                    {
                        "frame": index,
                        "reprojection_error_px": round(
                            candidate.reprojection_error, 2
                        ),
                        "inliers": candidate.inlier_count,
                        "reliable": candidate.is_reliable,
                    }
                )
            except ValueError as exc:
                calibrations.append({"frame": index, "error": str(exc)})

        detected_players = players_model.detect(frame)
        ball = ball_model.detect(frame)

        if ball is not None and calib is not None:
            court_pt = calib.to_court(ball.centre)
            ball_track.append(
                {
                    "frame": index,
                    "image": [float(v) for v in ball.centre],
                    "court": [float(court_pt[0]), float(court_pt[1])],
                    "confidence": round(ball.confidence, 3),
                }
            )

        if calib is not None:
            for player in detected_players:
                court_pt = calib.to_court(player.feet)
                player_track.append(
                    {
                        "frame": index,
                        "track_id": player.track_id,
                        "x_m": round(float(court_pt[0]), 3),
                        "y_m": round(float(court_pt[1]), 3),
                    }
                )
                if player.track_id is not None:
                    # Image-space box, not the court position: hit detection
                    # needs proximity, and an airborne ball's court
                    # coordinate is projected metres from where it truly is.
                    player_boxes.setdefault(player.track_id, {})[index] = (
                        player.bbox
                    )
                    # Court-space feet, for shot origin. Valid because a
                    # player stands on the plane the homography maps.
                    player_boxes_court.setdefault(player.track_id, {})[
                        index
                    ] = court_pt

        processed += 1
        if processed % 25 == 0:
            rate = processed / (time.time() - started)
            print(f"  {processed} frames ({rate:.1f} fps)", flush=True)

    elapsed = time.time() - started
    reliable = [c for c in calibrations if c.get("reliable")]

    # Everything above is per-frame detection. Everything below turns that into
    # a score: trajectory -> bounces and hits -> rallies -> points.
    trajectory = from_detections(ball_track)
    events = detect_events(trajectory, player_boxes=player_boxes, fps=info.fps)
    # Serves must be judged before scoring: a first-serve fault ends no point,
    # and without that the point is awarded to the wrong player.
    provisional = rally_module.segment(events, fps=info.fps)
    serves, faults, doubles = serve_module.analyse_serves(
        provisional, player_boxes_court
    )
    match, rallies = rally_module.score_match(
        events,
        fps=info.fps,
        fault_indices=set(faults),
        double_faults=dict(doubles),
    )
    rally_summary = rally_module.summarise(rallies, fps=info.fps)
    serve_summary = serve_module.summarise(serves, doubles)
    analysis = analysis_module.summarise(player_track, ball_track, info.fps)
    placement = placement_module.summarise(rallies, player_boxes_court)
    report = {
        "input": {
            "path": str(info.path),
            "fps": info.fps,
            "resolution": f"{info.width}x{info.height}",
            "frames_total": info.frame_count,
            "frames_processed": processed,
        },
        "performance": {
            "device": device,
            "seconds": round(elapsed, 1),
            "fps": round(processed / elapsed, 2) if elapsed else None,
        },
        "court_calibration": {
            "attempts": len(calibrations),
            "reliable": len(reliable),
            "median_reprojection_error_px": (
                round(
                    float(
                        np.median([c["reprojection_error_px"] for c in calibrations
                                   if "reprojection_error_px" in c])
                    ),
                    2,
                )
                if any("reprojection_error_px" in c for c in calibrations)
                else None
            ),
            "samples": calibrations[:10],
        },
        "detection": {
            "ball_frames": len(ball_track),
            "ball_detection_rate": (
                round(len(ball_track) / processed, 3) if processed else 0.0
            ),
            "player_observations": len(player_track),
        },
        "events": {
            "bounces": sum(1 for e in events if e.type is EventType.BOUNCE),
            "hits": sum(1 for e in events if e.type is EventType.HIT),
        },
        "rallies": rally_summary,
        "serves": serve_summary,
        "analysis": analysis,
        "placement": placement,
        "score": match.summary(),
    }

    for player in analysis["players"]:
        grid = analysis_module.court_coverage(player_track, player["track_id"])
        player["coverage"] = analysis_module.render_coverage(grid)
        # Kept raw as well as rendered: the HTML report draws the grid over a
        # court plan, and shading characters cannot be un-rendered back into
        # counts. Costs 36 integers per player.
        player["coverage_grid"] = grid

    report["points"] = [
        {
            "winner": r.winner,
            "reason": r.reason,
            "confidence": r.confidence,
            "start_frame": r.start_frame,
            "end_frame": r.end_frame,
            "shots": r.shot_count,
        }
        for r in rallies
    ]

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "ball_track.json").write_text(
        json.dumps(ball_track, indent=2), encoding="utf-8"
    )
    if not args.no_html:
        report_module.write(report, out_dir / "report.html")

    if not args.no_video:
        print("rendering annotated video...", flush=True)
        render_started = time.time()
        renderer = overlay.Renderer(
            width=info.width,
            height=info.height,
            fps=info.fps,
            ball_track=ball_track,
            player_boxes=player_boxes,
            player_court=player_boxes_court,
            events=events,
            rallies=rallies,
            match=match,
            calibration=calib,
            source_name=Path(args.input).name,
        )
        frames_written = overlay.render_video(
            args.input,
            out_dir / "annotated.mp4",
            renderer,
            start=args.start,
            limit=args.limit,
        )
        report["performance"]["render_seconds"] = round(
            time.time() - render_started, 1
        )
        report["performance"]["frames_rendered"] = frames_written
        # Rewritten so the render timing lands in the JSON too.
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    court = report["court_calibration"]
    detection = report["detection"]
    print("\n--- summary ---")
    print(f"frames processed   : {report['input']['frames_processed']}")
    print(f"speed              : {report['performance']['fps']} fps")
    print(
        f"court calibration  : {court['reliable']}/{court['attempts']} reliable, "
        f"median error {court['median_reprojection_error_px']} px"
    )
    print(f"ball detected in   : {detection['ball_detection_rate']:.1%} of frames")
    events = report["events"]
    rallies = report["rallies"]
    print(f"events             : {events['bounces']} bounces, {events['hits']} hits")

    print("\n--- players ---")
    for player in report["analysis"]["players"]:
        print(
            f"player {player['track_id']} ({player['side']} side): "
            f"{player['distance_covered_m']} m covered, "
            f"avg {player['average_speed_kmh']} km/h, "
            f"top {player['top_speed_kmh']} km/h, "
            f"{player['net_approaches']} net approaches"
        )
        for line in player["coverage"]:
            print(f"    {line}")

    placement = report["placement"]
    if placement["in_bounds"]:
        print("\n--- shot placement ---")
        overall = placement["overall"]
        print(
            f"{placement['in_bounds']} landings in, {placement['out_of_bounds']} out"
            f"  |  mean depth {overall['mean_depth_m']} m from net"
            f"  |  {overall['deep_share']:.0%} deep"
        )
        for label, grid in placement["grids"].items():
            if sum(map(sum, grid)):
                print(f"  {label} half (net at top):")
                for line in placement_module.render_grid(grid):
                    print(f"    {line}")
        for track_id, stats in placement["by_player"].items():
            if not stats.get("landings"):
                continue
            bands = stats["depth_bands"]
            directions = stats.get("directions", {})
            print(
                f"  player {track_id}: {stats['landings']} shots  "
                f"deep {bands['deep']} / mid {bands['mid']} / short {bands['short']}"
                + (f"  |  {', '.join(f'{k} {v}' for k, v in directions.items())}"
                   if directions else "")
            )

    serve_stats = report["serves"]
    if serve_stats["serves_detected"]:
        print("\n--- serves ---")
        first_pct = serve_stats["first_serve_percentage"]
        print(
            f"{serve_stats['serves_detected']} detected  |  "
            f"{serve_stats['in']} in, {serve_stats['faults']} faults, "
            f"{serve_stats['double_faults']} double faults"
            + (f"  |  first serve {first_pct:.0%}" if first_pct is not None else "")
        )
        if serve_stats["boxes"]:
            print("  placement: " + ", ".join(
                f"{box} {count}" for box, count in serve_stats["boxes"].items()))

    print("\n--- scoring ---")
    print(
        f"rallies            : {rallies['rallies_found']} found, "
        f"{rallies['points_decided']} scored, "
        f"{rallies['points_undecided']} undecided"
    )
    for point in report["points"]:
        if point["winner"] is not None:
            print(
                f"   f{point['start_frame']}-{point['end_frame']}: "
                f"player {point['winner']} ({point['reason']}, "
                f"confidence {point['confidence']})"
            )
    score = report["score"]
    if rallies["points_decided"]:
        print()
        print("   +------------------------------------------+")
        print(f"   |  SCORE   {score['scoreline']:<32}|")
        print(f"   |  points  {score['points_played']:<3} played"
              f"{'':<24}|")
        if score["low_confidence_points"]:
            print(f"   |  {score['low_confidence_points']} point(s) flagged low confidence"
                  f"{'':<9}|")
        print("   +------------------------------------------+")
    else:
        print("score              : no completed point found in this clip")
    print(f"\nreport             : {Path(args.out) / 'report.json'}")
    if not args.no_html:
        print(f"html report        : {Path(args.out) / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
