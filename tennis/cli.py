"""Command-line entry point.

    python -m tennis.cli --input input_videos/match.mp4 --out output/

Runs detection over the video, calibrates the court, and writes an annotated
video plus a JSON report. Bounce detection, rally segmentation and point
attribution are not wired in yet - the report says so explicitly rather than
emitting a placeholder score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tennis import analysis as analysis_module
from tennis import rally as rally_module
from tennis import video
from tennis.bounce import EventType, detect_events
from tennis.court import COURT_LINES, calibrate
from tennis.detect import BallDetector, CourtDetector, PlayerDetector
from tennis.trajectory import from_detections

# BGR
GREEN = (0, 220, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


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
    parser.add_argument("--player-model", default="yolov8x.pt")
    parser.add_argument("--ball-model", default="models/yolo5_last.pt")
    parser.add_argument("--court-model", default="models/keypoints_model.pth")
    parser.add_argument(
        "--no-video", action="store_true", help="write the JSON report only"
    )
    return parser.parse_args(argv)


def draw_overlay(
    frame: np.ndarray,
    players: list,
    ball,
    keypoints: np.ndarray | None,
    calib,
    frame_index: int,
) -> np.ndarray:
    canvas = frame.copy()

    if keypoints is not None:
        for index, (x, y) in enumerate(keypoints):
            cv2.circle(canvas, (int(x), int(y)), 4, RED, -1)
            cv2.putText(canvas, str(index), (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1)
        for a, b in COURT_LINES:
            cv2.line(canvas, tuple(keypoints[a].astype(int)),
                     tuple(keypoints[b].astype(int)), WHITE, 1)

    for player in players:
        x1, y1, x2, y2 = (int(v) for v in player.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
        label = f"P{player.track_id}" if player.track_id else "P?"
        if calib is not None:
            court_pt = calib.to_court(player.feet)
            label += f"  ({court_pt[0]:.1f}, {court_pt[1]:.1f}) m"
        cv2.putText(canvas, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

    if ball is not None:
        x1, y1, x2, y2 = (int(v) for v in ball.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), YELLOW, 2)
        cv2.putText(canvas, f"ball {ball.confidence:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    status = f"frame {frame_index}"
    if calib is not None:
        flag = "OK" if calib.is_reliable else "UNRELIABLE"
        status += f" | court {flag} ({calib.reprojection_error:.1f} px)"
    cv2.putText(canvas, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    return canvas


def run(args: argparse.Namespace) -> dict:
    info = video.probe(args.input)
    print(f"input: {info}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    print("loading models...", flush=True)
    players_model = PlayerDetector(args.player_model, device=device)
    ball_model = BallDetector(args.ball_model, device=device)
    court_model = CourtDetector(args.court_model, device=device)

    writer = None
    if not args.no_video:
        writer = video.VideoWriter(
            out_dir / "annotated.mp4", info.fps, info.width, info.height
        )

    calib = None
    keypoints = None
    calibrations: list[dict] = []
    ball_track: list[dict] = []
    player_track: list[dict] = []
    player_boxes: dict[int, dict[int, tuple]] = {}
    processed = 0
    started = time.time()

    try:
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

            if writer is not None:
                writer.write(
                    draw_overlay(frame, detected_players, ball, keypoints, calib, index)
                )

            processed += 1
            if processed % 25 == 0:
                rate = processed / (time.time() - started)
                print(f"  {processed} frames ({rate:.1f} fps)", flush=True)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - started
    reliable = [c for c in calibrations if c.get("reliable")]

    # Everything above is per-frame detection. Everything below turns that into
    # a score: trajectory -> bounces and hits -> rallies -> points.
    trajectory = from_detections(ball_track)
    events = detect_events(trajectory, player_boxes=player_boxes, fps=info.fps)
    match, rallies = rally_module.score_match(events, fps=info.fps)
    rally_summary = rally_module.summarise(rallies, fps=info.fps)
    analysis = analysis_module.summarise(player_track, ball_track, info.fps)
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
        "analysis": analysis,
        "score": match.summary(),
    }

    for player in analysis["players"]:
        player["coverage"] = analysis_module.render_coverage(
            analysis_module.court_coverage(player_track, player["track_id"])
        )

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
    if rallies["points_decided"]:
        print(f"score              : {report['score']['scoreline']}")
    else:
        print("score              : no completed point found in this clip")
    print(f"\nreport             : {Path(args.out) / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
