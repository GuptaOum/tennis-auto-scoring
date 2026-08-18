"""Proof of concept: track the ball, the players and the court in one pass.

Three trackers, three techniques, chosen because the targets differ:

* **Ball - TrackNet.** A tennis ball is ~10 px, motion-blurred into a streak,
  and often indistinguishable from a line marking in a single frame. TrackNet
  takes *three consecutive frames* and outputs a heatmap, so it can use motion
  to find something a per-frame detector cannot see at all.
* **Players - YOLO.** Large, high-contrast, slow-moving. A per-frame detector is
  the right tool and needs no temporal help.
* **Court - detect then follow.** Detected periodically and snapped onto the
  painted lines, then carried between detections by sparse optical flow, so it
  stays locked while the camera moves. See ``tennis/court_track.py``.

Output is an annotated video and a JSON of raw tracks. No scoring, no analysis,
no report - this exists to show the three tracking techniques working.

    python -m tennis.track --input input_videos/clip.mp4 --out output/tracked
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tennis.court import COURT_LENGTH, COURT_MODEL, DOUBLES_WIDTH, NET_Y
from tennis.court_track import CourtTracker
from tennis.detect import CourtDetector, PlayerDetector

# Saturated, dark-edged colours. Tennis footage is bright - grass, blue hard
# court, white clothing, sunlit crowd - so light markers wash out. These are
# fully saturated primaries at full depth, each drawn over a thick black
# outline, which is what actually carries contrast on a bright background.
BALL = (255, 0, 255)        # magenta - nothing on a tennis court is magenta
PLAYER = (255, 90, 0)       # deep blue
CORNER = (0, 0, 220)        # dark red
OUTLINE = (0, 0, 0)
# Furthest the ball can plausibly move between consecutive frames, in pixels at
# 1080p. Beyond this a "detection" is a false positive, not motion.
MAX_STEP_PX = 120.0

# After this many rejections in a row, believe the detector again. Without it a
# single bad lock would blind the tracker for the rest of the clip: the ball
# really does leave and re-enter, and a gate with no escape never re-acquires.
REACQUIRE_AFTER = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="tennis.track")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="output/tracked")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N frames")
    parser.add_argument("--ball-model", default="models/tracknet_tennis.pt")
    parser.add_argument(
        "--ball-detector", choices=("yolo", "tracknet"), default="tracknet",
        help="yolo uses the fine-tuned ball detector trained here "
             "(mAP50 0.90); tracknet uses the published TrackNet weights",
    )
    parser.add_argument(
        "--ball-imgsz", type=int, default=1600,
        help="inference resolution for the YOLO ball detector. The optimum "
             "tracks apparent ball size, so a wide broadcast shot needs more "
             "than a tight amateur clip",
    )
    parser.add_argument("--ball-conf", type=float, default=0.02)
    parser.add_argument("--player-model", default="models/yolov8x.pt")
    parser.add_argument("--court-model", default="models/keypoints_model.pth")
    parser.add_argument("--redetect-every", type=int, default=30)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def draw_ball(canvas, point):
    """Marker only, no trail.

    The trail was removed on purpose. A per-frame detector's occasional wrong
    lock draws a long line across the court, and a line is far more visually
    dominant than the stray dot that caused it - so the drawing exaggerated the
    error rather than reporting it. The rejected detections are still counted
    and reported as ``implausible_rejected``, which is the honest place for
    that information.

    Ring plus crosshair rather than a filled dot, so the ball stays visible
    inside the marker and a viewer can see the track really is on it.
    """
    if point is None:
        return
    x, y = int(point[0]), int(point[1])
    cv2.circle(canvas, (x, y), 13, OUTLINE, 5, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 13, BALL, 3, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = (x + dx * 18, y + dy * 18)
        b = (x + dx * 7, y + dy * 7)
        cv2.line(canvas, a, b, OUTLINE, 5, cv2.LINE_AA)
        cv2.line(canvas, a, b, BALL, 3, cv2.LINE_AA)


def describe_players(detections, calibration, previous, fps):
    """The two players, with court position and speed.

    Selection is delegated to ``tennis.players.select`` rather than reimplemented
    here. It already encodes the two facts that matter - a player is on the
    court, and there is exactly one per side of the net - and it re-identifies
    by side rather than by tracker id. Writing a fresh rule here ("nearest the
    net on each side") picked a ball kid crouched at the net and a bystander by
    the umpire's chair while the actual players went unmarked.
    """
    from tennis.players import FAR_PLAYER, select

    chosen = []
    for det in select(detections, calibration):
        court_xy = None
        if calibration is not None:
            # Feet, not centre: only the feet are on the plane the homography maps.
            point = calibration.to_court(np.asarray(det.feet, dtype=float))
            court_xy = [round(float(point[0]), 2), round(float(point[1]), 2)]
        name = "P2 far" if det.track_id == FAR_PLAYER else "P1 near"
        chosen.append({"bbox": list(det.bbox), "court": court_xy, "name": name})

    for player in chosen:
        player["speed_kmh"] = None
        was = previous.get(player["name"])
        if was is not None and player["court"] is not None:
            moved = float(np.linalg.norm(np.array(player["court"]) - np.array(was)))
            # Per-frame speed is jittery; cap it so one bad frame cannot print a
            # sprinter's number beside a stationary player.
            player["speed_kmh"] = min(moved * fps * 3.6, 40.0)
    return chosen


def _label(canvas, text, origin, colour, scale=0.5):
    """Text with a dark outline, so it survives grass, clay and a bright crowd."""
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 1, cv2.LINE_AA)


def draw_players(canvas, players):
    """Corner brackets, not full boxes - the feet stay visible.

    Position is measured at the feet, which is the only part of a player that
    is actually on the court plane the homography maps.
    """
    for player in players:
        x1, y1, x2, y2 = (int(v) for v in player["bbox"])
        run = max(12, (x2 - x1) // 4)
        for cx, cy, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            for colour, weight in ((OUTLINE, 6), (PLAYER, 3)):
                cv2.line(canvas, (cx, cy), (cx + sx * run, cy), colour,
                         weight, cv2.LINE_AA)
                cv2.line(canvas, (cx, cy), (cx, cy + sy * run), colour,
                         weight, cv2.LINE_AA)

        text = player["name"]
        if player["court"] is not None:
            text += "  %.1f, %.1f m" % tuple(player["court"])
        if player["speed_kmh"] is not None:
            text += "  %.0f km/h" % player["speed_kmh"]
        _label(canvas, text, (x1, max(y1 - 10, 16)), PLAYER, 0.55)


def draw_court(canvas, calibration):
    """All 14 court landmarks as dots, tracked frame by frame.

    Dots rather than lines: the court already has lines painted on it, and one
    sitting off its marking is visible evidence the track has drifted.
    """
    if calibration is None:
        _label(canvas, "COURT  searching", (18, 32), (180, 180, 180), 0.7)
        return
    height, width = canvas.shape[:2]
    points = calibration.to_image(COURT_MODEL)
    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        if not (-50 <= x <= width + 50 and -50 <= y <= height + 50):
            continue
        # The four doubles corners are the anchor the tracker actually carries,
        # so they are drawn larger than the ten it derives from them.
        radius = 6 if index < 4 else 4
        cv2.circle(canvas, (x, y), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, CORNER, -1, cv2.LINE_AA)


def draw_minimap(canvas, players, ball_court, box=None):
    """The court to scale in metres, with whoever is currently on it."""
    height, width = canvas.shape[:2]
    # Sized from the frame, not fixed: the same panel that reads well at 1080p
    # swallows a 360p frame.
    box = box or int(height * 0.30)
    pad = max(8, height // 60)
    span = COURT_LENGTH + 2.0
    scale = box / span
    map_w = int((DOUBLES_WIDTH + 2.0) * scale)
    x0, y0 = width - map_w - pad, pad

    panel = canvas[y0:y0 + box, x0:x0 + map_w]
    if panel.size:
        # Opaque, not translucent: a see-through panel over a bright crowd is
        # unreadable, which was a real defect in the previous overlay.
        panel[:] = (32, 32, 32)

    def to_px(x_m, y_m):
        return (int(x0 + (x_m + 1.0) * scale), int(y0 + (y_m + 1.0) * scale))

    cv2.rectangle(canvas, to_px(0, 0), to_px(DOUBLES_WIDTH, COURT_LENGTH),
                  (210, 210, 210), 1, cv2.LINE_AA)
    cv2.line(canvas, to_px(0, COURT_LENGTH / 2),
             to_px(DOUBLES_WIDTH, COURT_LENGTH / 2), (210, 210, 210), 2,
             cv2.LINE_AA)

    for player in players:
        if player["court"] is None:
            continue
        cv2.circle(canvas, to_px(*player["court"]), 5, PLAYER, -1, cv2.LINE_AA)
    if ball_court is not None:
        cv2.circle(canvas, to_px(*ball_court), 4, BALL, -1, cv2.LINE_AA)

    _label(canvas, "COURT VIEW (metres)", (x0, y0 + box + max(12, height // 70)),
           (200, 200, 200), max(0.35, height / 2400))


def draw_hud(canvas, index, total, found_ball, court_stats, tracking,
             detector_name="ball"):
    """What each tracker is doing right now, stated plainly."""
    lines = [
        ("BALL   %s" % detector_name, BALL if tracking else (140, 140, 140)),
        ("       %s" % ("locked" if tracking else "not visible"),
         BALL if tracking else (140, 140, 140)),
        ("COURT  %d detections, %d tracked, %d lost"
         % (court_stats["detections"], court_stats["tracked"],
            court_stats["lost"]), CORNER),
        ("FRAME  %d / %d   ball found %.0f%%"
         % (index, total, 100.0 * found_ball / max(index, 1)), (220, 220, 220)),
    ]
    height, width = canvas.shape[:2]
    scale = max(0.38, height / 2400)
    step = max(14, int(height / 48))
    pad = max(8, height // 90)
    panel = canvas[pad:pad + step * len(lines) + pad,
                   pad:pad + int(width * 0.30)]
    if panel.size:
        panel[:] = (32, 32, 32)
    for i, (text, colour) in enumerate(lines):
        _label(canvas, text, (pad * 2, pad + step * (i + 1)), colour, scale)


def open_writer(out_dir: Path, fps: float, width: int, height: int):
    """A VideoWriter that is actually writing, or a hard failure.

    ``cv2.VideoWriter`` does not raise when a codec is missing - it returns an
    object whose ``isOpened()`` is False and then silently discards every frame,
    leaving a 44-byte file. That is exactly what happened here, so the return is
    checked and several codecs are tried before giving up. Which one works
    depends on the OpenCV build, and the build differs between this machine and
    the GPU box.
    """
    attempts = [
        ("avc1", ".mp4"),   # H.264, plays everywhere
        ("mp4v", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),   # large, but present in essentially every build
    ]
    for fourcc, suffix in attempts:
        path = out_dir / ("tracked" + suffix)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if writer.isOpened():
            print("video codec       : %s -> %s" % (fourcc, path.name))
            return writer, path
        writer.release()
    raise SystemExit(
        "no working video codec found (tried %s). Re-run with --no-video."
        % ", ".join(f for f, _ in attempts)
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from tennis.tracknet import FRAME_STACK, TrackNetDetector

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("cannot open " + args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    use_tracknet = args.ball_detector == "tracknet"
    if use_tracknet:
        ball_model = TrackNetDetector(args.ball_model, device=device)
    else:
        from tennis.detect import BallDetector
        ball_model = BallDetector(args.ball_model, conf=args.ball_conf,
                                  device=device, imgsz=args.ball_imgsz)
    print("ball detector      : %s (%s)" % (args.ball_detector, args.ball_model))
    people_model = PlayerDetector(args.player_model, device=device)
    court = CourtTracker(CourtDetector(args.court_model, device=device),
                         redetect_every=args.redetect_every)

    writer = None
    video_path = None
    if not args.no_video:
        writer, video_path = open_writer(out_dir, fps, width, height)

    window = []
    previous = {}
    last_ball = None
    rejected = 0
    implausible = 0
    tracks = {"ball": [], "players": [], "court": []}
    index = 0
    found_ball = 0
    started = time.time()

    print("%s: %d frames, %dx%d @ %.0f fps, %s"
          % (args.input, total, width, height, fps, device))
    if device == "cpu":
        # Measured: 13.9 s/frame with yolov8x on CPU. Say so up front - a run
        # that looks hung for ten minutes is worse than a slow run you expected.
        print("  warning: CPU. yolov8x runs ~14 s/frame here; a minute of "
              "video is hours. Use a GPU, or --player-model yolov8n.pt")

    while True:
        ok, frame = cap.read()
        if not ok or (args.limit and index >= args.limit):
            break

        window.append((index, frame))
        if len(window) > FRAME_STACK:
            window.pop(0)

        # TrackNet needs three consecutive frames; the clip's first two have no
        # predecessors and so simply carry no ball.
        # Both detectors go through the same motion gate, or the comparison
        # between them is meaningless: an ungated detector reports every false
        # lock as a hit and scores higher for being wrong more often.
        candidate = None
        candidate_conf = None
        if use_tracknet:
            if len(window) == FRAME_STACK:
                found = ball_model.detect_sequence(window)
                if found:
                    candidate = np.asarray(found[-1].xy, dtype=float)
                    candidate_conf = found[-1].confidence
        else:
            hit = ball_model.detect(frame)
            if hit is not None:
                candidate = np.asarray(hit.centre, dtype=float)
                candidate_conf = hit.confidence

        point = None
        confidence = None
        if candidate is not None:
            if last_ball is None:
                point, confidence, rejected = candidate, candidate_conf, 0
            else:
                # Allowance widens while the ball is missing - it keeps moving.
                reach = MAX_STEP_PX * (1 + rejected)
                if float(np.hypot(*(candidate - last_ball))) <= reach:
                    point, confidence, rejected = candidate, candidate_conf, 0
                else:
                    rejected += 1
                    implausible += 1
                    if rejected >= REACQUIRE_AFTER:
                        point, confidence, rejected = candidate, candidate_conf, 0
            if point is not None:
                last_ball = point

        if point is not None:
            tracks["ball"].append({
                "frame": index, "x": float(point[0]), "y": float(point[1]),
                "confidence": confidence,
            })
            found_ball += 1

        people = people_model.detect(frame)
        calibration = court.update(frame, index)
        players = describe_players(people, calibration, previous, fps)
        previous = {p["name"]: p["court"] for p in players
                    if p["court"] is not None}

        ball_court = None
        if point is not None and calibration is not None:
            ball_court = calibration.to_court(np.asarray(point, dtype=float))

        tracks["players"].append({
            "frame": index,
            "boxes": [[float(v) for v in d.bbox] for d in people],
            "court": [p["court"] for p in players],
        })
        tracks["court"].append({
            "frame": index,
            "corners": (calibration.to_image(COURT_MODEL[:4]).tolist()
                        if calibration is not None else None),
        })

        if writer is not None:
            canvas = frame.copy()
            draw_court(canvas, calibration)
            draw_players(canvas, players)
            draw_ball(canvas, point)
            draw_minimap(canvas, players, ball_court)
            draw_hud(canvas, index + 1, total, found_ball, court.stats,
                     point is not None, args.ball_detector)
            writer.write(canvas)

        index += 1
        if index % 25 == 0:
            rate = index / max(time.time() - started, 1e-6)
            print("  %d frames (%.1f fps)" % (index, rate), flush=True)

    cap.release()
    if writer is not None:
        writer.release()

    summary = {
        "input": args.input,
        "frames": index,
        "ball_detector": args.ball_detector,
        "ball_detection_rate": round(found_ball / index, 3) if index else 0.0,
        # Detections the motion gate threw out. The honest companion to the
        # rate: a detector can score high by being confidently wrong often.
        "implausible_rejected": implausible,
        "court": court.stats,
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "tracks.json").write_text(json.dumps(tracks), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- tracking ---")
    print("frames            : %d" % index)
    print("implausible reject: %d" % implausible)
    print("ball found        : %d (%.1f%%)"
          % (found_ball, 100.0 * summary["ball_detection_rate"]))
    print("court detections  : %d, tracked %d, lost %d"
          % (court.stats["detections"], court.stats["tracked"],
             court.stats["lost"]))
    if writer is not None:
        print("video             : %s (%.1f MB)"
              % (video_path, video_path.stat().st_size / 1e6))
    print("tracks            : %s" % (out_dir / "tracks.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
