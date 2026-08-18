"""Measure ball-detector recall on one clip, across inference resolutions.

Why this exists
---------------
On the amateur clip the ball detector finds the ball in 98.2% of frames. On
Wimbledon broadcast footage it finds 69.3%. That gap is not cosmetic: the
trajectory layer interpolates gaps of up to 8 frames and refuses to invent a
path across anything longer, so every gap above that length cuts the ball's
flight in two. Rally segmentation then reads the two halves as two rallies, the
first hit of the second half is mistaken for a serve, that "serve" lands
nowhere near a service box, and the point is scored as a fault that never
happened.

So the number to optimise is not the detection rate. It is **how many gaps
exceed the interpolation limit**. A detector that finds 80% of frames in one
unbroken run is worth more than one that finds 90% shredded into fragments.

How it measures
---------------
One GPU pass per resolution, at a permissive confidence floor, recording the
top-k boxes for every frame. Because the per-frame ranking does not change when
the threshold is raised, every threshold at or above that floor can then be
evaluated from the same recorded pass. That turns an
O(resolutions x thresholds) sweep into O(resolutions).

Two proxies stand in for labels, which this clip does not have:

- *implausible jumps* - the share of consecutive detections that move faster
  than a tennis ball can. A serve at 200 km/h covers about 1.8 m per frame at
  30 fps, which is roughly 200 px on a 1080p broadcast wide shot. Anything
  beyond that is the detector locking onto a line marking or a shoe, so this
  rises when a lower threshold starts admitting false positives.
- *gaps over the interpolation limit* - the failure mode above, counted
  directly.

Usage
-----
    python -m training.sweep_ball --input input_videos/SAMPLE.mp4 \
        --imgsz 960 1280 1920 --out training/ball_sweep

    python -m training.sweep_ball --analyse training/ball_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tennis import video

# Confidence floor for the recorded pass. Every threshold evaluated offline
# must sit at or above this, or the recording is missing boxes it would need.
FLOOR = 0.02

# Boxes kept per frame. The detector's own pick is the highest-confidence box;
# keeping a few runners-up leaves room to ask, later, whether a
# trajectory-consistent choice beats a confidence-ranked one.
TOP_K = 5

# Frames apart that still count as consecutive when measuring step size. A
# single dropped frame should not be read as the ball teleporting.
MAX_STEP_GAP = 2

# Pixels per frame beyond which a step is not a tennis ball. See module docstring.
MAX_STEP_PX = 200.0

# The trajectory layer's interpolation limit, mirrored from
# tennis.trajectory.fill_gaps. Gaps longer than this are what split rallies.
INTERP_LIMIT = 8


def record(
    input_path: str,
    imgsz: int,
    model_path: str,
    device: str,
    limit: int | None = None,
) -> dict:
    """One GPU pass: top-k boxes per frame at the confidence floor."""
    from tennis.detect import BallDetector
    from ultralytics import YOLO

    detector = BallDetector(model_path, conf=FLOOR, device=device, imgsz=imgsz)
    model: YOLO = detector.model

    frames: list[dict] = []
    processed = 0
    started = time.time()
    for index, frame in video.frames(input_path, 0, limit):
        result = model.predict(
            frame, conf=FLOOR, imgsz=imgsz, device=device, verbose=False
        )[0]
        boxes = sorted(
            (
                {
                    "conf": float(b.conf.item()),
                    "xy": [
                        float((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                        float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                    ],
                }
                for b in result.boxes
            ),
            key=lambda d: d["conf"],
            reverse=True,
        )[:TOP_K]
        if boxes:
            frames.append({"frame": index, "boxes": boxes})
        processed += 1
        if processed % 200 == 0:
            rate = processed / (time.time() - started)
            print(f"  imgsz={imgsz}: {processed} frames ({rate:.1f} fps)", flush=True)

    elapsed = time.time() - started
    return {
        "imgsz": imgsz,
        "frames_processed": processed,
        "seconds": round(elapsed, 1),
        "fps": round(processed / elapsed, 2) if elapsed else None,
        "detections": frames,
    }


def evaluate(pass_data: dict, conf: float, selector: str = "maxconf") -> dict:
    """Score one recorded pass at one confidence threshold.

    ``maxconf`` mirrors ``BallDetector.detect``: the highest-confidence box
    that clears the threshold, one per frame. ``viterbi`` instead asks
    ``tennis.balltrack`` which sequence of boxes looks most like a ball
    flying, which is the comparison this sweep exists to make.
    """
    processed = pass_data["frames_processed"]
    picked: list[tuple[int, np.ndarray]] = []
    confidences: list[float] = []

    if selector == "viterbi":
        from tennis import balltrack

        for row in balltrack.resolve(pass_data["detections"], conf_floor=conf):
            picked.append((row["frame"], np.array(row["image"])))
            confidences.append(row["confidence"])
    else:
        for row in pass_data["detections"]:
            best = next((b for b in row["boxes"] if b["conf"] >= conf), None)
            if best is not None:
                picked.append((row["frame"], np.array(best["xy"])))
                confidences.append(best["conf"])

    detected = len(picked)
    rate = detected / processed if processed else 0.0

    # Step plausibility, over pairs close enough in time to compare.
    jumps = 0
    comparable = 0
    for (f0, p0), (f1, p1) in zip(picked, picked[1:]):
        span = f1 - f0
        if span > MAX_STEP_GAP:
            continue
        comparable += 1
        if float(np.linalg.norm(p1 - p0)) / span > MAX_STEP_PX:
            jumps += 1

    # Gaps, which is the number that actually predicts rally fragmentation.
    frames_seen = [f for f, _ in picked]
    gaps = [b - a for a, b in zip(frames_seen, frames_seen[1:])]
    over_limit = [g for g in gaps if g > INTERP_LIMIT]

    return {
        "conf": conf,
        "selector": selector,
        "detection_rate": round(rate, 4),
        "frames_detected": detected,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "implausible_jump_share": (
            round(jumps / comparable, 4) if comparable else 0.0
        ),
        "gaps_over_interp_limit": len(over_limit),
        "longest_gap_frames": max(gaps) if gaps else 0,
        "frames_lost_to_long_gaps": int(sum(over_limit)),
    }


def analyse(
    out_dir: Path, confs: list[float], selectors: list[str]
) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out_dir.glob("pass_imgsz*.json")):
        pass_data = json.loads(path.read_text(encoding="utf-8"))
        for selector in selectors:
            for conf in confs:
                row = evaluate(pass_data, conf, selector=selector)
                row["imgsz"] = pass_data["imgsz"]
                row["fps"] = pass_data["fps"]
                rows.append(row)
    return rows


def _table(rows: list[dict]) -> str:
    header = (
        f"{'imgsz':>6} {'conf':>5} {'selector':>9} {'detect':>7} {'meanC':>6} "
        f"{'jumps':>6} {'gaps>8':>7} {'longest':>8} {'lost':>6} {'fps':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda r: (r["imgsz"], r["selector"], r["conf"])):
        lines.append(
            f"{r['imgsz']:>6} {r['conf']:>5.2f} {r['selector']:>9} "
            f"{r['detection_rate']:>6.1%} "
            f"{r['mean_confidence']:>6.3f} {r['implausible_jump_share']:>5.1%} "
            f"{r['gaps_over_interp_limit']:>7} {r['longest_gap_frames']:>8} "
            f"{r['frames_lost_to_long_gaps']:>6} {str(r['fps']):>6}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sweep_ball")
    parser.add_argument("--input", help="clip to sweep")
    parser.add_argument(
        "--imgsz", type=int, nargs="+", default=[960, 1280, 1920],
        help="inference resolutions to record",
    )
    parser.add_argument(
        "--conf", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.25],
        help="confidence thresholds to evaluate offline",
    )
    parser.add_argument("--ball-model", default="models/ball_finetuned.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="training/ball_sweep")
    parser.add_argument(
        "--selector", nargs="+", default=["maxconf", "viterbi"],
        choices=["maxconf", "viterbi"],
        help="how to pick one box per frame from the recorded candidates",
    )
    parser.add_argument(
        "--analyse", metavar="DIR",
        help="skip the GPU passes and re-score recordings already in DIR",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.analyse:
        out_dir = Path(args.analyse)
    else:
        if not args.input:
            raise SystemExit("--input is required unless --analyse is given")
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        device = args.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}", flush=True)

        for imgsz in args.imgsz:
            print(f"recording imgsz={imgsz}...", flush=True)
            data = record(
                args.input, imgsz, args.ball_model, device, limit=args.limit
            )
            (out_dir / f"pass_imgsz{imgsz:04d}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
            print(
                f"  done: {data['frames_processed']} frames at {data['fps']} fps",
                flush=True,
            )

    rows = analyse(out_dir, args.conf, args.selector)
    if not rows:
        print("no recordings found")
        return 1
    print()
    print(_table(rows))
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    # The recommendation reads gaps first, then rate, because gaps are what
    # fragment rallies. A config that finds fewer frames in longer unbroken
    # runs beats a leakier one that finds more.
    best = min(
        rows,
        key=lambda r: (
            r["gaps_over_interp_limit"],
            -r["detection_rate"],
            r["implausible_jump_share"],
        ),
    )
    print(
        f"\nfewest fragmenting gaps: imgsz={best['imgsz']} conf={best['conf']} "
        f"selector={best['selector']} "
        f"({best['gaps_over_interp_limit']} gaps > {INTERP_LIMIT} frames, "
        f"{best['detection_rate']:.1%} detected, "
        f"{best['implausible_jump_share']:.1%} implausible steps)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
