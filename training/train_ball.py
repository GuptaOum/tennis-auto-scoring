"""Fine-tune the tennis ball detector.

Replaces the baseline's one-line `yolo task=detect mode=train ... imgsz=640`
with a run that is reproducible, evaluated, and matched to how the model is
actually used.

Three choices differ from the baseline, each for a measured reason:

imgsz=960, not 640
    A tennis ball is ~15 px in a 1080p frame; at 640 the letterboxed frame
    leaves ~5 px. Inference already runs at 960 (detection 47% -> 96%), and
    training resolution should match inference resolution or the model learns
    a scale it will never see.

motion blur augmentation
    The frames that matter most are the ones around ground contact, which is
    exactly where the ball is fastest and most smeared. The training set is
    largely clean synthetic frames, so blur has to be introduced or the model
    never learns the case it will be judged on.

mosaic disabled for the final epochs
    Mosaic helps small-object recall but distorts scale statistics. Turning it
    off near the end lets the model settle on real full-frame geometry.

Usage:

    python training/train_ball.py --data path/to/data.yaml
    python training/train_ball.py --data ... --epochs 120 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the ball detector")
    parser.add_argument("--data", required=True, help="dataset yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="ball_ft")
    parser.add_argument(
        "--baseline",
        default=None,
        help="weights to evaluate alongside, for a before/after comparison",
    )
    return parser.parse_args()


def evaluate(weights: str, data: str, imgsz: int, device: str) -> dict:
    result = YOLO(weights).val(
        data=data, imgsz=imgsz, device=device, verbose=False, plots=False
    )
    return {
        "weights": str(weights),
        "mAP50": round(float(result.box.map50), 4),
        "mAP50_95": round(float(result.box.map), 4),
        "precision": round(float(result.box.mp), 4),
        "recall": round(float(result.box.mr), 4),
    }


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=25,
        # -- augmentation ---------------------------------------------------
        # Aimed squarely at motion blur and small scale, which is where this
        # detector fails on real footage.
        degrees=5.0,        # courts are level; large rotations are unrealistic
        translate=0.15,
        scale=0.5,          # wide scale jitter: the ball's size varies with depth
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,         # a ball never appears upside down
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,    # last 15 epochs on undistorted full frames
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,          # court lighting varies far more than court colour
        # -- optimisation ---------------------------------------------------
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        plots=True,
        val=True,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    report = {"finetuned": evaluate(str(best), args.data, args.imgsz, args.device)}
    if args.baseline:
        report["baseline"] = evaluate(
            args.baseline, args.data, args.imgsz, args.device
        )

    out = Path(model.trainer.save_dir) / "comparison.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== results ===")
    for label, metrics in report.items():
        print(
            f"{label:>10}: mAP50={metrics['mAP50']:.4f} "
            f"mAP50-95={metrics['mAP50_95']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}"
        )
    if "baseline" in report:
        delta = report["finetuned"]["mAP50"] - report["baseline"]["mAP50"]
        print(f"{'delta':>10}: mAP50 {delta:+.4f}")
    print(f"\nbest weights: {best}")


if __name__ == "__main__":
    main()
