"""Train the learned bounce detector on the TrackNet tennis dataset.

The dataset
-----------
81 broadcast clips from 10 matches, 1280x720 at 30 fps, 19,835 labelled frames.
Each clip directory holds its frames and a ``Label.csv`` whose columns are
``file name, visibility, x, y, status``. ``status`` is the trajectory pattern,
and the value marking ground contact is what makes this dataset usable here:
it supplies the bounce labels the project has never had.

Why train at all
----------------
``training/calibrate_bounce.py`` swept the geometric prominence threshold over
its whole useful range against ground truth read off the broadcast scoreboard
and **no value found both points** in the test clip. A parameter that cannot be
set correctly at any value is the wrong model, so the decision moves to a
classifier over the trajectory features in ``tennis/bounce_learned.py``.

Two things this deliberately does not do
----------------------------------------
1. **It does not split frames randomly.** Consecutive frames inside one rally
   are near-duplicates of each other; a random split puts a frame's own
   neighbours in the training set and reports an accuracy that evaporates on
   real video. Splitting whole clips is the only honest option here.
2. **It does not build its own trajectory code.** Features come from
   ``trajectory.from_detections`` and ``bounce_learned.propose``, the same
   functions inference uses. Training on a parallel implementation is how a
   model comes to depend on a smoothing window that production does not have.

Usage
-----
    python -m training.train_bounce --data training/tracknet_data \\
        --out models/bounce_model.joblib
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis import bounce_learned
from tennis.court import calibrate
from tennis.trajectory import from_detections

# Values of the label file's status column that mean the ball touched the court.
# The dataset encodes flight as 1 and contact as 2; 0 means the ball is not
# visible. Kept as a set so an unexpected encoding fails loudly at load rather
# than silently training on nothing.
# Verified against the real files: 0 = ball in flight, 1 = struck by a player,
# 2 = bounced on the court. Every visible frame is usable trajectory; only 2 is
# a ground contact. Treating 0 as unusable discards 93% of the dataset.
BOUNCE_STATUS = {2}
FLIGHT_STATUS = {0, 1, 2}

# Fraction of clips held out. Whole clips, never frames - see module docstring.
TEST_SHARE = 0.25


@dataclass
class Clip:
    name: str
    detections: list[dict]      # frame, image, court, confidence
    bounce_frames: set[int]


def _read_label_csv(path: Path) -> list[tuple[int, int, float, float]]:
    """``Label.csv`` -> ``(frame_index, status, x, y)`` rows."""
    rows: list[tuple[int, int, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            keys = {k.strip().lower(): (v or "").strip() for k, v in record.items()}
            name = keys.get("file name") or keys.get("filename") or ""
            digits = re.findall(r"\d+", name)
            if not digits:
                continue
            status = keys.get("status") or keys.get("trajectory pattern") or "0"
            visibility = keys.get("visibility") or keys.get("visibility class") or "0"
            try:
                # The real header is ``x-coordinate``/``y-coordinate``. Reading
                # plain ``x``/``y`` drops every row and the run reports zero
                # candidates - it fails silently, so keep both spellings.
                x = float(keys.get("x-coordinate") or keys.get("x") or "nan")
                y = float(keys.get("y-coordinate") or keys.get("y") or "nan")
            except ValueError:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if int(float(visibility or 0)) == 0:
                continue
            rows.append((int(digits[-1]), int(float(status or 0)), x, y))
    return rows


def load_clip(clip_dir: Path, court_model, device: str) -> Clip | None:
    """One clip directory -> detections in court metres, plus bounce frames.

    Returns ``None`` when the court cannot be calibrated. A clip whose
    homography is unreliable would contribute features measured in the wrong
    units, which is worse than contributing nothing.
    """
    label_path = next(
        (p for p in clip_dir.iterdir() if p.name.lower() == "label.csv"), None
    )
    if label_path is None:
        return None

    rows = _read_label_csv(label_path)
    if len(rows) < 20:
        return None

    calibration = _calibrate_clip(clip_dir, court_model, device)
    if calibration is None:
        return None

    detections: list[dict] = []
    bounce_frames: set[int] = set()
    for frame_index, status, x, y in rows:
        if status not in FLIGHT_STATUS:
            continue
        court = calibration.to_court(np.array([x, y], dtype=float))
        detections.append(
            {
                "frame": frame_index,
                "image": [x, y],
                "court": [float(court[0]), float(court[1])],
                "confidence": 1.0,
            }
        )
        if status in BOUNCE_STATUS:
            bounce_frames.add(frame_index)

    if not bounce_frames:
        return None
    return Clip(clip_dir.name, detections, bounce_frames)


def _calibrate_clip(clip_dir: Path, court_model, device: str):
    """Fit the court once per clip - the camera does not move within one."""
    import cv2

    frames_dir = clip_dir / "frames" if (clip_dir / "frames").is_dir() else clip_dir
    images = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        return None

    # Try a few frames rather than trusting the first: a player can be standing
    # across a baseline at frame 0 and wreck the keypoints for that frame only.
    for candidate in (images[len(images) // 2], images[0], images[-1]):
        frame = cv2.imread(str(candidate))
        if frame is None:
            continue
        try:
            fitted = calibrate(court_model.detect(frame))
        except ValueError:
            continue
        if fitted.is_reliable:
            return fitted
    return None


def build_dataset(clips: list[Clip]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features, labels, and the clip index each row came from."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []

    for index, clip in enumerate(clips):
        trajectory = from_detections(clip.detections)
        candidates = bounce_learned.propose(trajectory)
        if not candidates:
            continue
        row_labels = bounce_learned.label_candidates(candidates, clip.bounce_frames)
        for candidate, label in zip(candidates, row_labels):
            features.append(candidate.features)
            labels.append(int(label))
            groups.append(index)

    if not features:
        raise SystemExit("no candidates proposed - check the dataset layout")
    return np.vstack(features), np.array(labels), np.array(groups)


def evaluate(model, features: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
    )

    scores = model.predict_proba(features)[:, 1]
    predicted = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(average_precision_score(labels, scores)), 4)
        if labels.any() else None,
        "positives": int(labels.sum()),
        "total": int(len(labels)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="train_bounce")
    parser.add_argument("--data", required=True, help="TrackNet dataset root")
    parser.add_argument("--out", default="models/bounce_model.joblib")
    parser.add_argument("--court-model", default="models/keypoints_model.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-clips", type=int, default=None)
    parser.add_argument(
        "--features",
        default=None,
        help="cache path (.npz) for the extracted features. Written after "
             "extraction, and reused instead of re-extracting if it exists. "
             "Feature extraction calibrates the court once per clip and takes "
             "far longer than fitting the model, so caching is what makes "
             "retraining - under a different sklearn, or with different "
             "hyperparameters - a seconds-long operation instead of a rerun",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(args.features) if args.features else None
    if cache is not None and cache.exists():
        print(f"reusing cached features from {cache}")
        blob = np.load(cache, allow_pickle=False)
        cached_names = [str(n) for n in blob["feature_names"]]
        # A cache built before a feature was added or reordered would train a
        # model whose inputs silently mean something else at inference. Names
        # are stored precisely so that failure is loud.
        if cached_names != list(bounce_learned.FEATURE_NAMES):
            raise SystemExit(
                f"{cache} was built with different features "
                f"({cached_names}) - delete it and re-extract"
            )
        return _fit_and_report(
            args, blob["features"], blob["labels"], blob["groups"]
        )

    from tennis.detect import CourtDetector

    court_model = CourtDetector(args.court_model, device=device)

    root = Path(args.data)
    clip_dirs = sorted(
        p for p in root.rglob("*")
        if p.is_dir() and any(c.name.lower() == "label.csv" for c in p.iterdir())
    )
    if args.limit_clips:
        clip_dirs = clip_dirs[: args.limit_clips]
    print(f"found {len(clip_dirs)} labelled clips under {root}", flush=True)

    clips: list[Clip] = []
    for clip_dir in clip_dirs:
        clip = load_clip(clip_dir, court_model, device)
        if clip is None:
            print(f"  skipped {clip_dir.name} (no calibration or no bounces)")
            continue
        clips.append(clip)
        print(
            f"  {clip_dir.name}: {len(clip.detections)} samples, "
            f"{len(clip.bounce_frames)} bounces",
            flush=True,
        )

    if len(clips) < 4:
        raise SystemExit(
            f"only {len(clips)} usable clips - too few to hold any out honestly"
        )

    features, labels, groups = build_dataset(clips)
    if args.features:
        Path(args.features).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.features, features=features, labels=labels, groups=groups,
            feature_names=np.array(bounce_learned.FEATURE_NAMES),
        )
        print(f"cached features to {args.features}")
    print(
        f"\n{len(features)} candidates, {int(labels.sum())} of them real bounces "
        f"({labels.mean():.1%} positive) across {len(clips)} clips"
    )

    return _fit_and_report(args, features, labels, groups)


def _fit_and_report(args, features, labels, groups) -> int:
    # Whole clips held out. Deterministic so a rerun compares like with like.
    unique = np.unique(groups)
    held_out = set(unique[:: int(1 / TEST_SHARE)].tolist())
    is_test = np.isin(groups, list(held_out))
    print(f"holding out {len(held_out)} clips ({is_test.mean():.0%} of candidates)")

    detector = bounce_learned.LearnedBounceDetector().fit(
        features[~is_test], labels[~is_test]
    )
    train_metrics = evaluate(detector.model, features[~is_test], labels[~is_test])
    test_metrics = evaluate(detector.model, features[is_test], labels[is_test])

    print("\ntrain:", json.dumps(train_metrics))
    print("held out:", json.dumps(test_metrics))

    _report_importance(detector.model, features[is_test], labels[is_test])

    detector.save(args.out)
    print(f"\nsaved {args.out}")
    Path(args.out).with_suffix(".metrics.json").write_text(
        json.dumps(
            {"train": train_metrics, "held_out": test_metrics,
             "clips": len(clips), "held_out_clips": len(held_out)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def _report_importance(model, features: np.ndarray, labels: np.ndarray) -> None:
    """Which features carry the decision.

    Worth printing rather than assuming: if prominence alone dominates, the
    classifier has learned the threshold it was meant to replace and the extra
    features are decoration.
    """
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        return
    if not labels.any():
        return
    result = permutation_importance(
        model, features, labels, n_repeats=5, random_state=0, scoring="average_precision"
    )
    order = np.argsort(result.importances_mean)[::-1]
    print("\nfeature importance (permutation, held out):")
    for index in order:
        print(
            f"  {bounce_learned.FEATURE_NAMES[index]:<28} "
            f"{result.importances_mean[index]:+.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
