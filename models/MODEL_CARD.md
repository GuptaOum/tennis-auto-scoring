---
license: agpl-3.0
tags:
  - object-detection
  - ultralytics
  - yolov8
  - tennis
  - sports-analytics
library_name: ultralytics
pipeline_tag: object-detection
---

# Tennis ball detector (YOLOv8m, 960px)

A fine-tuned tennis-ball detector for single-camera broadcast and amateur
footage. It is the detection stage of
[tennis-auto-scoring](https://github.com/GuptaOum/tennis-auto-scoring), which
turns a match video into an automatic score.

**One class:** `tennis ball`. 25.9M parameters, 50 MB.

## What is in this repo

The pipeline runs **three** models per frame. Two are here:

| file | stage | whose | status |
|---|---|---|---|
| `ball_finetuned.pt` | ball detection | **fine-tuned by me** | here, 50 MB |
| `third_party/yolov8x.pt` | person detection | Ultralytics, stock COCO | mirrored, 131 MB |
| *court keypoints* | 14 court landmarks | — | **planned** |

## Why this exists

A tennis ball is ~15 px across in a 1080p frame — near the smallest thing this
architecture can represent. At YOLO's default `imgsz=640` a 1080p frame is
letterboxed down 3×, leaving a ball of ~5 px. That is why off-the-shelf weights
do poorly on tennis.

Two changes account for nearly all the gain, and only one is training:

1. **Serve at 960px.** On a 451-frame clip, going from 640 to 960 took detection
   from 47.0% to 95.6% of frames *with no retraining at all*.
2. **Train at the resolution you serve at.** Fine-tuned at `imgsz=960`, which is
   why this 25.9M-param YOLOv8m beats the 86M-param YOLOv5l6u it replaced.

## Results

Same validation set, both evaluated at `imgsz=960`:

| metric | baseline (YOLOv5l6u, 86M) | this model (YOLOv8m, 25.9M) |
|---|---|---|
| mAP50 | 0.5878 | **0.8996** |
| mAP50-95 | 0.2212 | **0.4581** |
| precision | 0.633 | **0.925** |
| recall | 0.581 | **0.871** |

On an unseen 15-second amateur clip the ball is found in **98.2%** of frames.
End-to-end pipeline throughput is 9.3 fps on a Tesla T4.

### Resolution sweep

| imgsz | detection | mean conf | speed |
|---|---|---|---|
| 640 | 47.0% | 0.329 | 32 fps |
| **960** | **95.6%** | 0.366 | 20 fps |
| 1280 | 78.3% | 0.394 | 12 fps |
| 1920 | 96.2% | 0.337 | 6 fps |

1920 gains nothing over 960 at 3× the cost. **1280 scoring worse than both is
not noise** — it reproduces, and is a letterboxing artefact of this model's
stride. Choose resolution empirically, not by assuming bigger is better.

## Usage

```python
from ultralytics import YOLO

model = YOLO("ball_finetuned.pt")

# imgsz=960 is not optional - at the 640 default you lose half the detections.
result = model.predict("frame.jpg", imgsz=960, conf=0.15)[0]

if len(result.boxes):
    # A tennis ball is one object; extra boxes are line markings and shoes.
    best = max(result.boxes, key=lambda b: float(b.conf.item()))
    print(best.xyxy.tolist()[0], float(best.conf.item()))
```

The low threshold (0.15) is deliberate: a missed ball costs more downstream than
a spurious box, because a trajectory can be smoothed but a gap cannot be filled.

## Limits

Built for an **elevated camera behind the baseline** (broadcast framing); not
evaluated on overhead or drone angles.

- **It detects, it does not adjudicate.** The parent project makes no line calls —
  one camera cannot resolve whether a ball caught a line. Points are scored from
  rally outcomes: double bounce, landed out, failed to cross the net.
- **No ball speed or 3D position.** A homography maps the court *plane*, so an
  airborne ball projects beyond where it truly is. Recovering speed from one
  camera was attempted, measured and abandoned.
- Modest training and validation sets. Treat the metrics as evidence for this
  kind of footage, not a general benchmark.

## The person detector

`third_party/yolov8x.pt` is an **unmodified mirror** of Ultralytics' stock COCO
checkpoint, kept here only so a deployment can fetch every weight from one place.
Not my training, not tennis-specific: it uses class 0 `person`, tracked for
stable identities. AGPL-3.0 per Ultralytics' licence.

**A person detector does not give you players** — and getting this wrong corrupts
results silently. On a Vienna ATP clip it tracked **six** people: two players plus
ball kids, line judges and the chair umpire, at court positions like `(14.6, 5.4)`
and `(-3.4, -8.3)` m — outside a 10.97 × 23.77 m court. Unfiltered, all of them
feed distance covered, speed, net approaches and coverage heatmaps, so a ball kid
is reported as a player covering ground.

The fix uses two facts about singles, not tuned thresholds: **players are on the
court** once feet are projected to metres (bounds are generous — 3 m wide, 5 m
behind — since wide serves are returned outside the tramline), and **there is one
player per side of the net**, which settles the hard case of a ball kid standing
behind a real player. Feet are projected, not box centres: the homography maps the
plane, so a centre puts everyone a metre behind where they stand. See
[`tennis/players.py`](https://github.com/GuptaOum/tennis-auto-scoring/blob/main/tennis/players.py).

## The court keypoint model — planned

The third stage is not here. The project currently uses the upstream author's
ResNet-50 ([abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis)),
which reaches a **0.29 px** median reprojection error on unseen courts — but that
repository carries no licence file, so there is no grant to redistribute its
weights.

**A replacement trained by me is planned**, on the MIT-licensed
[TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) dataset
(8,841 images), via
[`training/train_keypoints.py`](https://github.com/GuptaOum/tennis-auto-scoring/blob/main/training/train_keypoints.py).
Two caveats: beating 0.29 px is unlikely, so the motivation is ownership and
licensing rather than accuracy; and it will ship as its own repo, since a keypoint
regressor needs its own metric (PCK, not mAP).

Everything above the detections — homography calibration, ground-contact
detection, rally segmentation, serve and fault logic, scoring, shot placement, the
HTML report and the annotated video — is mine.

## Training

- Base weights `yolov8m.pt`, `imgsz=960`, Tesla T4 (g4dn.xlarge)
- Dataset: tennis-ball-detection-6 (Roboflow)
- Script: [`training/train_ball.py`](https://github.com/GuptaOum/tennis-auto-scoring/blob/main/training/train_ball.py)

## License

AGPL-3.0, inherited from Ultralytics YOLOv8. Networked-service use carries
obligations — check them before deploying.

## Citation

```bibtex
@software{tennis_auto_scoring,
  author = {Gupta, Oum},
  title  = {tennis-auto-scoring: automatic tennis scoring from a single camera},
  url    = {https://github.com/GuptaOum/tennis-auto-scoring},
  year   = {2026}
}
```
