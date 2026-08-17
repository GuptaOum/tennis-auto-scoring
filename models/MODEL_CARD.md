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

## Why this exists

A tennis ball is roughly 15 pixels across in a 1080p frame. That is close to
the smallest thing a detector of this architecture can represent at all, and it
is the reason off-the-shelf weights do poorly on tennis: at YOLO's default
`imgsz=640`, a 1080p frame is letterboxed down by a factor of three, leaving a
ball of about 5 pixels.

Two changes account for essentially all of the gain, and only one of them is
training:

1. **Serve at 960px.** On a 451-frame clip, raising inference resolution from
   640 to 960 took detection from 47.0% to 95.6% of frames *with no retraining
   at all*.
2. **Train at the resolution you serve at.** This model was fine-tuned at
   `imgsz=960`, which is why a 25.9M-parameter YOLOv8m beats the 86M-parameter
   YOLOv5l6u baseline it replaced.

## Results

Fine-tuned vs. the baseline weights, same validation set, both evaluated at
`imgsz=960`:

| metric | baseline (YOLOv5l6u, 86M) | this model (YOLOv8m, 25.9M) |
|---|---|---|
| mAP50 | 0.5878 | **0.8996** |
| mAP50-95 | 0.2212 | **0.4581** |
| precision | 0.633 | **0.925** |
| recall | 0.581 | **0.871** |

On a 15-second amateur clip the system had never seen, the ball is found in
**98.2%** of frames. Throughput is 9.3 fps end-to-end on a Tesla T4 (that
figure is the whole pipeline — players, ball and court — not this model alone).

### Resolution sweep

Inference resolution matters more than anything else here, and not
monotonically:

| imgsz | detection rate | mean confidence | speed |
|---|---|---|---|
| 640 | 47.0% | 0.329 | 32 fps |
| **960** | **95.6%** | 0.366 | 20 fps |
| 1280 | 78.3% | 0.394 | 12 fps |
| 1920 | 96.2% | 0.337 | 6 fps |

1920 gains nothing over 960 at three times the cost. **1280 scoring worse than
both is not noise** — it reproduces across runs, and is a letterboxing artefact
of this model's stride. Pick the resolution empirically rather than assuming
bigger is better.

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

A low confidence threshold (0.15) is deliberate. Missing the ball entirely
costs more downstream than a spurious box does, because a trajectory can be
smoothed but a gap cannot be filled.

## Intended use and limits

Built for footage from an **elevated camera behind the baseline** — broadcast
framing. It has not been evaluated on true overhead or drone angles.

- **It detects, it does not adjudicate.** The parent project never uses this to
  make line calls. A single camera cannot resolve whether a ball caught a line,
  and points are scored from rally outcomes instead — double bounce, landed out,
  failed to cross the net.
- **No 3D position or ball speed.** A homography maps the court *plane*, so an
  airborne ball projects to metres beyond where it truly is. Recovering real
  ball speed from one camera was attempted, measured, and abandoned; it needs a
  second camera or a known vertical reference.
- Trained on a modest dataset and evaluated on a small validation set. Treat the
  metrics above as evidence it works on this kind of footage, not as a general
  benchmark claim.

## The rest of the pipeline

This model is one of three that
[tennis-auto-scoring](https://github.com/GuptaOum/tennis-auto-scoring) runs per
frame. The other two are **not mine and are not republished here** — they are
listed so the system is reproducible and so credit lands where it belongs:

| stage | model | source |
|---|---|---|
| ball detection | this repo, YOLOv8m @ 960px | fine-tuned by me |
| player detection | `yolov8x.pt`, stock COCO weights | [Ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 |
| court keypoints | ResNet-50 regressing 14 landmarks | [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) |

### The player detector

`third_party/yolov8x.pt`, if present in this repo, is an **unmodified mirror** of
Ultralytics' stock COCO checkpoint, kept only so a deployment can fetch every
weight the pipeline needs from one place. It is not my training and not
tennis-specific: it detects the 80 COCO classes, of which this project uses
`person` (class 0), tracked across frames for stable identities. Redistributed
under AGPL-3.0 per Ultralytics' licence.

#### Using the person detector for tennis

Worth spelling out, because the naive version is wrong in a way that quietly
corrupts every downstream number: **a person detector does not give you players.**

On a Vienna ATP clip, stock `yolov8x.pt` tracked **six** people — the two
players, plus ball kids at the net posts, line judges along the tramlines, and
the chair umpire. Projected to court coordinates, the extras sat at positions
like `(14.6, 5.4)` and `(-3.4, -8.3)` metres: outside a court that is 10.97 m
wide and 23.77 m long.

Left unfiltered that is not untidy, it is incorrect. Every one of those tracks
feeds distance covered, average speed, net approaches and the coverage heatmaps,
so a ball kid jogging to a post is reported as a player covering ground. It also
breaks shot attribution, since a ball passing near a line judge looks like a
shot struck by one.

The fix uses two facts about singles tennis rather than a tuned threshold:

1. **Players are on the court; everyone else is beside or behind it.** Once feet
   are projected to metres through the court homography, distance from the court
   separates them cleanly. The bounds are deliberately generous — 3 m wide, 5 m
   behind — because a wide serve is returned from outside the tramline and deep
   balls are retrieved well behind the baseline.
2. **There is exactly one player per side of the net.** This settles the only
   hard case, a ball kid standing behind a real player: they compete against
   that player alone, and lose on distance from the court.

With no court calibration there are no metres to measure, so it falls back to the
two largest boxes and nothing downstream claims a distance. Implementation:
[`tennis/players.py`](https://github.com/GuptaOum/tennis-auto-scoring/blob/main/tennis/players.py).

Feet, not box centres, are what get projected — the homography maps the court
*plane*, so a position is only meaningful at ground level. Using the box centre
places everyone about a metre behind where they stand.

**The court keypoint model is not mirrored here, and cannot be.** It is the
upstream author's work, and that repository carries no licence file — so there is
no grant to redistribute its weights, whatever their quality. The project
downloads it from the source instead. It is worth naming precisely because it is
good: a 0.29 px median reprojection error on courts it has never seen, which is
why this project deliberately does *not* retrain it.

Everything on top of those detections — homography calibration, ground-contact
detection, rally segmentation, serve and fault logic, tennis scoring, shot
placement, the HTML report and the annotated video — is mine.

## Training

- Base weights: `yolov8m.pt`
- `imgsz=960`, Tesla T4 (g4dn.xlarge)
- Dataset: [tennis-ball-detection-6](https://universe.roboflow.com/) (Roboflow)
- Script: [`training/train_ball.py`](https://github.com/GuptaOum/tennis-auto-scoring/blob/main/training/train_ball.py)

## License

AGPL-3.0, inherited from Ultralytics YOLOv8. If you use this in a networked
service, that license has requirements — check them before deploying.

## Citation

```bibtex
@software{tennis_auto_scoring,
  author = {Gupta, Oum},
  title  = {tennis-auto-scoring: automatic tennis scoring from a single camera},
  url    = {https://github.com/GuptaOum/tennis-auto-scoring},
  year   = {2026}
}
```
