# Tennis tracking — YOLO + TrackNet proof of concept

Track the **ball**, the **players** and the **court** in a tennis video, in one
pass, and draw the result. Three targets, three different techniques, because
they fail in three different ways.

```bash
python -m tennis.track --input input_videos/clip.mp4 --out output/tracked
```

Output: an annotated video and `tracks.json` with per-frame ball position,
player boxes and court corners.

![proof of concept](output/poc/preview.png)

## The three trackers

| target | technique | why not something simpler |
|---|---|---|
| **ball** | TrackNet — 3 stacked frames → heatmap | A ball is ~10 px and motion-blurred into a streak. In a single frame it is often indistinguishable from a line marking. Temporal context is not an optimisation here; it is the only signal. |
| **players** | YOLOv8 per frame | Large, high-contrast, slow. A per-frame detector is the right tool and needs no temporal help. |
| **court** | keypoints → snap to painted lines → optical-flow tracking | A single fit is stale within a second on a moving camera. |

## Measured results

Two clips, 1920×1080, Tesla T4, 5.3 fps:

| | `video_project3` (edited) | `SAMPLE` (single shot) |
|---|---|---|
| frames | 2237 | 1441 |
| ball found (TrackNet) | **73.1%** | **74.0%** |
| court detections | 77 | 47 |
| court frames carried by optical flow | 2149 | **1394** |
| court lost | 11 | **0** |

The contrast is the point. `SAMPLE` is one continuous shot and the court is
**never lost** — 47 detections cover 1441 frames, the other 1394 carried by
optical flow. `video_project3` is an edited video containing **22 hard cuts**,
and every one of the 11 losses is a cut: the tracker refuses to track through a
discontinuity rather than emit a confident wrong answer.

## Four things this project actually learned

Each of these was a measurement that contradicted an assumption.

**1. The bottleneck was a parameter, not the model.** A tennis ball is ~15 px at
1080p; YOLO's default `imgsz=640` letterboxes it to ~5 px. Raising inference
resolution to 960 took ball detection from **47% to 95.6% with no retraining**.
The optimum tracks *apparent ball size*, so a wide broadcast shot wants 1600
where a tight amateur clip wants 960.

**2. Reprojection error does not measure court accuracy.** It only asks whether
one homography explains the keypoint model's own 14 points — so a model wrong in
a mutually consistent way scores perfectly. Measured across two clips,
reprojection sat at **~0.77 px while the corners were a median 14 px and 68 px
off the painted lines**, once by 129 px. The two are uncorrelated.

The fix needs no retraining: use the keypoint model as an initial guess, then
optimise the homography so the projected lines land on *detected line pixels*.
Held out on lines the fit never saw, worst corner **129 px → 7 px**.

**3. A bounce is a corner, not a peak.** The bounce detector proposed candidates
only where the ball's projected court y was a local maximum. Against 136 real
labelled bounces that is true **2% of the time** — the height signal also carries
down-court travel, and that term usually dominates. Proposal recall was 34.6%.
Proposing on the upward kink in image y instead, normalised by the segment's
median speed so it is resolution-independent, took it to **99.3%**. No classifier
recovers an event that was never proposed.

**4. A person detector finds people, not players.** One professional clip
returned six tracks — two players plus ball kids, line judges and the chair
umpire — and the tracker reassigned ids constantly, producing sixteen
"identities" for two people. Filtering by court position and *exactly one player
per side of the net* is what turns person detection into player detection.
Identity is the side, not the tracker id.

## Models

Two were trained here, on measured data:

| model | result | link |
|---|---|---|
| Ball detector (YOLOv8m fine-tune) | mAP50 **0.5878 → 0.8996** | [kjfk/tennis-ball-detector-yolov8](https://huggingface.co/kjfk/tennis-ball-detector-yolov8) |
| Bounce detector (gradient boosting) | **0.951** PR-AUC, leave-one-match-out | [kjfk/tennis-bounce-detector](https://huggingface.co/kjfk/tennis-bounce-detector) |

Leave-one-match-out is the honest split for the bounce model: clips from one
match share a court, a camera and a pair of players, so a clip-level split lets
the model see the same camera in training.

Three more are **mirrored, not trained here** — each card credits its author:
[TrackNet](https://huggingface.co/kjfk/tracknet-tennis-ball-mirror) (yastrebksv),
[court keypoints](https://huggingface.co/kjfk/tennis-court-keypoints-mirror)
(abdullahtarek), [YOLOv8x](https://huggingface.co/kjfk/yolov8x-coco-mirror)
(Ultralytics).

## What this does not do

**It does not score the match.** That was attempted and does not work yet: point
counts run roughly 2–3× high. Two causes are known and unfixed — broadcast camera
cuts are not treated as rally boundaries, and a *missed* hit is indistinguishable
from *no* hit, which fires the double-bounce rule.

**It does not judge in/out.** A line call needs the exact frame of ground
contact. The bounce detector is accurate to a couple of frames, and a ball one
frame early is still airborne — its ray pierces the court plane well beyond
where it lands. With the court fitted correctly, 26 of 26 landings read "out".
An earlier 7-in/6-out result was not better; it was a mis-fitted court large
enough to swallow the overshoot.

Both are left in the code and both are off by default. The scoring modules
(`scoring.py`, `rally.py`, `serve.py`) still exist and still pass their tests.

## Layout

```
tennis/       track.py (entry point), tracknet.py, court_track.py,
              court_refine.py, detect.py, players.py, trajectory.py, ...
training/     train_ball.py, train_bounce.py, sweep_ball.py
models/       weights
tests/        202 tests
```

## Setup

```bash
pip install -r requirements.txt
```

Weights go in `models/`. A GPU is strongly recommended — YOLOv8x runs ~14 s/frame
on CPU at 1080p, versus 5.3 fps on a T4.

## Credits

Built on [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis)
(see `NOTICE.md`). Ball tracking follows
[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) and
[TennisProject](https://github.com/yastrebksv/TennisProject); the bounce model is
trained on the TrackNet tennis dataset.
