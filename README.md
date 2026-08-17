# Tennis Auto-Scoring

Feed in a video of two people playing tennis; get the score and a full match
analysis out.

Most tennis-CV projects stop at "here are bounding boxes and a speed readout."
This one maps the court properly, finds where the ball bounces, works out who
won each rally, and runs a real scoring state machine — **0 / 15 / 30 / 40 →
deuce → games → sets** — while reporting shot placement, court coverage and
player movement in metres.

```bash
analyze.bat input_videos/match.mp4
```

---

## Status

Working and measured on real footage. **90 tests passing.**

| Component | State |
|---|---|
| Court detection → homography | **0.29 px** reprojection (~1 cm), works on unseen courts |
| Ball detection | **98.2%** of frames, mAP50 **0.90** |
| Player tracking in metres | distance, speed, coverage, net approaches |
| Bounce / hit separation | working, confidence-scored |
| Rally segmentation | ends rallies at the event that ends them |
| Point attribution | working, confidence **0.73** on real footage |
| Scoring state machine | complete — deuce, advantage, tiebreak, sets |
| Shot placement | depth, width, direction, per-player zone grids |
| Runs on arbitrary video | any resolution, any fps |

**Not yet validated:** point-attribution accuracy against hand-labelled
rallies. That needs a clip with several completed points, and until it exists
no accuracy percentage is claimed here.

**Known limits:** ball speed is not reported (see
[below](#what-cannot-be-done-with-one-camera)); serve and fault detection are
not implemented; nothing has been run on a video longer than 16 seconds.

---

## Why this isn't the tutorial repo

The first commit is [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis),
a widely-forked tutorial project (see [NOTICE.md](NOTICE.md)) — a good scaffold
and a bad final product.

| Baseline | This repo |
|---|---|
| Court mapping by nearest-keypoint proximity, scaled by hardcoded player heights (1.88 m / 1.91 m — the two pros in its sample video) | Real `cv2.findHomography` → metric court coordinates, any players |
| Court keypoints and player identity fixed from **frame 0** for the whole video | Re-estimated periodically; identities ranked over the whole clip |
| `read_from_stub=True` hardcoded — replays cached detections from the author's video, never runs on yours | Runs on your video; caching is opt-in and keyed to a hash of the file |
| fps hardcoded to 24 (its own sample is 30 — every speed 25% off) | Read from the file |
| Ball detector called at `imgsz=640` | **960** — detection 47% → 96% with no retraining |
| Ball detector mAP50 **0.588** | Fine-tuned: **0.900** |
| Speed = straight line between two shots ÷ hardcoded fps | Player speed from the homography; ball speed **not claimed** |
| Live bug: player 1's speed ÷ player 2's shot count | Per-track stats |
| Whole video decoded into RAM | Streaming |
| **No evaluation of any kind** | 90 tests, measured before/after on every model change |
| No scoring logic at all | Full pipeline → scoring state machine |

Diff the work against the baseline:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD
```

---

## Three findings worth the read

### 1. The bottleneck was a parameter, not the model

A tennis ball is ~15 px across in a 1080p frame. YOLO's default `imgsz=640`
letterboxes the frame down threefold, leaving ~5 px — near the smallest object
the network can represent. Measured over 451 frames:

| imgsz | detection | mean conf | speed |
|---|---|---|---|
| 640 (default) | 47.0% | 0.329 | 32 fps |
| **960** | **95.6%** | 0.366 | 20 fps |
| 1280 | 78.3% | 0.394 | 12 fps |
| 1920 | 96.2% | 0.337 | 6 fps |

1920 gains nothing over 960 at 3× the cost, and 1280 scoring below both is a
letterboxing artefact of the model's stride that reproduces across runs — which
is why the default is chosen from measurement rather than by assuming bigger is
better.

### 2. The homography's error is a height sensor

A homography maps the court **plane**. A ball in the air is not on that plane,
so the camera ray through it pierces the plane *beyond* the ball — always away
from the camera, further the higher it is. On a real rally:

| frame | image y | projected court y | what it is |
|---|---|---|---|
| 9 | 736 | **+20.5 m** | ball on the ground, near end |
| 40 | 311 | **+0.3 m** | ball on the ground, far end |
| 56 | 243 | **−4.7 m** | apex of the arc, metres up |

The court runs 0 → 23.77 m, so −4.7 m is nonsense as a *position* and exactly
right as a *signal*:

- **local maximum** in projected court y → ground contact (bounce or hit)
- **local minimum** → apex of the flight, not an event

Raw image y cannot do this — in a perspective view it confounds height with
depth. An image-y detector reports the apex of every arc as an event, which is
precisely what the baseline did.

This splits the pipeline across two coordinate spaces, each used where it is
valid: **ground contact** in court metres (a bounce is on the plane), **player
proximity** in image pixels normalised by box height (an airborne ball's court
coordinate is metres from the truth, its image position is not).

### 3. Better detection broke rally segmentation

Segmentation originally split the event stream on two-second silences,
assuming the ball goes untracked between points. Raising detection to 98%
destroyed that assumption — the silences vanished and a 15-second clip
collapsed into one rally.

Boundaries are now read from structure, not absence: a rally ends at the event
that ends it. Since the same events supply the winner and the reason, a rally
boundary can never disagree with the point it produced.

---

## How scoring works

Single-camera in/out line calls are not reliable — Hawk-Eye uses ten calibrated
cameras. So the system never judges a line to decide a point. It scores from
**rally outcomes**, which are coarser and far more robust:

- two bounces on one side → the other player wins the point
- ball bounces outside the court → whoever hit it loses
- ball never crosses the net → the hitter loses

Those feed a plain-Python state machine (deuce, advantage, tiebreak, games,
sets) that is fully unit-testable with no video involved. Keeping that boundary
sharp is what makes the error rate measurable: a wrong scoreline is either a
wrong point attribution (vision) or a rules bug (provable by test), never an
ambiguous mix.

Every point carries a confidence. Rallies the system cannot read stay
**undecided** rather than being guessed, and the count of those is reported —
it is the honest denominator for any accuracy claim.

---

## Pipeline

```
video
  ├─ YOLOv8      → player boxes + track IDs
  ├─ YOLO (ft)   → ball boxes → interpolation → smoothed trajectory
  └─ ResNet50    → 14 court keypoints → homography → metric court coords
                                                          │
                          bounce / hit detection ─────────┤
                          rally segmentation ─────────────┤
                          point attribution ──────────────┤
                                                          ▼
                                      scoring state machine + analytics
```

---

## Results

End to end on a 15-second amateur singles clip the system had never seen,
downloaded from YouTube. Tesla T4.

| Metric | Value |
|---|---|
| Ball detection | **98.2%** of frames, mean confidence **0.688** |
| Court calibration | **14/16** reliable, median **0.29 px** |
| Events | 18 bounces, 17 hits |
| Points | 1 rally scored, confidence **0.726** |
| Throughput | **9.3 fps** (0.13 fps on CPU) |

### Ball detector fine-tune

Same held-out val set, same `imgsz=960`:

| Metric | Baseline `yolo5_last.pt` | Fine-tuned |
|---|---|---|
| mAP50 | 0.5878 | **0.8996** |
| mAP50-95 | 0.2212 | **0.4581** |
| Precision | 0.6330 | **0.9252** |
| Recall | 0.5806 | **0.8713** |
| Size | 165 MB (86M params) | **50 MB** (26M params) |

mAP50-95 more than doubling is the meaningful part: tighter boxes mean a more
accurate ball *position*, which is what bounce localisation consumes. A
26M-param model beat an 86M-param one on identical data, largely by training at
the resolution it is actually served at.

Run artifacts in [training/ball_ft_run/](training/ball_ft_run/).

---

## What cannot be done with one camera

**Ball speed is not reported, deliberately.** Measuring it needs the ball's 3-D
position, and a single view cannot supply it. The physically correct route —
recover the camera centre from the homography, lift each plane-projected
position to 3-D, fit a ballistic arc between bounces — was implemented and
measured:

| clip | recovered camera height | residual |
|---|---|---|
| amateur | 2.1 m | 0.464 |
| sample | degenerate, no solution | — |

A 2.1 m camera cannot see a whole court, and the residual — disagreement
between the two rotation constraints — should sit under 0.1. The 14 keypoints
are regressed independently on a squashed 224×224 input: accurate in aggregate,
which is all a plane mapping needs, but not geometrically consistent enough to
invert into 3-D. Doing it properly needs a second camera, a calibration target,
or a known vertical reference in frame.

Player speeds are unaffected — players stand *on* the court plane, so the
homography applies to them directly.

---

## Setup

Python 3.11–3.13.

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

Weights are not in the repo:

- `models/ball_finetuned.pt` — produced by `training/train_ball.py`
- `models/keypoints_model.pth` — [court keypoint ResNet50](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)

See [USAGE.md](USAGE.md) for running, options, and what makes a clip work.

## Training

```bash
python training/train_ball.py --data data.yaml --baseline models/yolo5_last.pt
python training/train_keypoints.py --data-dir data --epochs 60
```

Both evaluate against a baseline and write a comparison, so results are
measured deltas rather than claims. The upstream keypoint notebook does not
execute as published — `items['kps']`, `model.stat_dict()`, a `val_loader`
built and never used, no validation metric anywhere. It has been rewritten with
PCK evaluation and best-weight checkpointing.

Ball dataset: [Roboflow tennis-ball-detection](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6) (CC BY 4.0).
Court keypoints: [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) (MIT), 8,841 images.

---

## Next

1. **Measure point-attribution accuracy** against hand-labelled rallies — the
   one number that turns this from "runs" into "works, at X%"
2. Serve and fault detection
3. Validate on a full match rather than 15-second clips
