# Tennis Auto-Scoring

Feed in a video of two people playing tennis; get the score out.

Most tennis-CV projects stop at "here are bounding boxes and a speed readout."
This one goes the rest of the way: map the court properly, find where the ball
bounces, segment the video into rallies, decide who won each one, and run a
scoring state machine to produce **0 / 15 / 30 / 40 → games → sets**.

> **Status: pipeline complete end to end, not yet validated.** Detection,
> court calibration, bounce/hit detection, rally segmentation, point
> attribution and the scoring state machine all run on real video. What is
> missing is a match clip containing completed points to measure accuracy
> against, so the Results table below stays empty. No placeholder numbers.

---

## Why this isn't the tutorial repo

The first commit of this repo is [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis),
a well-known tutorial project (see [NOTICE.md](NOTICE.md)). It's a good scaffold
and a bad final product. Specifically:

| Baseline | This repo |
|---|---|
| Court mapping by nearest-keypoint proximity, scaled by hardcoded player heights (1.88 m / 1.91 m — the two pros in the sample video) | Real perspective transform via `cv2.findHomography` → metric court coordinates that work for any players |
| Court keypoints and player identities computed from **frame 0 only**, then reused for the whole video | Per-frame keypoint tracking with re-detection; identity re-association that survives ID switches |
| `read_from_stub=True` hardcoded — ships cached detections from the author's video and never runs the models on yours | Runs on your video; caching is opt-in and keyed to the input |
| fps hardcoded to 24 | Read from the file |
| **No evaluation of any kind** — no metrics, no validation loop, no test set | Eval harness: mAP@50 for ball detection, PCK for court keypoints, point-attribution accuracy against hand-labeled rallies |
| No scoring logic at all | Bounce detection → rally segmentation → point attribution → scoring state machine (unit-tested) |

Diff my work against the baseline:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD
```

---

## How scoring works

Single-camera in/out line calls are not reliable — Hawk-Eye uses ten calibrated
cameras, and this uses one. So the system doesn't judge lines. It scores from
**rally outcomes**, which are far more robust:

- two bounces on one side → the other player wins the point
- ball leaves the court without bouncing in → the last hitter loses the point
- ball fails to cross the net → the hitter loses the point

Those three events feed a plain-Python state machine (deuce, advantage, games,
sets) that is fully unit-testable with no video involved.

Every point is emitted with a confidence score, and low-confidence points are
flagged rather than silently guessed.

## Pipeline

```
video
  ├─ YOLOv8      → player boxes + track IDs
  ├─ YOLO (ft)   → ball boxes  → interpolation → smoothed trajectory
  └─ ResNet50    → 14 court keypoints → homography → metric court coords
                                                          │
                            bounce detection ─────────────┤
                            rally segmentation ───────────┤
                            point attribution ────────────┤
                                                          ▼
                                              scoring state machine
```

## The height problem, and how it is solved

A homography maps the court **plane**. A ball in the air is not on that plane,
so the camera ray through it pierces the plane *beyond* the ball's true
position — always away from the camera, and further the higher the ball is.

That error is not noise to be suppressed. It is a measurement. On a real rally:

| frame | image y | projected court y | what it is |
|---|---|---|---|
| 9 | 736 | **+20.5 m** | ball on the ground, near end |
| 40 | 311 | **+0.3 m** | ball on the ground, far end |
| 56 | 243 | **−4.7 m** | apex of the arc, several metres up |

The court runs 0 → 23.77 m, so −4.7 m is nonsense as a *position* and exactly
right as a *signal*. Projected court y falls as the ball rises and recovers as
it descends, which gives a clean rule:

- **local maximum** in projected court y → ball at its lowest: ground contact
- **local minimum** → apex of the flight: not an event at all

Raw image y cannot do this, because in a perspective view it confounds height
with depth — a ball high in the frame may be high up or merely far away. An
image-y detector (which is what the baseline used) reports the apex of every
arc as an event.

Two consequences run through the code:

- **Ground contact** is judged in court space, where a bounce genuinely lies on
  the mapped plane.
- **Proximity to a player** is judged in image space, normalised by the
  player's bounding-box height. An airborne ball's court coordinate is metres
  from the truth, so court-space distance to a player is meaningless; its image
  position is not. Normalising by box height lets one threshold serve a near
  player 400 px tall and a far player 90 px tall.

## Results

Measured, on the 7-second sample clip (Tesla T4):

| Metric | Value |
|---|---|
| Throughput | **9.4 fps** (0.13 fps on CPU — 65× slower) |
| Court calibration reliable | **8/8 attempts** |
| Median reprojection error | **0.86 px** (~1 cm on court) |
| Ball detection rate | **43.9%** of frames |
| Tests | **57 passing** |

Not yet measured — these need a clip containing completed points:

| Metric | Value |
|---|---|
| Bounce detection F1 | — |
| Point attribution accuracy | — |
| Games scored correctly | — |

**Known bottleneck: ball detection at 43.9%.** Bounces are decided by the
frames immediately around ground contact, which is exactly where the ball is
fastest and most motion-blurred. Every missed detection there is a missed
event. Fine-tuning the ball detector is the highest-value next step.

## Video requirements

The system assumes a fixed camera. For usable results:

- fixed tripod, single continuous shot (no cuts, no zoom)
- whole court visible in frame
- elevated, behind the baseline
- 30+ fps, 1080p

## Setup

Requires Python 3.11 or 3.12 (PyTorch has no 3.13+ wheels yet).

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```

Model weights are not in the repo. Download both into `models/`:

- fine-tuned YOLO ball detector — [link](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing) → `models/yolo5_last.pt`
- court keypoint ResNet50 — [link](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing) → `models/keypoints_model.pth`

Then:

```bash
python main.py --input input_videos/your_match.mp4
```

## Training

Training runs in Colab (free T4); local machine is inference only.

- Ball detector: `training/tennis_ball_detector_training.ipynb`
- Court keypoints: `training/tennis_court_keypoints_training.ipynb`

Note that the upstream keypoint notebook does not execute as published
(`items['kps']`, `model.stat_dict()`, an unused `val_loader`, and no validation
metric anywhere). Fixing it — and adding actual validation — is part of the work
here.
