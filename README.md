# Tennis Auto-Scoring

Feed in a video of two people playing tennis; get the score out.

Most tennis-CV projects stop at "here are bounding boxes and a speed readout."
This one goes the rest of the way: map the court properly, find where the ball
bounces, segment the video into rallies, decide who won each one, and run a
scoring state machine to produce **0 / 15 / 30 / 40 → games → sets**.

> **Status: in progress.** The baseline pipeline (player detection, ball
> detection, court keypoints) runs. The scoring layer is being built. Numbers in
> the Results table below will be filled in from the eval harness — no
> placeholder claims.

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

## Results

Filled in as the eval harness produces them. Empty rows mean not yet measured.

| Metric | Value |
|---|---|
| Ball detection mAP@50 | — |
| Court keypoint PCK@0.05 | — |
| Bounce detection F1 | — |
| Point attribution accuracy | — |
| Games scored correctly | — |

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
