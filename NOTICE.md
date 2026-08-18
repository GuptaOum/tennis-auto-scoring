# Attribution

The initial commit of this repository is the source code of
[abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis),
at commit `d557527793820f1e6b06872256824255facd47fd` (2024-03-22), used as a
starting baseline.

Two modifications were made to that baseline before committing:

1. **Credentials redacted.** The upstream notebooks contain a live Roboflow API
   key and a full Google session cookie header. Both were replaced with
   placeholders rather than republished.
2. **Notebook outputs cleared**, to keep the diff readable.

Everything after the initial commit is my own work. To see exactly what I
changed, diff against the first commit:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD
```

The upstream project is a video tutorial companion repo and carries no license
file. It is credited here in full; if the author requests removal of the
baseline commit, it will be rewritten out of history.

## Licence scope

This repository is released under the MIT Licence (see `LICENSE`). That licence
covers **my own work** — everything after the initial commit.

It does not, and cannot, relicense the baseline. The upstream project
[abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis)
publishes **no licence**, so its code remains under its author's rights. None of
it survives in the working tree — the baseline scaffolding (`utils`, `trackers`,
`court_line_detector`, `mini_court`, `constants`, `analysis`, `tracker_stubs`,
`main.py`, `yolo_inference.py`) was removed — but the first commit is still in
the history, so anyone reusing this repo should be aware of what that commit is.

Model weights are separate again and are not covered by this licence:

| weight | source | licence |
|---|---|---|
| `tracknet_tennis.pt` | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) | see upstream |
| `keypoints_model.pth` | abdullahtarek/tennis_analysis | no licence published |
| `yolov8x.pt` | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |

The Ultralytics AGPL-3.0 term is the one to check before any commercial use:
it applies to YOLO and to work that links it.
