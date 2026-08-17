# Model weights

The pipeline runs three models per frame. They have different owners and
different licences, which decides what can be published and what cannot.

| file | stage | size | owner | publishable |
|---|---|---|---|---|
| `ball_finetuned.pt` | ball detection | 50 MB | **mine** — fine-tuned YOLOv8m | yes |
| `../yolov8x.pt` | player detection | 131 MB | Ultralytics, stock COCO | yes, AGPL-3.0 |
| `keypoints_model.pth` | court keypoints | 91 MB | [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis) | **no — unlicensed** |
| `yolo5_last.pt` | *(retired)* | 165 MB | upstream | no — unlicensed |

`yolo5_last.pt` was the baseline's ball detector. The fine-tune replaced it and
nothing loads it any more; it is kept only so the before/after comparison can be
re-run.

## Why the keypoint model is not published

That repository carries **no LICENSE file**. Public on GitHub is not the same as
open source: with no licence, default copyright applies and all rights are
reserved. GitHub's terms permit viewing and forking on GitHub, not redistributing
the weights elsewhere. So the right to republish does not exist to be exercised,
regardless of the model's quality — and it is good, at 0.29 px median
reprojection error on unseen courts.

Two ways out, in increasing order of value:

1. Ask the upstream author to add a licence.
2. **Train our own** on [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
   (**MIT**, 8,841 images) using [`../training/train_keypoints.py`](../training/train_keypoints.py).
   Then it is ours, publishable and citable, and the project is fully owned end
   to end. Note the tradeoff: the upstream model is already at 0.29 px, so this
   is about ownership rather than accuracy.

## Getting the weights

`ball_finetuned.pt` is the CLI default and is what the fine-tune produced.
The other two are downloaded from their own sources — see the project README.

## Publishing to Hugging Face

One-time login, with a **Write** token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
.venv/Scripts/hf.exe auth login
```

Then publish. The script uploads the weights plus
[`MODEL_CARD.md`](MODEL_CARD.md) as the repo README:

```bash
python training/upload_to_hf.py
```

### Options

| flag | effect |
|---|---|
| `--with-player-model` | also mirror `yolov8x.pt` to `third_party/` (131 MB) |
| `--private` | create the repo unlisted |
| `--name NAME` | repo name (default `tennis-ball-detector-yolov8m`) |
| `--owner OWNER` | account or org (default: the logged-in user) |

`--with-player-model` exists for deployment convenience — so a deploy can fetch
every publishable weight from one place instead of two. It is **off by default**,
because a mirror of unmodified stock weights is not a result and should not be
the first thing a visitor to the repo sees.

The script deliberately takes **no token argument**. A token passed on a command
line lands in shell history; `hf auth login` keeps it out.

Re-running is safe: `create_repo` uses `exist_ok=True` and uploads overwrite, so
this is also how you push an updated model card.

### When the keypoint model is ours

Once `train_keypoints.py` has produced weights we own, publish them separately
rather than adding them here — they are a different architecture (ResNet-50
keypoint regression, not a detector) and belong in their own repo with their own
card and PCK metrics.

## Troubleshooting

| symptom | cause |
|---|---|
| `403 Forbidden` | the token is Read, not Write |
| `not logged in` | run `hf auth login` first |
| `missing weights` | run the fine-tune, or fetch `ball_finetuned.pt` |
