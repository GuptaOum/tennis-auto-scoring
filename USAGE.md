# Running it

## Quick start (Windows, local)

```bash
analyze.bat input_videos\your_match.mp4
```

Results appear in `output/`:

| file | what it is |
|---|---|
| `annotated.mp4` | your video with players, ball, court and live positions drawn on |
| `report.json` | every number the run produced |
| `ball_track.json` | the ball's path, frame by frame, in court metres |

The terminal prints a readable summary: distance covered, average and top
speed, net approaches, a court-occupancy grid per player, and the scoring
attempt.

## Speed, and why you probably want the GPU

Inference is the whole cost, and it is ~70× faster on a GPU:

| machine | throughput | 10 seconds of 1080p30 |
|---|---|---|
| this laptop (CPU) | 0.13 fps | ~38 minutes |
| Tesla T4 (EC2) | 9–12 fps | ~30 seconds |

So for anything past a few seconds locally, cut the clip down first or pass
`--limit`:

```bash
analyze.bat input_videos\your_match.mp4 --limit 150
```

`--limit 150` is five seconds at 30 fps — about 20 minutes on CPU.

## Running on the GPU box

```bash
scp -i "C:\Users\hp\.ssh\face-attendance.pem" your_match.mp4 ubuntu@15.207.237.253:~/tennis-auto-scoring/repo/input_videos/
```

```bash
ssh -i "C:\Users\hp\.ssh\face-attendance.pem" ubuntu@15.207.237.253 "cd ~/tennis-auto-scoring/repo && .venv/bin/python -m tennis.cli --input input_videos/your_match.mp4 --out output"
```

```bash
scp -i "C:\Users\hp\.ssh\face-attendance.pem" ubuntu@15.207.237.253:~/tennis-auto-scoring/repo/output/annotated.mp4 .
```

## Getting a clip

`yt-dlp` works from a home connection but **not** from the EC2 box — YouTube
blocks datacenter IP ranges with a bot check that needs browser cookies. So
download locally, then upload.

```bash
yt-dlp -f "bv*[height<=1080]" --download-sections "*1:30-1:45" -o clip.mp4 "<url>"
```

`--download-sections` fetches only that time range instead of the whole video.

### What makes a clip work

| | |
|---|---|
| camera | fixed, single continuous shot — no cuts, replays or zooms |
| angle | elevated, behind a baseline, whole court in frame |
| content | **at least one completed point**, otherwise there is nothing to score |
| quality | 720p or better, 25 fps or better |

The last one is the one people miss. A rally fragment produces movement
analysis but no score, because no point finished.

## Options

| flag | effect |
|---|---|
| `--limit N` | stop after N frames |
| `--start N` | begin at frame N |
| `--no-video` | skip the annotated video, write JSON only (much faster) |
| `--device cuda` | force GPU (default `auto` detects one) |
| `--recalibrate-every N` | re-detect court keypoints every N frames (default 30) |

## Reading the output

```
player 1 (near side): 17.9 m covered, avg 9.1 km/h, top 44.6 km/h, 0 net approaches
                
                
                
                
                
      ++@@::==  
```

The grid is court occupancy: rows run far baseline (top) to near baseline
(bottom), columns left to right. Darker means more time spent there. The
example shows a player who stayed pinned to their baseline for the whole clip.

```
score              : no completed point found in this clip
```

This is the system declining rather than failing. A score is only printed when
a rally ending was actually identified — no invented 0-0.
