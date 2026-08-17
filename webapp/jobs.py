"""Job model and the single-slot runner behind the web UI.

Analysis runs in a subprocess rather than in-process, for three reasons that
all showed up in practice: a decoder segfault or a CUDA OOM kills one job
instead of the whole server, the CLI stays the single source of truth for what
a run means, and stdout can be parsed for real progress.

Jobs run strictly one at a time. There is one GPU; two concurrent runs would
contend for it and finish no sooner, but they would make both progress bars
lie.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = Path(__file__).resolve().parent / "jobs_data"
JOBS_DIR.mkdir(exist_ok=True)

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB

# "  250 frames (9.3 fps)" - the CLI's progress line, printed every 25 frames.
PROGRESS_RE = re.compile(r"^\s*(\d+) frames \(([\d.]+) fps\)")
# "input: match.mp4: 1920x1080 @ 30 fps, 5400 frames (180.0s)"
PROBE_RE = re.compile(r"^input: .*?, (\d+) frames \(([\d.]+)s\)")

STATUSES = ("queued", "running", "done", "failed")


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    frames_total: int | None = None
    frames_done: int = 0
    fps: float | None = None
    duration_s: float | None = None
    error: str | None = None
    options: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def progress(self) -> float:
        if self.status == "done":
            return 1.0
        if not self.frames_total or not self.frames_done:
            return 0.0
        # Never show 100% while work remains - the report is written after the
        # last frame, and a bar that sits full for a minute reads as a hang.
        return min(self.frames_done / self.frames_total, 0.99)

    @property
    def eta_seconds(self) -> float | None:
        """Seconds remaining, from the rate this job is actually achieving.

        Measured throughput, not a nominal figure: the rate depends on the
        footage, and a wrong ETA on an hour-long job is worse than none.
        """
        if self.status != "running" or not (self.fps and self.frames_total):
            return None
        remaining = max(self.frames_total - self.frames_done, 0)
        return remaining / self.fps if self.fps > 0 else None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["progress"] = round(self.progress, 4)
        data["eta_seconds"] = self.eta_seconds
        data["elapsed_seconds"] = (
            round((self.finished or time.time()) - self.started, 1)
            if self.started
            else None
        )
        data["has_report"] = (self.output_dir / "report.html").exists()
        data["has_video"] = (self.output_dir / "annotated.mp4").exists()
        return data

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "job.json").write_text(json.dumps(asdict(self)), encoding="utf-8")

    def video_path(self) -> Path | None:
        if not self.dir.exists():
            return None
        for path in sorted(self.dir.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                return path
        return None


def build_command(job: Job, video: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tennis.cli",
        "--input",
        str(video),
        "--out",
        str(job.output_dir),
        "--device",
        job.options.get("device") or "cuda",
    ]
    if job.options.get("no_video"):
        command.append("--no-video")
    if job.options.get("limit"):
        command += ["--limit", str(job.options["limit"])]
    return command


def friendly_error(job: Job) -> str:
    """Turn the tail of a failed run into something a person can act on."""
    log_path = job.dir / "run.log"
    tail = ""
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    if "UndecodableVideo" in tail:
        return (
            "The file opens but no frame decodes. This is almost always AV1, "
            "which OpenCV cannot read. Re-encode to H.264 and upload again "
            '(yt-dlp: -f "bv*[vcodec^=avc1]").'
        )
    if "CUDA out of memory" in tail:
        return "The GPU ran out of memory. Try a shorter clip."
    if "could not open video" in tail:
        return "The file could not be opened as a video."
    if "FileNotFoundError" in tail and ".pt" in tail:
        return "A model weight file is missing on the server."
    lines = [line for line in tail.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "The run failed without producing any output."


def run_job(job: Job) -> None:
    video = job.video_path()
    if video is None:
        job.status, job.error = "failed", "The uploaded file is missing."
        job.finished = time.time()
        job.save()
        return

    job.status = "running"
    job.started = time.time()
    job.save()

    log = (job.dir / "run.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            build_command(job, video),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            probe = PROBE_RE.match(line)
            if probe:
                job.frames_total = int(probe.group(1))
                job.duration_s = float(probe.group(2))
                job.save()
                continue
            progress = PROGRESS_RE.match(line)
            if progress:
                job.frames_done = int(progress.group(1))
                job.fps = float(progress.group(2))
                job.save()
        code = process.wait()
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        job.status, job.error = "failed", str(exc)
        job.finished = time.time()
        job.save()
        return
    finally:
        log.close()

    if code != 0:
        job.status, job.error = "failed", friendly_error(job)
    else:
        job.status = "done"
        job.frames_done = job.frames_total or job.frames_done
    job.finished = time.time()
    job.save()


class JobStore:
    """Job registry, mirrored to disk so a restart keeps history."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.queue: Queue[str] = Queue()
        self._load()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load(self) -> None:
        for meta in sorted(JOBS_DIR.glob("*/job.json")):
            try:
                job = Job(**json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            # A job mid-flight when the server stopped did not survive it.
            # Saying so beats a progress bar frozen at 40%.
            if job.status in {"queued", "running"}:
                job.status = "failed"
                job.error = "The server restarted while this job was running."
            self._jobs[job.id] = job

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.get(job_id)
            if job is not None:
                try:
                    run_job(job)
                except Exception as exc:  # noqa: BLE001 - never kill the worker
                    job.status, job.error = "failed", str(exc)
                    job.finished = time.time()
                    job.save()
            self.queue.task_done()

    def submit(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
        job.save()
        self.queue.put(job.id)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def remove(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        return True
