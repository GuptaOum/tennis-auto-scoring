"""FastAPI server: upload a video, get a match report.

This runs on the GPU box and is reached through an SSH tunnel. Both are
deliberate:

- **It runs where the GPU is.** The video has to reach the T4 either way, but
  serving results from the same machine means the annotated video and the
  report never travel back. Only the upload crosses the link.
- **Nothing is exposed publicly.** There is no authentication here, and an
  open upload endpoint that shells out to a subprocess has no business facing
  the internet. The tunnel is the access control - the default bind is
  127.0.0.1 for that reason. Do not move it to 0.0.0.0 without putting a real
  login in front of it first.

    python -m webapp.server                    # on the GPU box
    ssh -i key.pem -L 8000:localhost:8000 ...  # from your machine

The React frontend is built to ``webapp/frontend/dist`` and served from here,
so the server needs no node runtime.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from webapp.jobs import ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, Job, JobStore

FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"

app = FastAPI(title="Tennis auto-scoring", docs_url="/api/docs")

# The Vite dev server runs on a different origin during development. In
# production the frontend is served from this same app and no CORS is needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = JobStore()


def _safe_name(filename: str) -> str:
    """Keep only the basename, and only characters that cannot escape a path."""
    stem = Path(filename).name
    cleaned = "".join(c for c in stem if c.isalnum() or c in "._- ").strip()
    return cleaned or "upload.mp4"


def _get(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job


@app.get("/api/health")
def health() -> dict:
    import torch

    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "queued": store.queue.qsize(),
    }


@app.post("/api/jobs", status_code=201)
async def create_job(
    video: UploadFile = File(...),
    limit: int | None = Form(None),
    no_video: bool = Form(False),
) -> dict:
    if not video.filename:
        raise HTTPException(status_code=400, detail="no file was uploaded")

    name = _safe_name(video.filename)
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type - expected one of "
                   f"{', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    job = Job(id=uuid.uuid4().hex[:12], filename=name)
    job.options = {"limit": limit, "no_video": no_video}
    job.dir.mkdir(parents=True, exist_ok=True)

    target = job.dir / name
    written = 0
    try:
        with target.open("wb") as handle:
            while chunk := await video.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds the "
                               f"{MAX_UPLOAD_BYTES // 1024**3} GB limit",
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise

    if written == 0:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="the uploaded file was empty")

    store.submit(job)
    return job.as_dict()


@app.get("/api/jobs")
def list_jobs() -> dict:
    return {
        "jobs": [job.as_dict() for job in store.all()],
        "queued": store.queue.qsize(),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _get(job_id).as_dict()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = _get(job_id)
    if job.status in {"queued", "running"}:
        raise HTTPException(
            status_code=409, detail="cannot delete a job that is still running"
        )
    store.remove(job_id)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str) -> str:
    log = _get(job_id).dir / "run.log"
    if not log.exists():
        return ""
    return log.read_text(encoding="utf-8", errors="replace")[-20000:]


@app.get("/api/jobs/{job_id}/report.json")
def job_report_json(job_id: str) -> JSONResponse:
    path = _get(job_id).output_dir / "report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no report yet")
    return FileResponse(path, media_type="application/json")


@app.get("/jobs/{job_id}/report")
def job_report(job_id: str) -> FileResponse:
    path = _get(job_id).output_dir / "report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no report yet")
    return FileResponse(path, media_type="text/html")


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    path = _get(job_id).output_dir / "annotated.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no annotated video")
    return FileResponse(path, media_type="video/mp4", filename="annotated.mp4")


# Mounted last so it never shadows an API route. Vite emits a hashed asset
# bundle plus index.html; html=True makes unknown paths fall back to it.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Tennis auto-scoring web server")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address - localhost only by default, reach it over an SSH "
             "tunnel rather than a public port",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
