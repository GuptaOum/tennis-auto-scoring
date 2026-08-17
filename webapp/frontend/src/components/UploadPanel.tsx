import { useRef, useState } from "react";
import { api, formatBytes, type Job } from "../api";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.m4v,.webm";

interface Props {
  onQueued: (job: Job) => void;
  busy: boolean;
}

export function UploadPanel({ onQueued, busy }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [quickTest, setQuickTest] = useState(false);
  const [noVideo, setNoVideo] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function choose(selected: File | null) {
    setError(null);
    setFile(selected);
  }

  async function submit() {
    if (!file) return;
    setUploading(true);
    setUploadProgress(0);
    setError(null);
    try {
      const job = await api.upload(
        file,
        { limit: quickTest ? 300 : null, noVideo },
        setUploadProgress,
      );
      onQueued(job);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="panel">
      <h2>Analyse a clip</h2>

      <div
        className={`dropzone${dragging ? " dragging" : ""}${file ? " filled" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          choose(event.dataTransfer.files[0] ?? null);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(event) => choose(event.target.files?.[0] ?? null)}
        />
        {file ? (
          <>
            <strong>{file.name}</strong>
            <span className="muted">{formatBytes(file.size)}</span>
          </>
        ) : (
          <>
            <strong>Drop a match clip here</strong>
            <span className="muted">or click to choose — mp4, mov, mkv, avi</span>
          </>
        )}
      </div>

      <div className="options">
        <label>
          <input
            type="checkbox"
            checked={quickTest}
            onChange={(event) => setQuickTest(event.target.checked)}
          />
          Quick test — first 300 frames only
          <span className="hint">
            ~10 seconds of footage. Use it to confirm the clip decodes and the
            court is found before committing to the full run.
          </span>
        </label>
        <label>
          <input
            type="checkbox"
            checked={noVideo}
            onChange={(event) => setNoVideo(event.target.checked)}
          />
          Skip the annotated video
          <span className="hint">
            Roughly a third faster, and the report is unaffected. Turn it off
            only if you want to watch the overlay.
          </span>
        </label>
      </div>

      {uploading && (
        <div className="upload-progress">
          <div className="bar">
            <div style={{ width: `${uploadProgress * 100}%` }} />
          </div>
          <span>uploading {(uploadProgress * 100).toFixed(0)}%</span>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      <button
        className="primary"
        disabled={!file || uploading}
        onClick={submit}
      >
        {uploading ? "Uploading…" : busy ? "Queue it" : "Analyse"}
      </button>
      {busy && !uploading && (
        <p className="muted small">
          A job is already running on the GPU. This one starts when that
          finishes — they run one at a time so the progress figures stay honest.
        </p>
      )}
    </section>
  );
}
