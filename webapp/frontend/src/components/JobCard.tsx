import { api, formatDuration, type Job } from "../api";

interface Props {
  job: Job;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
}

const LABELS: Record<Job["status"], string> = {
  queued: "queued",
  running: "running",
  done: "done",
  failed: "failed",
};

export function JobCard({ job, selected, onSelect, onDelete }: Props) {
  const active = job.status === "running";
  return (
    <li className={`job${selected ? " selected" : ""}`}>
      <button className="job-main" onClick={onSelect}>
        <div className="job-head">
          <span className="job-name" title={job.filename}>
            {job.filename}
          </span>
          <span className={`status ${job.status}`}>{LABELS[job.status]}</span>
        </div>

        {active && (
          <>
            <div className="bar">
              <div style={{ width: `${job.progress * 100}%` }} />
            </div>
            <div className="job-meta">
              <span>
                {job.frames_done}
                {job.frames_total ? ` / ${job.frames_total}` : ""} frames
              </span>
              <span>{job.fps ? `${job.fps.toFixed(1)} fps` : ""}</span>
              <span>
                {job.eta_seconds != null
                  ? `${formatDuration(job.eta_seconds)} left`
                  : "estimating…"}
              </span>
            </div>
          </>
        )}

        {job.status === "done" && (
          <div className="job-meta">
            <span>{job.frames_total ?? job.frames_done} frames</span>
            <span>{job.fps ? `${job.fps.toFixed(1)} fps` : ""}</span>
            <span>took {formatDuration(job.elapsed_seconds)}</span>
          </div>
        )}

        {job.status === "failed" && <p className="error small">{job.error}</p>}
        {job.status === "queued" && (
          <div className="job-meta">
            <span>waiting for the GPU</span>
          </div>
        )}
      </button>

      {job.status !== "running" && job.status !== "queued" && (
        <button
          className="ghost delete"
          title="delete this job and its files"
          onClick={onDelete}
        >
          ×
        </button>
      )}
    </li>
  );
}

export function JobLinks({ job }: { job: Job }) {
  if (job.status !== "done") return null;
  return (
    <div className="links">
      <a href={api.reportUrl(job.id)} target="_blank" rel="noreferrer">
        Open report in a new tab
      </a>
      <a href={api.reportJsonUrl(job.id)} target="_blank" rel="noreferrer">
        report.json
      </a>
      {job.has_video && (
        <a href={api.videoUrl(job.id)} download>
          Download annotated video
        </a>
      )}
    </div>
  );
}
