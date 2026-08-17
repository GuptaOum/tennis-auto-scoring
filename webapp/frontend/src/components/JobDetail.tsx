import { useEffect, useState } from "react";
import { api, formatDuration, type Job } from "../api";
import { JobLinks } from "./JobCard";

/** The finished report, the annotated video, or an honest account of why
 *  neither exists yet. */
export function JobDetail({ job }: { job: Job }) {
  const [tab, setTab] = useState<"report" | "video" | "log">("report");
  const [log, setLog] = useState("");

  // The log is the only thing worth polling here: the report is a static
  // file once it exists, and re-fetching an iframe would fight the scroll
  // position.
  useEffect(() => {
    if (tab !== "log" && job.status !== "failed") return;
    let cancelled = false;
    const load = () =>
      api
        .log(job.id)
        .then((text) => !cancelled && setLog(text))
        .catch(() => undefined);
    load();
    const timer = setInterval(load, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job.id, job.status, tab]);

  if (job.status === "queued") {
    return (
      <section className="panel detail">
        <h2>{job.filename}</h2>
        <p className="muted">
          Queued. Jobs run one at a time — there is a single GPU, and running
          two would make both of them slower and both progress bars wrong.
        </p>
      </section>
    );
  }

  if (job.status === "running") {
    return (
      <section className="panel detail">
        <h2>{job.filename}</h2>
        <div className="bar big">
          <div style={{ width: `${job.progress * 100}%` }} />
        </div>
        <div className="stat-row">
          <div>
            <strong>{(job.progress * 100).toFixed(0)}%</strong>
            <span>complete</span>
          </div>
          <div>
            <strong>{job.fps ? job.fps.toFixed(1) : "—"}</strong>
            <span>frames per second</span>
          </div>
          <div>
            <strong>{formatDuration(job.eta_seconds)}</strong>
            <span>remaining</span>
          </div>
          <div>
            <strong>{formatDuration(job.elapsed_seconds)}</strong>
            <span>elapsed</span>
          </div>
        </div>
        <p className="muted small">
          {job.frames_done}
          {job.frames_total ? ` of ${job.frames_total}` : ""} frames processed.
          The remaining time is measured from this run's own throughput, not a
          nominal figure — it settles after the first few hundred frames.
        </p>
        <details>
          <summary>Show the log</summary>
          <pre className="log">{log || "…"}</pre>
        </details>
      </section>
    );
  }

  if (job.status === "failed") {
    return (
      <section className="panel detail">
        <h2>{job.filename}</h2>
        <p className="error">{job.error}</p>
        <pre className="log">{log || "…"}</pre>
      </section>
    );
  }

  return (
    <section className="panel detail">
      <div className="detail-head">
        <h2>{job.filename}</h2>
        <div className="tabs">
          <button
            className={tab === "report" ? "active" : ""}
            onClick={() => setTab("report")}
          >
            Report
          </button>
          {job.has_video && (
            <button
              className={tab === "video" ? "active" : ""}
              onClick={() => setTab("video")}
            >
              Annotated video
            </button>
          )}
          <button
            className={tab === "log" ? "active" : ""}
            onClick={() => setTab("log")}
          >
            Log
          </button>
        </div>
      </div>

      <JobLinks job={job} />

      {tab === "report" &&
        (job.has_report ? (
          <iframe
            className="report-frame"
            src={api.reportUrl(job.id)}
            title={`match report for ${job.filename}`}
          />
        ) : (
          <p className="muted">This run produced no report.</p>
        ))}

      {tab === "video" && job.has_video && (
        <video className="annotated" controls src={api.videoUrl(job.id)} />
      )}

      {tab === "log" && <pre className="log">{log || "…"}</pre>}
    </section>
  );
}
