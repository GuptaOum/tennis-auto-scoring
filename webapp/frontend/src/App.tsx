import { useCallback, useEffect, useState } from "react";
import { api, type Health, type Job } from "./api";
import { JobCard } from "./components/JobCard";
import { JobDetail } from "./components/JobDetail";
import { UploadPanel } from "./components/UploadPanel";

// Polled rather than pushed. A websocket would be tidier, but the payload is
// a handful of job records and the poll only runs while something is active -
// which is most of the complexity of a socket for none of its benefit.
const ACTIVE_POLL_MS = 1500;
const IDLE_POLL_MS = 15000;

export default function App() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listJobs();
      setJobs(data.jobs);
      setOffline(false);
      setSelectedId((current) =>
        current && data.jobs.some((job) => job.id === current)
          ? current
          : (data.jobs[0]?.id ?? null),
      );
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    api.health().then(setHealth).catch(() => setHealth(null));
  }, [refresh]);

  const busy = jobs.some(
    (job) => job.status === "running" || job.status === "queued",
  );

  useEffect(() => {
    const timer = setInterval(refresh, busy ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    return () => clearInterval(timer);
  }, [busy, refresh]);

  const selected = jobs.find((job) => job.id === selectedId) ?? null;

  async function remove(id: string) {
    await api.deleteJob(id).catch(() => undefined);
    refresh();
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Tennis auto-scoring</h1>
          <p className="muted">
            Upload a match clip; get a scored, annotated report back.
          </p>
        </div>
        <div className="health">
          {offline ? (
            <span className="chip bad">server unreachable</span>
          ) : health ? (
            <span className={`chip ${health.cuda ? "good" : "warn"}`}>
              {health.cuda ? health.gpu : "CPU only — this will be slow"}
            </span>
          ) : (
            <span className="chip">checking…</span>
          )}
        </div>
      </header>

      <div className="layout">
        <div className="column">
          <UploadPanel
            busy={busy}
            onQueued={(job) => {
              setJobs((current) => [job, ...current]);
              setSelectedId(job.id);
            }}
          />

          <section className="panel">
            <h2>Runs</h2>
            {jobs.length === 0 ? (
              <p className="muted">Nothing analysed yet.</p>
            ) : (
              <ul className="jobs">
                {jobs.map((job) => (
                  <JobCard
                    key={job.id}
                    job={job}
                    selected={job.id === selectedId}
                    onSelect={() => setSelectedId(job.id)}
                    onDelete={() => remove(job.id)}
                  />
                ))}
              </ul>
            )}
          </section>
        </div>

        <div className="column wide">
          {selected ? (
            <JobDetail job={selected} />
          ) : (
            <section className="panel detail empty">
              <p className="muted">
                Upload a clip to get started. A three-minute clip takes roughly
                ten minutes on the GPU.
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
