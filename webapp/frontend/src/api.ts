// Typed client for the FastAPI backend.
//
// In development Vite serves this app on :5173 and proxies /api and /jobs to
// the server on :8000. In production the server serves the built bundle
// itself, so a relative base works in both cases.

export type JobStatus = "queued" | "running" | "done" | "failed";

export interface Job {
  id: string;
  filename: string;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  frames_total: number | null;
  frames_done: number;
  fps: number | null;
  duration_s: number | null;
  error: string | null;
  options: { limit?: number | null; no_video?: boolean };
  progress: number;
  eta_seconds: number | null;
  elapsed_seconds: number | null;
  has_report: boolean;
  has_video: boolean;
}

export interface Health {
  ok: boolean;
  cuda: boolean;
  gpu: string | null;
  queued: number;
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* the body was not JSON; the status text is the best we have */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/api/health").then(json<Health>),

  listJobs: () =>
    fetch("/api/jobs").then(json<{ jobs: Job[]; queued: number }>),

  getJob: (id: string) => fetch(`/api/jobs/${id}`).then(json<Job>),

  deleteJob: (id: string) =>
    fetch(`/api/jobs/${id}`, { method: "DELETE" }).then(json<{ deleted: string }>),

  log: (id: string) => fetch(`/api/jobs/${id}/log`).then((r) => r.text()),

  reportUrl: (id: string) => `/jobs/${id}/report`,
  videoUrl: (id: string) => `/jobs/${id}/video`,
  reportJsonUrl: (id: string) => `/api/jobs/${id}/report.json`,

  // XMLHttpRequest rather than fetch: uploads are the slow half of this on a
  // home connection, and fetch still cannot report upload progress.
  upload(
    file: File,
    options: { limit?: number | null; noVideo?: boolean },
    onProgress: (fraction: number) => void,
  ): Promise<Job> {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("video", file);
      if (options.limit) form.append("limit", String(options.limit));
      form.append("no_video", options.noVideo ? "true" : "false");

      const request = new XMLHttpRequest();
      request.open("POST", "/api/jobs");
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) onProgress(event.loaded / event.total);
      });
      request.addEventListener("load", () => {
        if (request.status >= 200 && request.status < 300) {
          resolve(JSON.parse(request.responseText) as Job);
        } else {
          let detail = `upload failed (${request.status})`;
          try {
            detail = JSON.parse(request.responseText).detail ?? detail;
          } catch {
            /* keep the status-code message */
          }
          reject(new Error(detail));
        }
      });
      request.addEventListener("error", () =>
        reject(new Error("the connection dropped during upload")),
      );
      request.addEventListener("abort", () =>
        reject(new Error("the upload was cancelled")),
      );
      request.send(form);
    });
  },
};

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "—";
  const total = Math.round(seconds);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${total % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
