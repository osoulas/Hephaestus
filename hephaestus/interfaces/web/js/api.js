// HTTP API client (docs/INTERFACES.md). Every call returns parsed JSON or throws ApiError.

export class ApiError extends Error {
  /**
   * @param {number} status HTTP status.
   * @param {string} message The server's `detail`.
   */
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * Call the API.
 * @param {string} path Path, e.g. "/runs".
 * @param {object} [opts] `method`, `body` (JSON), `text` (return text instead of JSON).
 * @returns {Promise<any>}
 */
export async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new ApiError(0, "The server is not reachable.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch (e) {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return opts.text ? res.text() : res.json();
}

/**
 * Stream a run's events (server-sent events). History arrives first, then live events.
 * @param {string} runId Run or replay ID.
 * @param {(event: object) => void} onEvent Called per ProgressEvent.
 * @param {() => void} onEnd Called once the run has ended.
 * @returns {() => void} Call to close the stream.
 */
export function streamEvents(runId, onEvent, onEnd) {
  const es = new EventSource(`/runs/${encodeURIComponent(runId)}/events`);
  es.addEventListener("progress", (m) => {
    try {
      onEvent(JSON.parse(m.data));
    } catch (e) {
      console.error("bad event", e);
    }
  });
  es.addEventListener("end", () => {
    es.close(); // otherwise the browser reconnects and the history replays again
    onEnd();
  });
  return () => es.close();
}
