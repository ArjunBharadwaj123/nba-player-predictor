// Backend base URL. Set VITE_API_URL (Vercel env var) to override. When unset,
// a production build points at the deployed API so the hosted app works without
// extra config, while local dev falls back to the local server. Shared by the
// NBA and NFL pages.
export const API =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? "https://nba-player-predictor-api.onrender.com"
    : "http://localhost:8000");

// Thin fetch helper that throws a useful message on non-2xx responses.
export async function apiFetch(path, options) {
  const resp = await fetch(`${API}${path}`, options);
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(detail);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}
