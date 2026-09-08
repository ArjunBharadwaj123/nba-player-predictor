import { API } from "./api";

// Admin API helpers. The shared password is sent in the X-Admin-Password header
// and held only in sessionStorage (per-tab, cleared on close) — never baked in.
const KEY = "nfl_admin_password";

export const getStoredPassword = () => {
  try {
    return window.sessionStorage.getItem(KEY) || "";
  } catch {
    return "";
  }
};

export const storePassword = (pw) => {
  try {
    window.sessionStorage.setItem(KEY, pw);
  } catch {
    /* sessionStorage unavailable */
  }
};

export const clearPassword = () => {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
};

async function adminFetch(path, password) {
  const resp = await fetch(`${API}${path}`, {
    headers: { "X-Admin-Password": password },
  });
  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`;
    try {
      detail = (await resp.json()).detail || detail;
    } catch {
      /* non-JSON */
    }
    const err = new Error(detail);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

export const verifyPassword = (pw) => adminFetch("/admin/verify", pw);
export const fetchAdminStats = (pw) => adminFetch("/admin/stats", pw);
export const fetchTopPredictions = (pw) => adminFetch("/admin/top-predictions", pw);
