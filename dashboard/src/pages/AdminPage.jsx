import { useState, useEffect, useCallback } from "react";
import { Card, ErrorBox, Spinner } from "../components/ui";
import {
  getStoredPassword, storePassword, clearPassword,
  verifyPassword, fetchAdminStats, fetchTopPredictions,
} from "../lib/admin";

const GREEN = "#2F7D4F";
const BLUE = "#185FA5";
const fmt = (v, d = 2) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—");

const th = { textAlign: "left", fontSize: 11, fontWeight: 600, color: "var(--color-text-tertiary, #6b7280)", padding: "6px 10px", textTransform: "uppercase", letterSpacing: "0.05em", whiteSpace: "nowrap" };
const td = { fontSize: 13, padding: "7px 10px", borderTop: "1px solid var(--color-border-secondary, #2e303a)", whiteSpace: "nowrap" };

function SportBadge({ sport }) {
  const c = sport === "NFL" ? GREEN : BLUE;
  return <span style={{ fontSize: 10, fontWeight: 700, color: "#fff", background: c, borderRadius: 4, padding: "2px 6px" }}>{sport}</span>;
}

function Tile({ label, value }) {
  return (
    <div style={{ padding: "10px 14px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: 10, minWidth: 110 }}>
      <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)" }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600, marginTop: 2 }}>{value}</div>
    </div>
  );
}

function ScrollTable({ children }) {
  return <div style={{ overflowX: "auto", marginTop: 10 }}><table style={{ borderCollapse: "collapse", width: "100%", minWidth: 640 }}>{children}</table></div>;
}

function TopPredictions({ data }) {
  const preds = data?.predictions || [];
  const maxScore = Math.max(1e-6, ...preds.map((p) => p.confidence_score || 0));
  return (
    <Card style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 15, fontWeight: 600 }}>Top 10 predictions — confidence-weighted edge</div>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary, #9ca3af)", marginTop: 3 }}>
        {data?.note || "Ranked by reliability × standardized edge vs the player's recent baseline."}
        {data?.as_of_date ? ` · as of ${data.as_of_date}` : ""}
      </div>
      {preds.length === 0 ? (
        <div style={{ marginTop: 12, fontSize: 13, color: "var(--color-text-tertiary, #6b7280)" }}>
          No picks yet — this fills in once players have upcoming games (and after the
          weekly pipeline runs <code>analytics.top_predictions</code>).
        </div>
      ) : (
        <ScrollTable>
          <thead><tr>
            {["#", "Sport", "Player", "Team", "Stat", "Proj", "Baseline", "Edge", "R²", "Score", "Matchup"].map((h) => <th key={h} style={th}>{h}</th>)}
          </tr></thead>
          <tbody>
            {preds.map((p, i) => (
              <tr key={i}>
                <td style={td}>{i + 1}</td>
                <td style={td}><SportBadge sport={p.sport} /></td>
                <td style={{ ...td, fontWeight: 500 }}>{p.player}{p.team_changed ? " ↔" : ""}</td>
                <td style={td}>{p.team}</td>
                <td style={td}>{p.stat}</td>
                <td style={td}>{fmt(p.projection, 1)}</td>
                <td style={{ ...td, color: "var(--color-text-tertiary, #6b7280)" }}>{fmt(p.baseline, 1)}</td>
                <td style={{ ...td, color: p.direction === "over" ? GREEN : "#993C1D" }}>{p.edge > 0 ? "+" : ""}{fmt(p.edge, 1)}</td>
                <td style={td}>{fmt(p.reliability, 2)}</td>
                <td style={td}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 44, height: 5, background: "var(--color-background-tertiary, #2a2b33)", borderRadius: 3 }}>
                      <div style={{ width: `${Math.min(100, (p.confidence_score / maxScore) * 100)}%`, height: "100%", background: GREEN, borderRadius: 3 }} />
                    </div>
                    {fmt(p.confidence_score, 2)}
                  </div>
                </td>
                <td style={{ ...td, color: "var(--color-text-tertiary, #6b7280)" }}>{p.game || ""}</td>
              </tr>
            ))}
          </tbody>
        </ScrollTable>
      )}
    </Card>
  );
}

function SportPanel({ s }) {
  if (!s) return null;
  const targets = s.targets || [];
  return (
    <Card style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <div style={{ fontSize: 15, fontWeight: 600 }}>{s.sport} model performance</div>
        {s.mode && <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 6, background: "rgba(255,255,255,0.06)", color: "var(--color-text-secondary, #9ca3af)" }}>{s.mode}</span>}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
        {s.seasons && <Tile label="Seasons" value={`${s.seasons[0]}–${s.seasons[s.seasons.length - 1]}`} />}
        {s.n_rows != null && <Tile label="Train rows" value={s.n_rows.toLocaleString()} />}
        {s.n_players != null && <Tile label="Players" value={s.n_players.toLocaleString()} />}
        {s.n_features != null && <Tile label="Features" value={s.n_features} />}
        <Tile label="Targets" value={targets.length} />
      </div>
      {s.updated_at && <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", marginTop: 8 }}>Updated {String(s.updated_at).slice(0, 10)}</div>}
      {s.failed_baselines?.length > 0 && (
        <div style={{ marginTop: 10, fontSize: 12, color: "#8a5a0b", background: "#FAEEDA", borderRadius: 8, padding: "8px 12px" }}>
          Fails baseline ({s.failed_baselines.length}): {s.failed_baselines.join(", ")}
        </div>
      )}
      <ScrollTable>
        <thead><tr>
          {["Pos", "Target", "MAE", "Baseline", "Impr", "R²", "Coverage", "Eval n"].map((h) => <th key={h} style={th}>{h}</th>)}
        </tr></thead>
        <tbody>
          {targets.map((t, i) => (
            <tr key={i}>
              <td style={td}>{t.position}</td>
              <td style={{ ...td, fontWeight: 500 }}>{t.target}</td>
              <td style={td}>{fmt(t.mae)}</td>
              <td style={{ ...td, color: "var(--color-text-tertiary, #6b7280)" }}>{fmt(t.baseline_mae)}</td>
              <td style={{ ...td, color: t.improvement > 0 ? GREEN : "#993C1D" }}>{t.improvement != null ? `${(t.improvement * 100).toFixed(0)}%` : "—"}</td>
              <td style={td}>{fmt(t.r2)}</td>
              <td style={td}>{t.coverage != null ? `${(t.coverage * 100).toFixed(0)}%` : "—"}</td>
              <td style={{ ...td, color: "var(--color-text-tertiary, #6b7280)" }}>{t.n_eval ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </ScrollTable>
    </Card>
  );
}

export default function AdminPage() {
  const [pw, setPw] = useState("");
  const [authed, setAuthed] = useState(false);
  const [stats, setStats] = useState(null);
  const [top, setTop] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async (password) => {
    setLoading(true);
    setError(null);
    try {
      await verifyPassword(password);
      storePassword(password);
      const [s, t] = await Promise.all([fetchAdminStats(password), fetchTopPredictions(password)]);
      setStats(s);
      setTop(t);
      setAuthed(true);
    } catch (e) {
      setAuthed(false);
      if (e.status === 401) setError("Incorrect password.");
      else if (e.status === 503) setError("Admin dashboard isn't configured on the server (ADMIN_PASSWORD not set).");
      else setError(e.message);
      clearPassword();
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const stored = getStoredPassword();
    if (stored) load(stored);
  }, [load]);

  if (!authed) {
    return (
      <div style={{ maxWidth: 420, margin: "3rem auto" }}>
        <Card>
          <h1 style={{ fontSize: 20, fontWeight: 600, margin: "0 0 4px" }}>Admin dashboard</h1>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)", marginTop: 0 }}>
            Enter the admin password to view model performance.
          </p>
          <form onSubmit={(e) => { e.preventDefault(); if (pw) load(pw); }}>
            <input
              type="password"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
              placeholder="Password"
              autoFocus
              style={{ width: "100%", boxSizing: "border-box", padding: "10px 12px", borderRadius: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--color-text-primary, #fff)", fontSize: 14 }}
            />
            <button type="submit" disabled={loading || !pw} style={{ width: "100%", marginTop: 10, padding: "11px 0", borderRadius: 10, border: "none", background: GREEN, color: "#fff", fontSize: 14, fontWeight: 600, cursor: loading || !pw ? "not-allowed" : "pointer", opacity: loading || !pw ? 0.6 : 1 }}>
              {loading ? "Checking…" : "Sign in"}
            </button>
          </form>
          {error && <div style={{ marginTop: 12 }}><ErrorBox>{error}</ErrorBox></div>}
        </Card>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem" }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0 }}>Admin dashboard</h1>
        <button onClick={() => { clearPassword(); setAuthed(false); setPw(""); }} style={{ fontSize: 12, padding: "6px 12px", borderRadius: 8, background: "rgba(255,255,255,0.04)", border: "1px solid var(--color-border-secondary, #2e303a)", color: "var(--color-text-secondary, #9ca3af)", cursor: "pointer" }}>Sign out</button>
      </div>
      {loading && <Spinner label="Loading model stats…" />}
      <TopPredictions data={top} />
      <SportPanel s={stats?.nfl} />
      <SportPanel s={stats?.nba} />
    </div>
  );
}
