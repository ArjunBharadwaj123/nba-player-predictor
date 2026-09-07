// Shared presentational atoms used by both the NBA and NFL pages. Kept
// style-driven (colors/labels passed in) so the two sports share one
// implementation without one dictating the other's palette.

export const Card = ({ children, style = {} }) => (
  <div
    style={{
      background: "var(--color-background-primary, #1b1c22)",
      border: "1px solid var(--color-border-secondary, #2e303a)",
      borderRadius: "12px",
      padding: "1rem",
      boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
      ...style,
    }}
  >
    {children}
  </div>
);

export const PosBadge = ({ pos, colors = {}, textColors = {} }) => (
  <span
    style={{
      fontSize: 11,
      fontWeight: 500,
      padding: "2px 7px",
      borderRadius: 4,
      background: colors[pos] || "#F1EFE8",
      color: textColors[pos] || "#2C2C2A",
    }}
  >
    {pos}
  </span>
);

export const StatBar = ({ label, value, range, max, color, unit }) => {
  const [lo, hi] = range || [value, value];
  const pct = (v) => `${Math.min((v / max) * 100, 100).toFixed(1)}%`;
  return (
    <div style={{ marginBottom: "1.4rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)" }}>
          {label}
        </span>
        <div>
          <span style={{ fontSize: 14, fontWeight: 500 }}>{value}</span>
          {unit ? (
            <span style={{ fontSize: 10, color: "var(--color-text-tertiary, #6b7280)", marginLeft: 3 }}>
              {unit}
            </span>
          ) : null}
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", marginLeft: 6 }}>
            {lo}–{hi}
          </span>
        </div>
      </div>
      <div
        style={{
          position: "relative",
          height: 6,
          borderRadius: 3,
          background: "var(--color-background-tertiary, #2a2b33)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: pct(lo),
            width: `${Math.min(((hi - lo) / max) * 100, 100).toFixed(1)}%`,
            height: "100%",
            background: color,
            opacity: 0.2,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: 0,
            width: pct(value),
            height: "100%",
            background: color,
            borderRadius: 3,
            transition: "width 0.5s ease",
          }}
        />
      </div>
    </div>
  );
};

export const ProbGauge = ({ prob, caption, accent }) => {
  const pct = Math.round(prob * 100);
  const color = accent || (pct >= 65 ? "#0F6E56" : pct >= 35 ? "#854F0B" : "#993C1D");
  return (
    <div style={{ textAlign: "center", padding: "0.75rem 0" }}>
      <div style={{ fontSize: 48, fontWeight: 500, color, lineHeight: 1 }}>{pct}%</div>
      <div style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)", margin: "4px 0 14px" }}>
        {caption}
      </div>
      <div style={{ position: "relative", height: 8, borderRadius: 4, background: "var(--color-background-tertiary, #2a2b33)" }}>
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.6s ease" }} />
        <div style={{ position: "absolute", left: "50%", top: -2, width: 1.5, height: 12, background: "var(--color-border-primary, #3a3c48)", borderRadius: 1 }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 11, color: "var(--color-text-tertiary, #6b7280)" }}>
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  );
};

export const Spinner = ({ label }) => (
  <div style={{ padding: "11px 0", textAlign: "center", fontSize: 14, color: "var(--color-text-secondary, #9ca3af)" }}>
    {label || "Loading…"}
  </div>
);

export const ErrorBox = ({ children, onRetry }) => (
  <div
    style={{
      padding: "10px 14px",
      borderRadius: 8,
      background: "#FCEBEB",
      color: "#791F1F",
      fontSize: 13,
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      gap: 10,
    }}
  >
    <span>{children}</span>
    {onRetry ? (
      <button
        onClick={onRetry}
        style={{
          fontSize: 12,
          padding: "5px 10px",
          borderRadius: 8,
          border: "1px solid #791F1F33",
          background: "transparent",
          color: "#791F1F",
          cursor: "pointer",
          flexShrink: 0,
        }}
      >
        Retry
      </button>
    ) : null}
  </div>
);

export const EmptyState = ({ icon = "📅", children }) => (
  <div
    style={{
      padding: "14px 16px",
      borderRadius: 8,
      background: "var(--color-background-secondary, #202128)",
      border: "1px solid var(--color-border-secondary, #2e303a)",
      color: "var(--color-text-primary, #f3f4f6)",
      fontSize: 13,
      lineHeight: 1.5,
      display: "flex",
      gap: 10,
      alignItems: "flex-start",
    }}
  >
    <span style={{ fontSize: 18, lineHeight: 1 }}>{icon}</span>
    <span>{children}</span>
  </div>
);
