import { useState, useCallback, useRef, useEffect } from "react";

// Backend base URL. In production set VITE_API_URL in Vercel's env vars to the
// deployed API (e.g. https://nba-player-predictor-api.onrender.com).
// Falls back to the local dev server when unset.
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const STAT_LABELS = { pts:"Points", reb:"Rebounds", ast:"Assists", stl:"Steals", blk:"Blocks", minutes:"Minutes" };
const STAT_COLORS = { pts:"#185FA5", reb:"#0F6E56", ast:"#534AB7", stl:"#993C1D", blk:"#854F0B", minutes:"#5F5E5A" };
const STAT_MAX    = { pts:55, reb:22, ast:16, stl:5, blk:5, minutes:48 };

const POS_COLORS = { PG:"#E6F1FB", SG:"#EAF3DE", SF:"#FAEEDA", PF:"#FAECE7", C:"#EEEDFE" };
const POS_TEXT   = { PG:"#0C447C", SG:"#27500A", SF:"#633806", PF:"#712B13", C:"#3C3489" };

// ── Atoms ──────────────────────────────────────────────────────────────────────

const Card = ({ children, style = {} }) => (
  <div style={{
    background: "var(--color-background-primary)",
    border: "1px solid var(--color-border-secondary)",
    borderRadius: "12px",
    padding: "1rem",
    boxShadow: "0 4px 20px rgba(0,0,0,0.25)",
    ...style,
  }}>{children}</div>
);

const PosBadge = ({ pos }) => (
  <span style={{
    fontSize: 11, fontWeight: 500, padding: "2px 7px",
    borderRadius: 4,
    background: POS_COLORS[pos] || "#F1EFE8",
    color: POS_TEXT[pos] || "#2C2C2A",
  }}>{pos}</span>
);

const StatBar = ({ label, value, range, max, color }) => {
  const [lo, hi] = range || [value, value];
  const pct = v => `${Math.min((v / max) * 100, 100).toFixed(1)}%`;
  return (
    <div style={{ marginBottom: "1.5rem", textAlign: "center" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{label}</span>
        <div>
          <span style={{ fontSize: 14, fontWeight: 500 }}>{value}</span>
          <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginLeft: 6 }}>{lo}–{hi}</span>
        </div>
      </div>
      <div style={{ position: "relative", height: 6, borderRadius: 3, background: "var(--color-background-tertiary)", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: pct(lo), width: `${Math.min(((hi - lo) / max) * 100, 100).toFixed(1)}%`, height: "100%", background: color, opacity: 0.2 }} />
        <div style={{ position: "absolute", left: 0, width: pct(value), height: "100%", background: color, borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
    </div>
  );
};

const ReasonLine = ({ line }) => {
  if (!line.trim() || line === "Reasoning:") return null;
  const pos = line.startsWith("+"), neg = line.startsWith("-");
  if (!pos && !neg) return null;
  return (
    <div style={{ display: "flex", gap: 8, padding: "8px 10px", background: "rgba(255,255,255,0.02)", borderRadius: "8px", borderBottom: "0.5px solid var(--color-border-tertiary)", alignItems: "flex-start" }}>
      <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 6px", borderRadius: 4, flexShrink: 0, marginTop: 1,
        background: pos ? "#E1F5EE" : "#FCEBEB", color: pos ? "#085041" : "#791F1F" }}>
        {pos ? "+" : "−"}
      </span>
      <span style={{ fontSize: 13, lineHeight: 1.5 }}>{line.slice(2).trim()}</span>
    </div>
  );
};

const ProbGauge = ({ prob, direction, threshold, stat }) => {
  const pct   = Math.round(prob * 100);
  const color = pct >= 65 ? "#0F6E56" : pct >= 35 ? "#854F0B" : "#993C1D";
  return (
    <div style={{ textAlign: "center", padding: "0.75rem 0" }}>
      <div style={{ fontSize: 48, fontWeight: 500, color, lineHeight: 1 }}>{pct}%</div>
      <div style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: "4px 0 14px" }}>
        chance of {direction} {threshold} {(STAT_LABELS[stat] || stat).toLowerCase()}
      </div>
      <div style={{ position: "relative", height: 8, borderRadius: 4, background: "var(--color-background-tertiary)" }}>
        <div style={{ position: "absolute", left: 0, width: `${pct}%`, height: "100%", background: color, borderRadius: 4, transition: "width 0.6s ease" }} />
        <div style={{ position: "absolute", left: "50%", top: -2, width: 1.5, height: 12, background: "var(--color-border-primary)", borderRadius: 1 }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, fontSize: 11, color: "var(--color-text-tertiary)" }}>
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>
  );
};

// ── Player search ─────────────────────────────────────────────────────────────

const PlayerSearch = ({ onSelect, selectedPlayer, players }) => {
  const [query, setQuery]     = useState("");
  const [open, setOpen]       = useState(false);
  const [focused, setFocused] = useState(0);
  const inputRef = useRef(null);
  const listRef  = useRef(null);

  const results = query.length < 1
    ? []
    : players.filter(p =>
        p.name.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 8);

  const choose = p => {
    onSelect(p);
    setQuery("");
    setOpen(false);
  };

  useEffect(() => {
    setFocused(0);
  }, [query]);

  const onKey = e => {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown")  { e.preventDefault(); setFocused(f => Math.min(f + 1, results.length - 1)); }
    if (e.key === "ArrowUp")    { e.preventDefault(); setFocused(f => Math.max(f - 1, 0)); }
    if (e.key === "Enter")      { e.preventDefault(); choose(results[focused]); }
    if (e.key === "Escape")     { setOpen(false); setQuery(""); }
  };

  return (
    <div style={{ position: "relative" }}>

      {/* SELECTED PLAYER */}
      {selectedPlayer && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginBottom: 10,
          padding: "10px 14px",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 12
        }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ fontSize: 15, fontWeight: 500 }}>
                {selectedPlayer.name}
              </div>
              <PosBadge pos={selectedPlayer.pos} />
            </div>
          </div>

          <button
            onClick={() => {
              onSelect(null);
              setQuery("");
              inputRef.current?.focus();
            }}
            style={{
              fontSize: 12,
              padding: "6px 10px",
              borderRadius: 8,
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "#9CA3AF",
              cursor: "pointer"
            }}
          >
            Change
          </button>
        </div>
      )}

      {/* SEARCH INPUT */}
      {!selectedPlayer && (
        <div style={{ position: "relative" }}>
          <input
            ref={inputRef}
            type="text"
            placeholder="Search players — e.g. LeBron, Curry, Luka..."
            value={query}
            onChange={e => { setQuery(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 150)}
            onKeyDown={onKey}
            style={{
              width: "80%",
              fontSize: 14,
              padding: "10px 12px 10px 36px",
              borderRadius: 12,
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "#FFFFFF",
              outline: "none"
            }}
            autoFocus
          />

          {/* SEARCH ICON */}
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            style={{
              position: "absolute",
              left: 12,
              top: "50%",
              transform: "translateY(-50%)",
              opacity: 0.5
            }}
            fill="none"
          >
            <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M10.5 10.5L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
        </div>
      )}

      {/* DROPDOWN */}
      {open && results.length > 0 && (
        <div
          ref={listRef}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 100,
            marginTop: 6,
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 12,
            boxShadow: "0 10px 30px rgba(0,0,0,0.5)",
            overflow: "hidden"
          }}
        >
          {results.map((p, i) => {
            const isFocused = i === focused;

            return (
              <div
                key={p.id}
                onMouseDown={() => choose(p)}
                onMouseEnter={() => setFocused(i)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 14px",
                  cursor: "pointer",

                  background: isFocused
                    ? "rgba(255,255,255,0.08)"
                    : "transparent",

                  borderBottom:
                    i < results.length - 1
                      ? "1px solid rgba(255,255,255,0.05)"
                      : "none",

                  transition: "background 0.15s ease"
                }}
              >
                <span style={{ flex: 1, fontSize: 14 }}>
                  {p.name}
                </span>
                <PosBadge pos={p.pos} />
              </div>
            );
          })}
        </div>
      )}

      {/* NO RESULTS */}
      {open && query.length > 1 && results.length === 0 && (
        <div style={{
          position: "absolute",
          top: "100%",
          left: 0,
          right: 0,
          zIndex: 100,
          marginTop: 6,
          padding: "12px 14px",
          borderRadius: 12,
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          fontSize: 13,
          color: "#9CA3AF"
        }}>
          No players found for "{query}"
        </div>
      )}
    </div>
  );
};

// ── Main app ──────────────────────────────────────────────────────────────────

export default function App() {
  const [players, setPlayers]   = useState([]);
  const [playersErr, setPlayersErr] = useState(null);
  const [player, setPlayer]     = useState(null);
  const [context, setContext]   = useState(null);
  const [ctxLoad, setCtxLoad]   = useState(false);
  const [ctxError, setCtxError] = useState(null);
  const [result, setResult]     = useState(null);
  const [predLoad, setPredLoad] = useState(false);
  const [predError, setPredErr] = useState(null);
  const [probStat, setProbStat]     = useState("pts");
  const [probThresh, setProbThresh] = useState("");
  const [probDir, setProbDir]       = useState("over");
  const [probResult, setProbResult] = useState(null);
  const [probLoad, setProbLoad]     = useState(false);
  const [probError, setProbErr]     = useState(null);

  // Load the searchable roster from the API (players the model can predict).
  // Replaces the old hardcoded list, so it stays in sync with scraped data.
  useEffect(() => {
    fetch(`${API}/players`)
      .then(r => r.json())
      .then(d => setPlayers(d.players || []))
      .catch(() => setPlayersErr("Couldn't load players — is the API running on :8000?"));
  }, []);

  const handleSelectPlayer = p => {
    setPlayer(p);
    setContext(null);
    setResult(null);
    setProbResult(null);
    setCtxError(null);
    setPredErr(null);
  };

  const fetchContext = useCallback(async p => {
    if (!p) return;
    setCtxLoad(true);
    setCtxError(null);
    setContext(null);
    setResult(null);
    setProbResult(null);
    try {
      const resp = await fetch(
        `${API}/next-game/${encodeURIComponent(p.name)}?player_id=${p.id}&position=${p.pos}`
      );
      if (!resp.ok) throw new Error((await resp.json()).detail || "Failed");
      setContext(await resp.json());
    } catch (e) {
      setCtxError(e.message);
    } finally {
      setCtxLoad(false);
    }
  }, []);

  const predict = async () => {
    if (!context) return;
    setPredLoad(true);
    setPredErr(null);
    setResult(null);
    setProbResult(null);
    try {
      const body = Object.fromEntries(
        Object.entries(context).filter(([k]) => !k.startsWith("_"))
      );
      const injWarn  = context._injury_warning || "";
      const gameDate = context._game_date || "";
      const resp = await fetch(
        `${API}/predict?game_date=${encodeURIComponent(gameDate)}&injury_warning=${encodeURIComponent(injWarn)}`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
      if (!resp.ok) throw new Error((await resp.json()).detail || "Failed");
      const data = await resp.json();
      data.injury_status = context._injury_status || null;
      setResult(data);
    } catch (e) {
      setPredErr(e.message);
    } finally {
      setPredLoad(false);
    }
  };

  const computeProb = async () => {
    if (!probThresh || isNaN(Number(probThresh)) || !player) return;
    setProbLoad(true);
    setProbErr(null);
    setProbResult(null);
    try {
      const resp = await fetch(
        `${API}/probability?` + new URLSearchParams({
          stat: probStat, threshold: probThresh,
          direction: probDir, player_name: player.name,
        })
      );
      if (!resp.ok) throw new Error((await resp.json()).detail || "Failed");
      setProbResult(await resp.json());
    } catch (e) {
      setProbErr(e.message);
    } finally {
      setProbLoad(false);
    }
  };

  const preds  = result?.predictions;
  const ranges = result?.ranges;

  const isInjured = ctx => ctx?._injury_status && !["Active", "Unknown", null].includes(ctx._injury_status);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "2rem 1rem", fontFamily: "var(--font-sans)" }}>

      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 6px" }}>NBA Predictor</h1>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
          Search a player to predict their next game stat line
        </p>
      </div>

      {/* Step 1: Search */}
      <Card style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Player
          </div>
          {players.length > 0 && (
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
              {players.length} players
            </div>
          )}
        </div>
        {playersErr && (
          <div style={{ padding: "10px 14px", borderRadius: "var(--border-radius-md)", background: "#FCEBEB", color: "#791F1F", fontSize: 13, marginBottom: 10 }}>
            {playersErr}
          </div>
        )}
        <PlayerSearch onSelect={handleSelectPlayer} selectedPlayer={player} players={players} />

        {/* Step 2: Find game — only shows once player is selected */}
        {player && !context && !ctxLoad && (
          <button onClick={() => fetchContext(player)} style={{
            marginTop: 12, width: "100%", padding: "11px 0",
            fontSize: 14, cursor: "pointer",
            background: "var(--color-background-secondary)",
            border: "1px solid var(--color-border-secondary)",
            borderRadius: "10px",
            fontWeight: 500,
            color: "var(--color-text-primary)",
          }}>
            Find next game for {player.name} →
          </button>
        )}

        {ctxLoad && (
          <div style={{ marginTop: 12, padding: "11px 0", textAlign: "center", fontSize: 14, color: "var(--color-text-secondary)" }}>
            Fetching schedule, defense stats, injury report...
          </div>
        )}

        {ctxError && (
          <div style={{ marginTop: 12, padding: "10px 14px", borderRadius: "var(--border-radius-md)", background: "#FCEBEB", color: "#791F1F", fontSize: 13 }}>
            {ctxError}
          </div>
        )}

        {/* Game context card */}
        {context && (
          <div style={{ marginTop: 12 }}>

            {/* Injury warning */}
            {isInjured(context) && (
              <div style={{ padding: "10px 14px", borderRadius: "var(--border-radius-md)", background: "#FCEBEB", color: "#791F1F", fontSize: 13, marginBottom: 10, display: "flex", gap: 8, alignItems: "flex-start" }}>
                <span style={{ fontWeight: 500, flexShrink: 0 }}>Injury alert</span>
                <span>{context._injury_warning}</span>
              </div>
            )}

            {/* Game summary */}
            <div style={{ padding: "12px 14px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--color-border-secondary)", borderRadius: "var(--border-radius-md)", marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div>
                  <span style={{ fontSize: 15, fontWeight: 500 }}>
                    {context._player_team_abbrev} {context.home_game ? "vs" : "@"} {context.opponent_abbrev}
                  </span>
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary)", marginLeft: 10 }}>{context._game_date}</span>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {context.back_to_back && (
                    <span style={{ fontSize: 11, padding: "2px 7px", borderRadius: 4, background: "#FAEEDA", color: "#633806", fontWeight: 500 }}>B2B</span>
                  )}
                  <span style={{ fontSize: 11, padding: "2px 7px", borderRadius: 4, background: "var(--color-background-tertiary)", color: "var(--color-text-secondary)" }}>
                    {context.rest_days === 0 ? "0 days rest" : `${context.rest_days}d rest`}
                  </span>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                {[
                  { label: "Opp def rating", value: context.opp_def_rating },
                  { label: "Def rank",        value: `${context.opp_def_rank}/30` },
                  { label: "Team pace",       value: context.team_pace },
                ].map(({ label, value }) => (
                  <div key={label} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 500 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Predict button */}
            <button onClick={predict} disabled={predLoad} style={{
              width: "100%", padding: "13px 0", fontSize: 15, fontWeight: 500,
              cursor: predLoad ? "not-allowed" : "pointer",
              background: predLoad ? "var(--color-background-secondary)" : "#1E293B",
              color: "#FFFFFF",
              border: "1px solid var(--color-border-secondary)",
              borderRadius: "10px",
              transition: "opacity 0.15s",
            }}>
              {predLoad ? "Running prediction..." : `Predict ${player?.name} →`}
            </button>

            {predError && (
              <div style={{ marginTop: 8, padding: "10px 14px", borderRadius: "var(--border-radius-md)", background: "#FCEBEB", color: "#791F1F", fontSize: 13 }}>
                {predError}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Results */}
      {result && preds && (
        <>
          {/* Fantasy score + stat bars */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 500 }}>{result.player_name}</div>
                  <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 2 }}>
                    vs {result.opponent} · {context?.home_game ? "Home" : "Away"}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 2 }}>DK fantasy score</div>
                  <div style={{ fontSize: 30, fontWeight: 500, color: "#185FA5" }}>{result.fantasy_score.toFixed(1)}</div>
                </div>
              </div>

              <div style={{ display: "flex", gap: 10, marginBottom: 14, fontSize: 11, color: "var(--color-text-tertiary)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ display: "inline-block", width: 18, height: 4, borderRadius: 2, background: "#185FA5" }} />
                  Prediction
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ display: "inline-block", width: 18, height: 4, borderRadius: 2, background: "#185FA5", opacity: 0.2 }} />
                  ±1 std dev
                </span>
              </div>

              {Object.entries(STAT_LABELS).map(([key, label]) => (
                <StatBar key={key} label={label} value={preds[key]}
                  range={ranges?.[key]} max={STAT_MAX[key]} color={STAT_COLORS[key]} />
              ))}
            </Card>

            {/* Reasoning + chips */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Card style={{ padding: "0.75rem" }}>
                <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Why this prediction</div>
                {result.reasoning.split("\n").map((line, i) => (
                  <ReasonLine key={i} line={line} />
                ))}
              </Card>

              {/* Stat chips */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(6, minmax(0, 1fr))", gap: 8 }}>
                {Object.entries(STAT_LABELS).map(([key, label]) => {
                  const [lo, hi] = ranges?.[key] || [preds[key], preds[key]];
                  return (
                    <div key={key} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid var(--color-border-secondary)", borderRadius: "10px", padding: "10px 4px", textAlign: "center" }}>
                      <div style={{ fontSize: 10, color: "var(--color-text-secondary)", marginBottom: 2 }}>{label.slice(0,3).toUpperCase()}</div>
                      <div style={{ fontSize: 17, fontWeight: 500, color: STAT_COLORS[key] }}>{preds[key]}</div>
                      <div style={{ fontSize: 9, color: "var(--color-text-tertiary)", marginTop: 1 }}>{lo}–{hi}</div>
                    </div>
                  );
                })}
              </div>

              {result.warnings?.length > 0 && result.warnings.map((w, i) => (
                <div key={i} style={{ padding: "10px 14px", borderRadius: "var(--border-radius-md)", background: "#FAEEDA", color: "#633806", fontSize: 13, lineHeight: 1.5 }}>
                  {w}
                </div>
              ))}
            </div>
          </div>

          {/* Probability calculator */}
          <Card style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 14,
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
              Probability calculator
            </div>

            <div style={{ fontSize: 13, color: "#9CA3AF", marginBottom: 18 }}>
              What are the odds {player?.name} goes over or under a number tonight?
            </div>

            <div style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr 1.2fr auto",
              gap: 12,
              alignItems: "flex-end"
            }}>

              {/* STAT */}
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
                  Stat
                </div>

                <select
                  value={probStat}
                  onChange={e => { setProbStat(e.target.value); setProbResult(null); }}
                  style={{
                    width: "100%",
                    padding: "9px 10px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    color: "#FFFFFF",
                    fontSize: 13
                  }}
                >
                  {Object.entries(STAT_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>{v}</option>
                  ))}
                </select>
              </div>

              {/* DIRECTION */}
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
                  Direction
                </div>

                <div style={{
                  display: "flex",
                  gap: 6,
                  background: "rgba(255,255,255,0.04)",
                  padding: 4,
                  borderRadius: 12
                }}>
                  {["over", "under"].map(d => {
                    const isSelected = probDir === d;

                    return (
                      <button
                        key={d}
                        onClick={() => { setProbDir(d); setProbResult(null); }}
                        onMouseEnter={e => {
                          if (!isSelected) e.currentTarget.style.background = "rgba(255,255,255,0.08)";
                        }}
                        onMouseLeave={e => {
                          if (!isSelected) e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                        }}
                        style={{
                          flex: 1,
                          padding: "9px 0",
                          fontSize: 13,
                          fontWeight: 500,
                          cursor: "pointer",
                          borderRadius: 10,

                          background: isSelected
                            ? (d === "over" ? "#10B981" : "#EF4444")
                            : "rgba(255,255,255,0.03)",

                          color: isSelected ? "#FFFFFF" : "#9CA3AF",

                          border: isSelected
                            ? `1px solid ${d === "over" ? "#10B981" : "#EF4444"}`
                            : "1px solid rgba(255,255,255,0.08)",

                          transition: "all 0.2s ease"
                        }}
                      >
                        {d.charAt(0).toUpperCase() + d.slice(1)}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* THRESHOLD */}
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
                  Threshold{" "}
                  <span style={{ color: "#6B7280" }}>
                    (predicted: {preds?.[probStat]})
                  </span>
                </div>

                <input
                  type="number"
                  step="0.5"
                  placeholder={preds ? String(preds[probStat]) : "e.g. 26"}
                  value={probThresh}
                  onChange={e => { setProbThresh(e.target.value); setProbResult(null); }}
                  onKeyDown={e => e.key === "Enter" && computeProb()}
                  style={{
                    width: "90%",
                    padding: "9px 10px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    color: "#FFFFFF",
                    fontSize: 13
                  }}
                />
              </div>

              {/* CALCULATE BUTTON */}
              <button
                onClick={computeProb}
                disabled={probLoad || !probThresh}
                style={{
                  padding: "10px 20px",
                  fontSize: 13,
                  fontWeight: 600,
                  borderRadius: 10,
                  border: "1px solid rgba(255,255,255,0.08)",

                  background: (probLoad || !probThresh)
                    ? "rgba(255,255,255,0.05)"
                    : "#1F2937",

                  color: (probLoad || !probThresh)
                    ? "#6B7280"
                    : "#FFFFFF",

                  cursor: (probLoad || !probThresh)
                    ? "not-allowed"
                    : "pointer",

                  transition: "all 0.2s ease",
                  whiteSpace: "nowrap"
                }}
              >
                {probLoad ? "..." : "Calculate →"}
              </button>
            </div>

            {/* ERROR */}
            {probError && (
              <div style={{
                marginTop: 12,
                padding: "10px",
                borderRadius: 10,
                background: "rgba(239,68,68,0.15)",
                color: "#FCA5A5",
                fontSize: 13
              }}>
                {probError}
              </div>
            )}

            {/* RESULT */}
            {probResult && (
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 20,
                marginTop: 22,
                paddingTop: 18,
                borderTop: "1px solid rgba(255,255,255,0.08)"
              }}>
                <ProbGauge
                  prob={probResult.probability}
                  direction={probResult.direction}
                  threshold={probResult.threshold}
                  stat={probResult.stat}
                />

                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    ["Probability", probResult.pct_display],
                    ["Empirical rate", `${Math.round(probResult.hit_rate * 100)}%`],
                    ["Normal dist", `${Math.round(probResult.normal_prob * 100)}%`],
                    ["Recent average", `${probResult.stat_mean}`],
                    ["Std deviation", `±${probResult.stat_std}`],
                    ["Games sampled", `${probResult.sample_size}`],
                    ["Data source", probResult.data_source || "training data"],
                  ].map(([label, value]) => (
                    <div key={label} style={{
                      display: "flex",
                      justifyContent: "space-between",
                      fontSize: 13,
                      padding: "6px 0",
                      borderBottom: "1px solid rgba(255,255,255,0.06)"
                    }}>
                      <span style={{ color: "#9CA3AF" }}>{label}</span>
                      <span style={{ fontWeight: 500 }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}