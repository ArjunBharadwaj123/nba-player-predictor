import { useState, useCallback, useEffect, useMemo } from "react";
import { API, apiFetch } from "../lib/api";
import { Card, StatBar, ProbGauge, ErrorBox, EmptyState, Spinner } from "../components/ui";
import PlayerSearch from "../components/PlayerSearch";
import { useLocalStorage } from "../lib/useLocalStorage";
import {
  POSITION_CONFIG, POS_COLORS, POS_TEXT, NFL_GREEN, NFL_GREEN_SOFT,
  statMeta, fmt,
} from "../lib/nflConfig";
import { fantasyPoints, SCORING_FORMAT_LABELS, SCORING_FORMATS } from "../lib/fantasy";

const POSITIONS = ["All", "QB", "RB", "WR", "TE", "K"];
const SCORING_KEY = "nfl_scoring_format"; // isolated from NBA state

export default function NflPage() {
  const [players, setPlayers] = useState([]);
  const [playersErr, setPlayersErr] = useState(null);
  const [playersLoad, setPlayersLoad] = useState(true);
  const [posFilter, setPosFilter] = useState("All");
  const [player, setPlayer] = useState(null);

  const [scoring, setScoring] = useLocalStorage(SCORING_KEY, "ppr");

  const [context, setContext] = useState(null);
  const [ctxLoad, setCtxLoad] = useState(false);
  const [ctxError, setCtxError] = useState(null);
  const [noGame, setNoGame] = useState(null);

  const [result, setResult] = useState(null);
  const [predLoad, setPredLoad] = useState(false);
  const [predError, setPredErr] = useState(null);

  // Locally-refined fantasy (recomputed on scoring change, no ML rerun).
  const [fantasy, setFantasy] = useState(null);

  const [probStat, setProbStat] = useState("");
  const [probThresh, setProbThresh] = useState("");
  const [probDir, setProbDir] = useState("over");
  const [probResult, setProbResult] = useState(null);
  const [probLoad, setProbLoad] = useState(false);
  const [probError, setProbErr] = useState(null);

  const [freshness, setFreshness] = useState(null);

  // ── Load players + freshness ────────────────────────────────────────────────
  const loadPlayers = useCallback(() => {
    setPlayersLoad(true);
    setPlayersErr(null);
    apiFetch("/nfl/players")
      .then((d) => setPlayers(d.players || []))
      .catch(() => setPlayersErr("Couldn't load NFL players — is the API running on :8000?"))
      .finally(() => setPlayersLoad(false));
  }, []);

  useEffect(() => {
    loadPlayers();
    apiFetch("/nfl/health").then((d) => setFreshness(d.data_freshness)).catch(() => {});
  }, [loadPlayers]);

  const filtered = useMemo(
    () => (posFilter === "All" ? players : players.filter((p) => p.position === posFilter)),
    [players, posFilter],
  );

  const posCfg = player ? POSITION_CONFIG[player.position] : null;

  const resetForPlayer = (p) => {
    setPlayer(p);
    setContext(null); setCtxError(null); setNoGame(null);
    setResult(null); setFantasy(null); setPredErr(null);
    setProbResult(null); setProbStat("");
  };

  // ── Upcoming game lookup ────────────────────────────────────────────────────
  const fetchContext = useCallback(async (p) => {
    if (!p) return;
    setCtxLoad(true); setCtxError(null); setNoGame(null); setContext(null);
    try {
      const data = await apiFetch(`/nfl/next-game/${encodeURIComponent(p.id)}`);
      setContext(data);
    } catch (e) {
      if (e.status === 404) setNoGame(e.message);
      else setCtxError(e.message);
    } finally {
      setCtxLoad(false);
    }
  }, []);

  // ── Prediction (runs ML models once) ────────────────────────────────────────
  const predict = useCallback(async () => {
    if (!player) return;
    setPredLoad(true); setPredErr(null); setResult(null); setProbResult(null);
    try {
      const body = {
        player_id: player.id,
        scoring_format: scoring,
        context: context
          ? {
              opponent_team: context.opponent_team,
              home_away: context.home_away,
              days_rest: context.days_rest,
              spread_line_team: context.spread_line_team,
              game_total: context.game_total,
              implied_team_total: context.implied_team_total,
              is_indoor: context.is_indoor,
              is_grass: context.is_grass,
              temp: context.temp,
              wind: context.wind,
              div_game: context.div_game,
              injury_designation: context.injury_designation,
            }
          : {},
      };
      const data = await apiFetch("/nfl/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setResult(data);
      setFantasy(data.fantasy_points);
    } catch (e) {
      setPredErr(e.message);
    } finally {
      setPredLoad(false);
    }
  }, [player, context, scoring]);

  // ── Recompute fantasy on scoring change (NO ML rerun) ───────────────────────
  useEffect(() => {
    if (!result || !player) return;
    // Instant local point estimate.
    const localPoint = fantasyPoints(result.predictions, player.position, scoring);
    setFantasy((prev) => ({
      ...(prev || {}),
      scoring_format: scoring,
      label: `${SCORING_FORMAT_LABELS[scoring]} Fantasy Points`,
      point_estimate: Math.round(localPoint * 100) / 100,
    }));
    // Refine simulated interval + probability via the lightweight scoring
    // endpoint (reuses the same component predictions — never reruns XGBoost).
    let cancelled = false;
    apiFetch("/nfl/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        position: player.position,
        scoring_format: scoring,
        predictions: result.predictions,
        intervals: result.intervals,
        fantasy_threshold: null,
      }),
    })
      .then((d) => { if (!cancelled) setFantasy((prev) => ({ ...(prev || {}), ...d })); })
      .catch(() => {});
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scoring, result]);

  // ── Over/under probability ──────────────────────────────────────────────────
  const computeProb = async () => {
    if (!player || !probStat || !probThresh || isNaN(Number(probThresh))) return;
    setProbLoad(true); setProbErr(null); setProbResult(null);
    try {
      const data = await apiFetch(
        `/nfl/probability?` +
          new URLSearchParams({
            player_id: player.id,
            stat: probStat,
            threshold: probThresh,
            direction: probDir,
            scoring_format: scoring,
          }),
      );
      setProbResult(data);
    } catch (e) {
      setProbErr(e.message);
    } finally {
      setProbLoad(false);
    }
  };

  const preds = result?.predictions;
  const intervals = result?.intervals;

  return (
    <div>
      {/* 1. Heading + description */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 6px" }}>NFL Predictor</h1>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary, #9ca3af)", margin: 0 }}>
          Position-specific projections, fantasy points, and over/under odds for a player&apos;s next game
        </p>
      </div>

      <Card style={{ marginBottom: 12 }}>
        {/* 3. Scoring format selector */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary, #6b7280)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Scoring format</div>
          <div role="radiogroup" aria-label="Scoring format" style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.04)", padding: 4, borderRadius: 10 }}>
            {SCORING_FORMATS.map((f) => {
              const sel = scoring === f;
              return (
                <button key={f} role="radio" aria-checked={sel} onClick={() => setScoring(f)} style={{ padding: "7px 12px", fontSize: 12, fontWeight: 600, borderRadius: 8, cursor: "pointer", border: sel ? `1px solid ${NFL_GREEN}` : "1px solid transparent", background: sel ? NFL_GREEN : "transparent", color: sel ? "#fff" : "var(--color-text-secondary, #9ca3af)" }}>
                  {SCORING_FORMAT_LABELS[f]}
                </button>
              );
            })}
          </div>
        </div>

        {/* 4. Position filters */}
        <div style={{ display: "flex", gap: 6, marginBottom: 12, flexWrap: "wrap" }}>
          {POSITIONS.map((pos) => {
            const sel = posFilter === pos;
            return (
              <button key={pos} onClick={() => setPosFilter(pos)} style={{ padding: "6px 12px", fontSize: 12, fontWeight: 600, borderRadius: 8, cursor: "pointer", border: sel ? `1px solid ${NFL_GREEN}` : "1px solid var(--color-border-secondary, #2e303a)", background: sel ? NFL_GREEN_SOFT : "transparent", color: sel ? NFL_GREEN : "var(--color-text-secondary, #9ca3af)" }}>
                {pos}
              </button>
            );
          })}
          {players.length > 0 && (
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", alignSelf: "center" }}>{filtered.length} players</span>
          )}
        </div>

        {/* 5. Player search */}
        {playersLoad && <Spinner label="Loading NFL players…" />}
        {playersErr && <ErrorBox onRetry={loadPlayers}>{playersErr}</ErrorBox>}
        {!playersLoad && !playersErr && (
          <PlayerSearch
            players={filtered}
            selectedPlayer={player}
            onSelect={(p) => (p ? resetForPlayer(p) : resetForPlayer(null))}
            positionKey="position"
            colors={POS_COLORS}
            textColors={POS_TEXT}
            showTeam
            groupByPosition
            accent={NFL_GREEN}
            placeholder="Search players — e.g. Mahomes, McCaffrey, Kelce…"
          />
        )}

        {/* 6. Selected-player card + upcoming-game lookup */}
        {player && (
          <div style={{ marginTop: 12 }}>
            <button onClick={() => fetchContext(player)} disabled={ctxLoad} style={secondaryBtn}>
              {ctxLoad ? "Finding next game…" : `Find upcoming game for ${player.name} →`}
            </button>
            {ctxLoad && <div style={{ marginTop: 10 }}><Spinner label="Fetching schedule, injury report, matchup…" /></div>}
            {ctxError && <div style={{ marginTop: 10 }}><ErrorBox onRetry={() => fetchContext(player)}>{ctxError}</ErrorBox></div>}
            {noGame && (
              <div style={{ marginTop: 10 }}>
                <EmptyState icon="🏈">{noGame} No projection is made without a scheduled game.</EmptyState>
              </div>
            )}

            {/* 8. Game-context card */}
            {context && <GameContextCard context={context} />}

            {/* 9. Injury / depth warnings */}
            {context && (context.injury_designation || context.depth_rank) && (
              <InjuryDepth context={context} />
            )}

            {/* 10. Run prediction — enabled only once an upcoming game is found */}
            {context && (
              <>
                <button onClick={predict} disabled={predLoad} style={{ ...primaryBtn, marginTop: 12, background: predLoad ? "var(--color-background-secondary, #202128)" : NFL_GREEN }}>
                  {predLoad ? "Running projection…" : `Project ${player.name} →`}
                </button>
                {predError && <div style={{ marginTop: 8 }}><ErrorBox onRetry={predict}>{predError}</ErrorBox></div>}
              </>
            )}
          </div>
        )}
      </Card>

      {/* Results */}
      {result && preds && posCfg && (
        <>
          <div className="nfl-results-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            {/* 11-13. Position-specific predictions + intervals + fantasy */}
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 500 }}>{result.player.name}</div>
                  <div style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)", marginTop: 2 }}>
                    {result.player.position} · {result.player.team}
                    {context ? ` · ${context.home_away ? "vs" : "@"} ${context.opponent_team}` : " · neutral matchup"}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", marginBottom: 2 }}>{fantasy?.label || "Fantasy Points"}</div>
                  <div style={{ fontSize: 30, fontWeight: 500, color: NFL_GREEN }}>{fmt(fantasy?.point_estimate, "kicking_points")}</div>
                  {fantasy?.interval && (
                    <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)" }}>
                      {fantasy.interval[0]?.toFixed?.(1)}–{fantasy.interval[1]?.toFixed?.(1)} range
                    </div>
                  )}
                </div>
              </div>
              <div style={{ display: "flex", gap: 10, marginBottom: 14, fontSize: 11, color: "var(--color-text-tertiary, #6b7280)" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ display: "inline-block", width: 18, height: 4, borderRadius: 2, background: NFL_GREEN }} /> Projection
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ display: "inline-block", width: 18, height: 4, borderRadius: 2, background: NFL_GREEN, opacity: 0.2 }} /> p15–p85
                </span>
              </div>
              {posCfg.order.map((key) => {
                const m = statMeta(key);
                const range = intervals?.[key];
                return (
                  <StatBar
                    key={key}
                    label={m.label}
                    value={fmt(preds[key], key)}
                    range={range ? [fmt(range[0], key), fmt(range[1], key)] : undefined}
                    max={m.max}
                    color={m.color}
                    unit={m.unit}
                  />
                );
              })}
            </Card>

            {/* 14. SHAP explanation + warnings */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Card style={{ padding: "0.75rem" }}>
                <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Why this projection</div>
                {result.reasons?.length ? (
                  result.reasons.map((r, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, padding: "8px 10px", background: "rgba(255,255,255,0.02)", borderRadius: 8, borderBottom: "0.5px solid var(--color-border-tertiary, #2e303a)", alignItems: "flex-start" }}>
                      <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 6px", borderRadius: 4, flexShrink: 0, marginTop: 1, background: r.direction === "+" ? "#E1F5EE" : "#FCEBEB", color: r.direction === "+" ? "#085041" : "#791F1F" }}>
                        {r.direction === "+" ? "+" : "−"}
                      </span>
                      <span style={{ fontSize: 13, lineHeight: 1.5 }}>{r.label}</span>
                    </div>
                  ))
                ) : (
                  <div style={{ fontSize: 12, color: "var(--color-text-tertiary, #6b7280)" }}>No standout factors — projection near the player&apos;s recent baseline.</div>
                )}
                <div style={{ fontSize: 10, color: "var(--color-text-tertiary, #6b7280)", marginTop: 8, fontStyle: "italic" }}>
                  Feature attributions (SHAP) show what moved this projection — not proof of causation.
                </div>
              </Card>

              {result.warnings?.length > 0 && result.warnings.map((w, i) => (
                <div key={i} style={{ padding: "10px 14px", borderRadius: 8, background: "#FAEEDA", color: "#633806", fontSize: 13, lineHeight: 1.5 }}>{w}</div>
              ))}
            </div>
          </div>

          {/* 15. Over/under probability calculator */}
          <Card style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 14, marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Over/under probability</div>
            <div style={{ fontSize: 13, color: "#9CA3AF", marginBottom: 4 }}>
              Odds {player.name} clears a number in this game.
            </div>
            {probStat === "fantasy_points" && (
              <div style={{ fontSize: 12, color: NFL_GREEN, marginBottom: 12, fontWeight: 500 }}>
                Active scoring: {SCORING_FORMAT_LABELS[scoring]} — this probability is specific to {SCORING_FORMAT_LABELS[scoring]} scoring.
              </div>
            )}
            <div className="prob-grid" style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr 1.1fr auto", gap: 12, alignItems: "flex-end", marginTop: 8 }}>
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>Stat</div>
                <select value={probStat} onChange={(e) => { setProbStat(e.target.value); setProbResult(null); }} style={selectStyle}>
                  <option value="">Select…</option>
                  {posCfg.probabilityOptions.map((k) => (
                    <option key={k} value={k}>{k === "fantasy_points" ? `${SCORING_FORMAT_LABELS[scoring]} Fantasy Points` : statMeta(k).label}</option>
                  ))}
                </select>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>Direction</div>
                <div style={{ display: "flex", gap: 6, background: "rgba(255,255,255,0.04)", padding: 4, borderRadius: 12 }}>
                  {["over", "under"].map((d) => {
                    const sel = probDir === d;
                    return (
                      <button key={d} onClick={() => { setProbDir(d); setProbResult(null); }} style={{ flex: 1, padding: "9px 0", fontSize: 13, fontWeight: 500, cursor: "pointer", borderRadius: 10, background: sel ? (d === "over" ? NFL_GREEN : "#EF4444") : "rgba(255,255,255,0.03)", color: sel ? "#fff" : "#9CA3AF", border: sel ? `1px solid ${d === "over" ? NFL_GREEN : "#EF4444"}` : "1px solid rgba(255,255,255,0.08)" }}>
                        {d.charAt(0).toUpperCase() + d.slice(1)}
                      </button>
                    );
                  })}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>
                  Threshold {probStat && probStat !== "fantasy_points" && preds?.[probStat] != null ? <span style={{ color: "#6B7280" }}>(proj: {fmt(preds[probStat], probStat)})</span> : null}
                </div>
                <input type="number" step="0.5" placeholder="e.g. 60" value={probThresh} onChange={(e) => { setProbThresh(e.target.value); setProbResult(null); }} onKeyDown={(e) => e.key === "Enter" && computeProb()} style={{ ...selectStyle, width: "100%", boxSizing: "border-box" }} />
              </div>
              <button onClick={computeProb} disabled={probLoad || !probThresh || !probStat} style={{ padding: "10px 20px", fontSize: 13, fontWeight: 600, borderRadius: 10, border: "1px solid rgba(255,255,255,0.08)", background: probLoad || !probThresh || !probStat ? "rgba(255,255,255,0.05)" : NFL_GREEN, color: probLoad || !probThresh || !probStat ? "#6B7280" : "#fff", cursor: probLoad || !probThresh || !probStat ? "not-allowed" : "pointer", whiteSpace: "nowrap" }}>
                {probLoad ? "…" : "Calculate →"}
              </button>
            </div>
            {probError && <div style={{ marginTop: 12 }}><ErrorBox onRetry={computeProb}>{probError}</ErrorBox></div>}
            {probResult && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 22, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                <ProbGauge prob={probResult.probability} accent={NFL_GREEN} caption={`chance of ${probResult.direction} ${probResult.threshold} ${probStat === "fantasy_points" ? `${SCORING_FORMAT_LABELS[scoring]} pts` : statMeta(probStat).label.toLowerCase()}`} />
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    ["Probability", probResult.pct_display],
                    ["Projected", `${probResult.predicted}`],
                    probStat === "fantasy_points" ? ["Scoring", probResult.scoring_label] : null,
                    ["Direction", probResult.direction],
                    ["Threshold", probResult.threshold],
                  ].filter(Boolean).map(([label, value]) => (
                    <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
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

      {/* 16. Data-freshness info */}
      {freshness && (
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", textAlign: "center", padding: "4px 0 12px" }}>
          Data: {freshness.mode || "?"} mode · seasons {(freshness.seasons || []).join(", ")}
          {freshness.updated_at ? ` · updated ${String(freshness.updated_at).slice(0, 10)}` : ""}
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────
function GameContextCard({ context }) {
  const chips = [];
  if (context.spread_line_team != null) chips.push(["Spread", context.spread_line_team > 0 ? `+${context.spread_line_team}` : context.spread_line_team]);
  else if (context.missing_odds) chips.push(["Spread", "n/a"]);
  if (context.game_total != null) chips.push(["Total", context.game_total]);
  if (context.implied_team_total != null) chips.push(["Implied", context.implied_team_total]);
  if (!context.missing_weather && context.wind != null) chips.push(["Wind", `${context.wind} mph`]);
  if (!context.missing_weather && context.temp != null) chips.push(["Temp", `${context.temp}°`]);
  return (
    <div style={{ padding: "12px 14px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: 8, marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
        <span style={{ fontSize: 15, fontWeight: 500 }}>
          {context.team} {context.home_away ? "vs" : "@"} {context.opponent_team}
        </span>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)" }}>
          {context.game_date} · Wk {context.week} · {context.days_rest != null ? `${context.days_rest}d rest` : "rest n/a"}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {chips.map(([label, value]) => (
          <div key={label} style={{ fontSize: 11, padding: "4px 9px", borderRadius: 6, background: "var(--color-background-tertiary, #2a2b33)", color: "var(--color-text-secondary, #9ca3af)" }}>
            <span style={{ color: "var(--color-text-tertiary, #6b7280)" }}>{label}: </span>{value}
          </div>
        ))}
        {context.is_indoor ? <div style={chipStyle}>Indoor</div> : <div style={chipStyle}>Outdoor</div>}
        <div style={chipStyle}>{context.is_grass ? "Grass" : "Turf"}</div>
        {context.missing_odds ? <div style={{ ...chipStyle, color: "#8a5a0b" }}>odds unavailable</div> : null}
      </div>
    </div>
  );
}

function InjuryDepth({ context }) {
  const des = (context.injury_designation || "").toUpperCase();
  const danger = des === "OUT" || des === "DOUBTFUL";
  return (
    <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
      {context.injury_designation && (
        <div style={{ padding: "10px 14px", borderRadius: 8, background: danger ? "#FCEBEB" : "#FAEEDA", color: danger ? "#791F1F" : "#633806", fontSize: 13, display: "flex", gap: 8 }}>
          <span style={{ fontWeight: 500 }}>Injury:</span>
          <span>{context.injury_designation}{danger ? " — elevated risk of missing/limited play" : " — monitor status before kickoff"}</span>
        </div>
      )}
      {context.depth_rank != null && (
        <div style={{ fontSize: 12, color: "var(--color-text-secondary, #9ca3af)" }}>
          Depth chart: {context.depth_rank === 1 ? "listed starter" : `#${context.depth_rank} on the depth chart`}
        </div>
      )}
    </div>
  );
}

const chipStyle = { fontSize: 11, padding: "4px 9px", borderRadius: 6, background: "var(--color-background-tertiary, #2a2b33)", color: "var(--color-text-secondary, #9ca3af)" };
const selectStyle = { width: "100%", padding: "9px 10px", borderRadius: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", color: "var(--color-text-primary, #fff)", fontSize: 13, boxSizing: "border-box" };
const primaryBtn = { width: "100%", padding: "13px 0", fontSize: 15, fontWeight: 500, cursor: "pointer", color: "#fff", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: 10 };
const secondaryBtn = { width: "100%", padding: "11px 0", fontSize: 14, cursor: "pointer", background: "var(--color-background-secondary, #202128)", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: 10, fontWeight: 500, color: "var(--color-text-primary, #f3f4f6)" };
