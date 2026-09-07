import { useState, useCallback, useEffect } from "react";
import { API } from "../lib/api";
import { Card, StatBar, ProbGauge, ErrorBox, EmptyState, Spinner } from "../components/ui";
import PlayerSearch from "../components/PlayerSearch";

// NBA predictor page. Behaviour is unchanged from the original single-file app —
// only the shared atoms (Card/StatBar/ProbGauge/PlayerSearch) and the API base
// are now imported from common modules.

const STAT_LABELS = { pts: "Points", reb: "Rebounds", ast: "Assists", stl: "Steals", blk: "Blocks", minutes: "Minutes" };
const STAT_COLORS = { pts: "#185FA5", reb: "#0F6E56", ast: "#534AB7", stl: "#993C1D", blk: "#854F0B", minutes: "#5F5E5A" };
const STAT_MAX = { pts: 55, reb: 22, ast: 16, stl: 5, blk: 5, minutes: 48 };

const POS_COLORS = { PG: "#E6F1FB", SG: "#EAF3DE", SF: "#FAEEDA", PF: "#FAECE7", C: "#EEEDFE" };
const POS_TEXT = { PG: "#0C447C", SG: "#27500A", SF: "#633806", PF: "#712B13", C: "#3C3489" };

const ReasonLine = ({ line }) => {
  if (!line.trim() || line === "Reasoning:") return null;
  const pos = line.startsWith("+"), neg = line.startsWith("-");
  if (!pos && !neg) return null;
  return (
    <div style={{ display: "flex", gap: 8, padding: "8px 10px", background: "rgba(255,255,255,0.02)", borderRadius: "8px", borderBottom: "0.5px solid var(--color-border-tertiary, #2e303a)", alignItems: "flex-start" }}>
      <span style={{ fontSize: 11, fontWeight: 500, padding: "2px 6px", borderRadius: 4, flexShrink: 0, marginTop: 1, background: pos ? "#E1F5EE" : "#FCEBEB", color: pos ? "#085041" : "#791F1F" }}>
        {pos ? "+" : "−"}
      </span>
      <span style={{ fontSize: 13, lineHeight: 1.5 }}>{line.slice(2).trim()}</span>
    </div>
  );
};

export default function NbaPage() {
  const [players, setPlayers] = useState([]);
  const [playersErr, setPlayersErr] = useState(null);
  const [player, setPlayer] = useState(null);
  const [context, setContext] = useState(null);
  const [ctxLoad, setCtxLoad] = useState(false);
  const [ctxError, setCtxError] = useState(null);
  const [noGame, setNoGame] = useState(null);
  const [result, setResult] = useState(null);
  const [predLoad, setPredLoad] = useState(false);
  const [predError, setPredErr] = useState(null);
  const [probStat, setProbStat] = useState("pts");
  const [probThresh, setProbThresh] = useState("");
  const [probDir, setProbDir] = useState("over");
  const [probResult, setProbResult] = useState(null);
  const [probLoad, setProbLoad] = useState(false);
  const [probError, setProbErr] = useState(null);

  const loadPlayers = useCallback(() => {
    setPlayersErr(null);
    fetch(`${API}/players`)
      .then((r) => r.json())
      .then((d) => setPlayers(d.players || []))
      .catch(() => setPlayersErr("Couldn't load players — is the API running on :8000?"));
  }, []);

  useEffect(() => { loadPlayers(); }, [loadPlayers]);

  const handleSelectPlayer = (p) => {
    setPlayer(p);
    setContext(null);
    setResult(null);
    setProbResult(null);
    setCtxError(null);
    setNoGame(null);
    setPredErr(null);
  };

  const fetchContext = useCallback(async (p) => {
    if (!p) return;
    setCtxLoad(true);
    setCtxError(null);
    setNoGame(null);
    setContext(null);
    setResult(null);
    setProbResult(null);
    try {
      const resp = await fetch(`${API}/next-game/${encodeURIComponent(p.name)}?player_id=${p.id}&position=${p.pos}`);
      if (resp.status === 404) {
        setNoGame(`${p.name} doesn't have a game coming up anytime soon — there's no upcoming game scheduled right now. Check back closer to game day (the NBA season runs from late October through April).`);
        return;
      }
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
      const body = Object.fromEntries(Object.entries(context).filter(([k]) => !k.startsWith("_")));
      const injWarn = context._injury_warning || "";
      const gameDate = context._game_date || "";
      const resp = await fetch(`${API}/predict?game_date=${encodeURIComponent(gameDate)}&injury_warning=${encodeURIComponent(injWarn)}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
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
      const resp = await fetch(`${API}/probability?` + new URLSearchParams({ stat: probStat, threshold: probThresh, direction: probDir, player_name: player.name }));
      if (!resp.ok) throw new Error((await resp.json()).detail || "Failed");
      setProbResult(await resp.json());
    } catch (e) {
      setProbErr(e.message);
    } finally {
      setProbLoad(false);
    }
  };

  const preds = result?.predictions;
  const ranges = result?.ranges;
  const isInjured = (ctx) => ctx?._injury_status && !["Active", "Unknown", null].includes(ctx._injury_status);

  return (
    <div>
      <div style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: "0 0 6px" }}>NBA Predictor</h1>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary, #9ca3af)", margin: 0 }}>
          Search a player to predict their next game stat line
        </p>
      </div>

      <Card style={{ marginBottom: 12 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--color-text-tertiary, #6b7280)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Player</div>
          {players.length > 0 && <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)" }}>{players.length} players</div>}
        </div>
        {playersErr && <div style={{ marginBottom: 10 }}><ErrorBox onRetry={loadPlayers}>{playersErr}</ErrorBox></div>}
        <PlayerSearch onSelect={handleSelectPlayer} selectedPlayer={player} players={players} positionKey="pos" colors={POS_COLORS} textColors={POS_TEXT} placeholder="Search players — e.g. LeBron, Curry, Luka…" />

        {player && !context && !ctxLoad && (
          <button onClick={() => fetchContext(player)} style={{ marginTop: 12, width: "100%", padding: "11px 0", fontSize: 14, cursor: "pointer", background: "var(--color-background-secondary, #202128)", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: "10px", fontWeight: 500, color: "var(--color-text-primary, #f3f4f6)" }}>
            Find next game for {player.name} →
          </button>
        )}

        {ctxLoad && <div style={{ marginTop: 12 }}><Spinner label="Fetching schedule, defense stats, injury report…" /></div>}
        {ctxError && <div style={{ marginTop: 12 }}><ErrorBox onRetry={() => fetchContext(player)}>{ctxError}</ErrorBox></div>}
        {noGame && <div style={{ marginTop: 12 }}><EmptyState>{noGame}</EmptyState></div>}

        {context && (
          <div style={{ marginTop: 12 }}>
            {isInjured(context) && (
              <div style={{ padding: "10px 14px", borderRadius: 8, background: "#FCEBEB", color: "#791F1F", fontSize: 13, marginBottom: 10, display: "flex", gap: 8, alignItems: "flex-start" }}>
                <span style={{ fontWeight: 500, flexShrink: 0 }}>Injury alert</span>
                <span>{context._injury_warning}</span>
              </div>
            )}
            <div style={{ padding: "12px 14px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: 8, marginBottom: 12 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div>
                  <span style={{ fontSize: 15, fontWeight: 500 }}>{context._player_team_abbrev} {context.home_game ? "vs" : "@"} {context.opponent_abbrev}</span>
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)", marginLeft: 10 }}>{context._game_date}</span>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {context.back_to_back && <span style={{ fontSize: 11, padding: "2px 7px", borderRadius: 4, background: "#FAEEDA", color: "#633806", fontWeight: 500 }}>B2B</span>}
                  <span style={{ fontSize: 11, padding: "2px 7px", borderRadius: 4, background: "var(--color-background-tertiary, #2a2b33)", color: "var(--color-text-secondary, #9ca3af)" }}>
                    {context.rest_days === 0 ? "0 days rest" : `${context.rest_days}d rest`}
                  </span>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 }}>
                {[{ label: "Opp def rating", value: context.opp_def_rating }, { label: "Def rank", value: `${context.opp_def_rank}/30` }, { label: "Team pace", value: context.team_pace }].map(({ label, value }) => (
                  <div key={label} style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", marginBottom: 2 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 500 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
            <button onClick={predict} disabled={predLoad} style={{ width: "100%", padding: "13px 0", fontSize: 15, fontWeight: 500, cursor: predLoad ? "not-allowed" : "pointer", background: predLoad ? "var(--color-background-secondary, #202128)" : "#1E293B", color: "#FFFFFF", border: "1px solid var(--color-border-secondary, #2e303a)", borderRadius: "10px" }}>
              {predLoad ? "Running prediction…" : `Predict ${player?.name} →`}
            </button>
            {predError && <div style={{ marginTop: 8 }}><ErrorBox onRetry={predict}>{predError}</ErrorBox></div>}
          </div>
        )}
      </Card>

      {result && preds && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }} className="nba-results-grid">
            <Card>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 500 }}>{result.player_name}</div>
                  <div style={{ fontSize: 13, color: "var(--color-text-secondary, #9ca3af)", marginTop: 2 }}>vs {result.opponent} · {context?.home_game ? "Home" : "Away"}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary, #6b7280)", marginBottom: 2 }}>DK fantasy score</div>
                  <div style={{ fontSize: 30, fontWeight: 500, color: "#185FA5" }}>{result.fantasy_score.toFixed(1)}</div>
                </div>
              </div>
              {Object.entries(STAT_LABELS).map(([key, label]) => (
                <StatBar key={key} label={label} value={preds[key]} range={ranges?.[key]} max={STAT_MAX[key]} color={STAT_COLORS[key]} />
              ))}
            </Card>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <Card style={{ padding: "0.75rem" }}>
                <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 10 }}>Why this prediction</div>
                {result.reasoning.split("\n").map((line, i) => <ReasonLine key={i} line={line} />)}
              </Card>
              {result.warnings?.length > 0 && result.warnings.map((w, i) => (
                <div key={i} style={{ padding: "10px 14px", borderRadius: 8, background: "#FAEEDA", color: "#633806", fontSize: 13, lineHeight: 1.5 }}>{w}</div>
              ))}
            </div>
          </div>

          <Card style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 14 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Probability calculator</div>
            <div style={{ fontSize: 13, color: "#9CA3AF", marginBottom: 18 }}>What are the odds {player?.name} goes over or under a number tonight?</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.2fr auto", gap: 12, alignItems: "flex-end" }} className="prob-grid">
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>Stat</div>
                <select value={probStat} onChange={(e) => { setProbStat(e.target.value); setProbResult(null); }} style={selectStyle}>
                  {Object.entries(STAT_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                </select>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>Direction</div>
                <div style={{ display: "flex", gap: 6, background: "rgba(255,255,255,0.04)", padding: 4, borderRadius: 12 }}>
                  {["over", "under"].map((d) => {
                    const sel = probDir === d;
                    return (
                      <button key={d} onClick={() => { setProbDir(d); setProbResult(null); }} style={{ flex: 1, padding: "9px 0", fontSize: 13, fontWeight: 500, cursor: "pointer", borderRadius: 10, background: sel ? (d === "over" ? "#10B981" : "#EF4444") : "rgba(255,255,255,0.03)", color: sel ? "#fff" : "#9CA3AF", border: sel ? `1px solid ${d === "over" ? "#10B981" : "#EF4444"}` : "1px solid rgba(255,255,255,0.08)" }}>
                        {d.charAt(0).toUpperCase() + d.slice(1)}
                      </button>
                    );
                  })}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 6 }}>Threshold <span style={{ color: "#6B7280" }}>(predicted: {preds?.[probStat]})</span></div>
                <input type="number" step="0.5" placeholder={preds ? String(preds[probStat]) : "e.g. 26"} value={probThresh} onChange={(e) => { setProbThresh(e.target.value); setProbResult(null); }} onKeyDown={(e) => e.key === "Enter" && computeProb()} style={{ ...selectStyle, width: "100%", boxSizing: "border-box" }} />
              </div>
              <button onClick={computeProb} disabled={probLoad || !probThresh} style={{ padding: "10px 20px", fontSize: 13, fontWeight: 600, borderRadius: 10, border: "1px solid rgba(255,255,255,0.08)", background: probLoad || !probThresh ? "rgba(255,255,255,0.05)" : "#1F2937", color: probLoad || !probThresh ? "#6B7280" : "#fff", cursor: probLoad || !probThresh ? "not-allowed" : "pointer", whiteSpace: "nowrap" }}>
                {probLoad ? "…" : "Calculate →"}
              </button>
            </div>
            {probError && <div style={{ marginTop: 12 }}><ErrorBox onRetry={computeProb}>{probError}</ErrorBox></div>}
            {probResult && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 22, paddingTop: 18, borderTop: "1px solid rgba(255,255,255,0.08)" }}>
                <ProbGauge prob={probResult.probability} caption={`chance of ${probResult.direction} ${probResult.threshold} ${(STAT_LABELS[probResult.stat] || probResult.stat).toLowerCase()}`} />
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[["Probability", probResult.pct_display], ["Empirical rate", `${Math.round(probResult.hit_rate * 100)}%`], ["Normal dist", `${Math.round(probResult.normal_prob * 100)}%`], ["Recent average", `${probResult.stat_mean}`], ["Std deviation", `±${probResult.stat_std}`], ["Games sampled", `${probResult.sample_size}`], ["Data source", probResult.data_source || "training data"]].map(([label, value]) => (
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
    </div>
  );
}

const selectStyle = {
  width: "100%",
  padding: "9px 10px",
  borderRadius: 10,
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.08)",
  color: "var(--color-text-primary, #fff)",
  fontSize: 13,
};
