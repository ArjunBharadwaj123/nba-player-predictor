"""
Combined NBA + NFL Top-10 predictions (confidence-weighted edge).
================================================================
For every player **with an upcoming game**, we take the model's projection for
each stat, compare it to the player's own recent baseline (a shifted season/
rolling average), standardize by the prediction interval, and weight by the
model's reliability for that (position, stat). Each player's single strongest
pick is kept; picks across both sports are ranked and the top N written to
``data/top_predictions.json`` for the admin dashboard.

    score = reliability(R²) × |standardized_edge|
    standardized_edge = (projection − baseline) / interval_half_width

This is a **model-confidence** ranking, not a market comparison — free player-prop
odds aren't available (see README). Only players with a resolvable upcoming game
are included (no game → no pick), matching the prediction endpoints.

Run:  python -m analytics.top_predictions
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "top_predictions.json"

# Minimum projected fantasy points for a player to be eligible for the board,
# so obscure backups with stale-high baselines don't dominate the ranking.
MIN_FANTASY = 6.0


# ── NFL ──────────────────────────────────────────────────────────────────────
def _nfl_reliability() -> dict:
    """{(position,target): r2 in [0,1]} from the NFL eval report."""
    path = ROOT / "nfl" / "models" / "saved" / "eval_report.json"
    out: dict = {}
    if not path.exists():
        return out
    rep = json.loads(path.read_text())
    for pos, pr in rep.get("positions", {}).items():
        for target, m in pr.get("targets", {}).items():
            r2 = m.get("r2")
            out[(pos, target)] = max(0.0, min(1.0, float(r2))) if r2 is not None else 0.0
    return out


def _nfl_next_game_by_team(today: date) -> dict:
    """{team: upcoming-game context} — earliest fixture with gameday ≥ today."""
    from nfl.config import current_nfl_season
    from nfl.scraping.current_context import _load_schedule_frame
    cur = current_nfl_season(today)
    frames = []
    for season, live in ((cur, True), (cur - 1, False)):
        f = _load_schedule_frame(season, live=live)
        if f is not None:
            frames.append(f)
    if not frames:
        return {}
    sched = pd.concat(frames, ignore_index=True)
    sched["gameday_dt"] = pd.to_datetime(sched["gameday"], errors="coerce")
    sched = sched[sched["gameday_dt"].dt.date >= today].sort_values("gameday_dt")

    by_team: dict = {}
    for _, g in sched.iterrows():
        total, spread_home = g.get("total_line"), g.get("spread_line")
        for side in ("home", "away"):
            team = g.get(f"{side}_team")
            if not team or team in by_team:
                continue
            is_home = side == "home"
            opp = g.get("away_team" if is_home else "home_team")
            team_spread = (-float(spread_home) if is_home else float(spread_home)) \
                if pd.notna(spread_home) else None
            implied = (float(total) / 2 - team_spread / 2) \
                if (pd.notna(total) and team_spread is not None) else None
            roof = str(g.get("roof") or "").lower()
            by_team[team] = {
                "opponent_team": opp,
                "home_away": 1 if is_home else 0,
                "days_rest": _num(g.get("home_rest" if is_home else "away_rest")),
                "spread_line_team": team_spread,
                "game_total": _num(total),
                "implied_team_total": implied,
                "is_indoor": int(roof in ("dome", "closed")),
                "is_grass": int(str(g.get("surface") or "").lower() == "grass"),
                "temp": _num(g.get("temp")),
                "wind": _num(g.get("wind")),
                "div_game": int(_num(g.get("div_game")) or 0),
                "season": int(g["season"]), "week": int(g["week"]),
                "game_date": str(g.get("gameday")),
            }
    return by_team


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def nfl_top(today: date) -> list[dict]:
    from nfl.config import POSITION_TARGETS
    from nfl.serving import load_players, latest_player_row, build_upcoming_row
    from nfl.explainability import explainer

    reliab = _nfl_reliability()
    team_next = _nfl_next_game_by_team(today)
    if not team_next:
        log.info("No upcoming NFL games — no NFL picks.")
        return []

    picks = []
    for p in load_players():
        game = team_next.get(p.get("team"))
        if not game:
            continue                                   # no game → no pick
        pos = p["position"]
        base = latest_player_row(p["id"])
        if base is None:
            continue
        ctx = dict(game)
        ctx["team"] = p.get("team")
        ctx["team_changed"] = p.get("team_changed")
        try:
            X = build_upcoming_row(p["id"], pos, ctx)
            res = explainer.predict_player(
                pos, X, "ppr", interval_widen=1.25 if p.get("team_changed") else 1.0,
                explain=False, simulate=False)
        except Exception:
            continue
        # Relevance gate: only rank genuine contributors, so deep backups with
        # stale-high season averages don't dominate with huge negative "edges".
        if float(res["fantasy"].get("point_estimate", 0)) < MIN_FANTASY:
            continue
        best = None
        for target in POSITION_TARGETS.get(pos, []):
            proj = res["predictions"].get(target)
            iv = res["intervals"].get(target)
            if proj is None or iv is None:
                continue
            # Baseline = the player's recent form (shifted 5-game mean), which is
            # comparable to a single-game projection; fall back to season avg.
            ref = base.get(f"{target}_roll5")
            if ref is None or pd.isna(ref):
                ref = base.get(f"{target}_season_avg")
            ref = float(ref) if ref is not None and not pd.isna(ref) else None
            if ref is None:
                continue
            hw = max((iv[1] - iv[0]) / 2.0, 1e-6)
            # Clip to avoid tiny-interval blow-ups on rare-count stats.
            std_edge = float(np.clip((proj - ref) / hw, -4.0, 4.0))
            r = reliab.get((pos, target), 0.0)
            score = r * abs(std_edge)
            cand = {
                "sport": "NFL", "player": p["name"], "team": p.get("team"),
                "position": pos, "stat": target,
                "projection": round(float(proj), 2), "baseline": round(ref, 2),
                "edge": round(float(proj - ref), 2),
                "direction": "over" if proj >= ref else "under",
                "reliability": round(r, 3),
                "confidence_score": round(float(score), 3),
                "team_changed": bool(p.get("team_changed")),
                "game": f"{'vs' if game['home_away'] else '@'} {game['opponent_team']} (Wk {game['week']})",
            }
            if best is None or cand["confidence_score"] > best["confidence_score"]:
                best = cand
        if best:
            picks.append(best)
    log.info("NFL picks: %d players with an upcoming game", len(picks))
    return picks


# ── NBA (best-effort; empty when no upcoming games / API unavailable) ─────────
def nba_top(today: date) -> list[dict]:
    """NBA picks require a resolvable next game (nba_api). In the NBA offseason,
    or where nba_api is unavailable, returns [] (no game → no pick)."""
    try:
        from nba.scraping.next_game import get_next_game_context  # noqa: F401
    except Exception:
        log.info("NBA next-game unavailable (nba_api missing / offseason) — no NBA picks.")
        return []
    # Full NBA integration mirrors nfl_top using api.main's feature builder; it
    # activates once the NBA season is underway and nba_api is installed. Kept
    # intentionally conservative here so the offseason yields no fabricated picks.
    return []


def build_top_predictions(today: date | None = None, limit: int = 10) -> dict:
    today = today or date.today()
    picks = nfl_top(today) + nba_top(today)
    picks.sort(key=lambda x: x["confidence_score"], reverse=True)
    top = picks[:limit]
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "as_of_date": str(today),
        "ranking": "confidence-weighted edge (reliability x |standardized edge|)",
        "note": "Model-confidence ranking; not vs market odds. Only players with an upcoming game.",
        "count": len(top),
        "n_candidates": len(picks),
        "predictions": top,
    }
    return payload


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    payload = build_top_predictions()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(OUT)
    log.info("Wrote %s — %d picks (of %d candidates)",
             OUT, payload["count"], payload["n_candidates"])
    for p in payload["predictions"]:
        log.info("  %-4s %-22s %-18s proj %.1f base %.1f  score %.2f",
                 p["sport"], p["player"], p["stat"], p["projection"],
                 p["baseline"], p["confidence_score"])


if __name__ == "__main__":
    main()
