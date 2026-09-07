// Client-side fantasy scoring mirror of nfl/scoring.py.
// -----------------------------------------------------
// Used for INSTANT local recomputation of the fantasy-point total when the PPR
// format changes — this never reruns the ML models (the component stat
// predictions are unchanged; only the reception conversion differs). The
// simulated interval + over/under probability are refined by the /nfl/score
// endpoint, which likewise reuses the existing predictions.

export const RECEPTION_MULTIPLIERS = { ppr: 1.0, half_ppr: 0.5, no_ppr: 0.0 };
export const SCORING_FORMAT_LABELS = {
  ppr: "PPR",
  half_ppr: "Half PPR",
  no_ppr: "No PPR",
};
export const SCORING_FORMATS = ["ppr", "half_ppr", "no_ppr"];

const num = (stats, key) => {
  const v = stats?.[key];
  const f = typeof v === "number" ? v : parseFloat(v);
  return Number.isFinite(f) ? f : 0;
};

// Kicker scoring is independent of PPR format.
export function kickerFantasyPoints(stats) {
  const ep = num(stats, "extra_points_made") * 1.0;
  const hasBuckets =
    stats &&
    ["field_goals_made_0_39", "field_goals_made_40_49", "field_goals_made_50_plus"].some(
      (k) => stats[k] != null,
    );
  let fg;
  if (hasBuckets) {
    fg =
      num(stats, "field_goals_made_0_39") * 3.0 +
      num(stats, "field_goals_made_40_49") * 4.0 +
      num(stats, "field_goals_made_50_plus") * 5.0;
  } else {
    fg = num(stats, "field_goals_made") * 3.0; // documented simplified fallback
  }
  return ep + fg;
}

export function skillFantasyPoints(stats, format) {
  const mult = RECEPTION_MULTIPLIERS[format] ?? 1.0;
  let pts =
    num(stats, "passing_yards") * 0.04 +
    num(stats, "passing_tds") * 4.0 +
    num(stats, "passing_interceptions") * -2.0 +
    num(stats, "rushing_yards") * 0.1 +
    num(stats, "rushing_tds") * 6.0 +
    num(stats, "receiving_yards") * 0.1 +
    num(stats, "receiving_tds") * 6.0 +
    num(stats, "receptions") * mult;
  if (stats && stats.fumbles_lost != null) {
    pts += num(stats, "fumbles_lost") * -2.0;
  }
  return pts;
}

export function fantasyPoints(stats, position, format) {
  if (String(position).toUpperCase() === "K") return kickerFantasyPoints(stats);
  return skillFantasyPoints(stats, format);
}
