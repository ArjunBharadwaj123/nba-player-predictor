// Position-keyed display configuration for the NFL predictor.
// Which stats a position shows, their labels/order/colors/bar-maximums/units,
// decimal formatting, and the probability-calculator stat options. The page is
// driven entirely by this object, so a QB never renders receiving cards and a WR
// never renders passing cards.

export const NFL_GREEN = "#2F7D4F"; // restrained field-green accent for NFL
export const NFL_GREEN_SOFT = "rgba(47,125,79,0.16)";

// Per-stat presentation. `d` = decimal places, `max` = progress-bar ceiling.
const STAT = {
  attempts: { label: "Pass Attempts", color: "#2F7D4F", max: 50, d: 1, unit: "att" },
  completions: { label: "Completions", color: "#2F7D4F", max: 40, d: 1, unit: "cmp" },
  passing_yards: { label: "Passing Yards", color: "#1E6F45", max: 400, d: 0, unit: "yds" },
  passing_tds: { label: "Passing TDs", color: "#3C8A5C", max: 5, d: 2, unit: "td" },
  passing_interceptions: { label: "Interceptions", color: "#9B3D2E", max: 3, d: 2, unit: "int" },
  rushing_yards: { label: "Rushing Yards", color: "#8A5A0B", max: 150, d: 0, unit: "yds" },
  rushing_tds: { label: "Rushing TDs", color: "#8A5A0B", max: 3, d: 2, unit: "td" },
  carries: { label: "Carries", color: "#6B4A1F", max: 30, d: 1, unit: "att" },
  targets: { label: "Targets", color: "#185FA5", max: 16, d: 1, unit: "tgt" },
  receptions: { label: "Receptions", color: "#185FA5", max: 14, d: 1, unit: "rec" },
  receiving_yards: { label: "Receiving Yards", color: "#1E6F45", max: 150, d: 0, unit: "yds" },
  receiving_tds: { label: "Receiving TDs", color: "#3C8A5C", max: 3, d: 2, unit: "td" },
  field_goals_made: { label: "Field Goals Made", color: "#2F7D4F", max: 5, d: 1, unit: "fg" },
  extra_points_made: { label: "Extra Points", color: "#3C8A5C", max: 6, d: 1, unit: "xp" },
  kicking_points: { label: "Kicking Points", color: "#1E6F45", max: 20, d: 1, unit: "pts" },
};

// Ordered target lists per position (mirrors backend POSITION_TARGETS).
const ORDER = {
  QB: ["passing_yards", "passing_tds", "attempts", "completions",
       "passing_interceptions", "rushing_yards", "rushing_tds"],
  RB: ["rushing_yards", "carries", "rushing_tds", "receptions", "targets",
       "receiving_yards", "receiving_tds"],
  WR: ["receiving_yards", "receptions", "targets", "receiving_tds", "rushing_yards"],
  TE: ["receiving_yards", "receptions", "targets", "receiving_tds"],
  K: ["kicking_points", "field_goals_made", "extra_points_made"],
};

export const POSITION_CONFIG = Object.fromEntries(
  Object.entries(ORDER).map(([pos, keys]) => [
    pos,
    {
      order: keys,
      stats: Object.fromEntries(keys.map((k) => [k, STAT[k]])),
      // Probability calculator options: real stats + fantasy points.
      probabilityOptions: [...keys, "fantasy_points"],
    },
  ]),
);

export const POS_COLORS = {
  QB: "#E1F0E7", RB: "#F3EAD9", WR: "#E6F1FB", TE: "#EDE9FB", K: "#EAF3DE",
};
export const POS_TEXT = {
  QB: "#1E6F45", RB: "#6B4A1F", SF: "#0C447C", WR: "#0C447C", TE: "#3C3489", K: "#27500A",
};

export function fmt(value, key) {
  const cfg = STAT[key];
  const d = cfg ? cfg.d : 1;
  const n = typeof value === "number" ? value : parseFloat(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export function statMeta(key) {
  return STAT[key] || { label: key, color: NFL_GREEN, max: 100, d: 1, unit: "" };
}
