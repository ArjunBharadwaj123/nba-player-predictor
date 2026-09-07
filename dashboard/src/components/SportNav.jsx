import { Link, useLocation } from "react-router-dom";
import { NFL_GREEN } from "../lib/nflConfig";

// Compact segmented control to switch between the NBA and NFL predictors. Each
// sport keeps its own accent when active (NBA blue, NFL field-green).
const TABS = [
  { to: "/nba", label: "🏀 NBA", accent: "#185FA5" },
  { to: "/nfl", label: "🏈 NFL", accent: NFL_GREEN },
];

export default function SportNav() {
  const { pathname } = useLocation();
  return (
    <nav
      aria-label="Sport"
      style={{
        display: "inline-flex",
        gap: 4,
        padding: 4,
        borderRadius: 12,
        background: "rgba(255,255,255,0.04)",
        border: "1px solid var(--color-border-secondary, #2e303a)",
      }}
    >
      {TABS.map(({ to, label, accent }) => {
        const active = pathname === to || (to === "/nba" && pathname === "/");
        return (
          <Link
            key={to}
            to={to}
            aria-current={active ? "page" : undefined}
            style={{
              textDecoration: "none",
              fontSize: 13,
              fontWeight: 600,
              padding: "7px 16px",
              borderRadius: 9,
              color: active ? "#fff" : "var(--color-text-secondary, #9ca3af)",
              background: active ? accent : "transparent",
              transition: "all 0.15s ease",
            }}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
