import { useState, useRef, useMemo } from "react";
import { PosBadge } from "./ui";

// Generic, keyboard-accessible player picker shared by the NBA and NFL pages.
// Props:
//   players         : array of { id, name, [positionKey], team?, headshot_url? }
//   selectedPlayer  : currently selected player object (or null)
//   onSelect        : (player | null) => void
//   positionKey     : field holding the position ("pos" for NBA, "position" NFL)
//   placeholder     : input placeholder
//   colors/textColors: position -> color maps for the badge
//   showTeam        : render the player's team (disambiguates duplicate names)
//   groupByPosition : group the dropdown results under position headers
export default function PlayerSearch({
  players,
  selectedPlayer,
  onSelect,
  positionKey = "pos",
  placeholder = "Search players…",
  colors = {},
  textColors = {},
  showTeam = false,
  groupByPosition = false,
  accent = "#185FA5",
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(0);
  const inputRef = useRef(null);

  const results = useMemo(() => {
    if (query.trim().length < 1) return [];
    const q = query.toLowerCase();
    const matched = players.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        (showTeam && (p.team || "").toLowerCase().includes(q)),
    );
    if (!groupByPosition) return matched.slice(0, 10);
    // Group by position, but keep one flat ordered array for keyboard nav.
    const order = ["QB", "RB", "WR", "TE", "K"];
    const flat = [];
    order.forEach((pos) => {
      matched
        .filter((p) => p[positionKey] === pos)
        .forEach((p) => flat.push(p));
    });
    matched.filter((p) => !order.includes(p[positionKey])).forEach((p) => flat.push(p));
    return flat.slice(0, 12);
  }, [query, players, positionKey, showTeam, groupByPosition]);

  // Keep the highlighted index within range as results shrink/grow.
  const activeIndex = Math.min(focused, Math.max(results.length - 1, 0));

  const onQueryChange = (v) => {
    setQuery(v);
    setFocused(0);
    setOpen(true);
  };

  const choose = (p) => {
    onSelect(p);
    setQuery("");
    setOpen(false);
  };

  const onKey = (e) => {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocused((f) => Math.min(f + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocused((f) => Math.max(f - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(results[activeIndex]);
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery("");
    }
  };

  return (
    <div style={{ position: "relative" }}>
      {selectedPlayer && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginBottom: 10,
            padding: "10px 14px",
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 12,
          }}
        >
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{ fontSize: 15, fontWeight: 500 }}>{selectedPlayer.name}</div>
              <PosBadge pos={selectedPlayer[positionKey]} colors={colors} textColors={textColors} />
              {showTeam && selectedPlayer.team ? (
                <span style={{ fontSize: 12, color: "var(--color-text-tertiary, #6b7280)" }}>
                  {selectedPlayer.team}
                </span>
              ) : null}
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
              cursor: "pointer",
            }}
          >
            Change
          </button>
        </div>
      )}

      {!selectedPlayer && (
        <div style={{ position: "relative" }}>
          <input
            ref={inputRef}
            type="text"
            role="combobox"
            aria-expanded={open}
            aria-controls="player-listbox"
            aria-autocomplete="list"
            placeholder={placeholder}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onFocus={() => setOpen(true)}
            onBlur={() => setTimeout(() => setOpen(false), 150)}
            onKeyDown={onKey}
            style={{
              width: "100%",
              boxSizing: "border-box",
              fontSize: 14,
              padding: "10px 12px 10px 36px",
              borderRadius: 12,
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "var(--color-text-primary, #fff)",
              outline: "none",
            }}
          />
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", opacity: 0.5 }}
            fill="none"
            aria-hidden="true"
          >
            <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" strokeWidth="1.5" />
            <path d="M10.5 10.5L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </div>
      )}

      {open && results.length > 0 && (
        <div
          id="player-listbox"
          role="listbox"
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 100,
            marginTop: 6,
            background: "var(--color-background-primary, #1b1c22)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: 12,
            boxShadow: "0 10px 30px rgba(0,0,0,0.5)",
            overflow: "hidden",
            maxHeight: 320,
            overflowY: "auto",
          }}
        >
          {results.map((p, i) => {
            const isFocused = i === activeIndex;
            const showHeader =
              groupByPosition && (i === 0 || results[i - 1][positionKey] !== p[positionKey]);
            return (
              <div key={p.id}>
                {showHeader && (
                  <div
                    style={{
                      padding: "6px 14px 2px",
                      fontSize: 10,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "var(--color-text-tertiary, #6b7280)",
                    }}
                  >
                    {p[positionKey]}
                  </div>
                )}
                <div
                  role="option"
                  aria-selected={isFocused}
                  tabIndex={-1}
                  onMouseDown={() => choose(p)}
                  onMouseEnter={() => setFocused(i)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "9px 14px",
                    cursor: "pointer",
                    background: isFocused ? "rgba(255,255,255,0.08)" : "transparent",
                    borderLeft: isFocused ? `2px solid ${accent}` : "2px solid transparent",
                  }}
                >
                  <span style={{ flex: 1, fontSize: 14 }}>{p.name}</span>
                  {showTeam && p.team ? (
                    <span style={{ fontSize: 12, color: "var(--color-text-tertiary, #6b7280)" }}>{p.team}</span>
                  ) : null}
                  <PosBadge pos={p[positionKey]} colors={colors} textColors={textColors} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {open && query.trim().length >= 1 && results.length === 0 && (
        <div
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            zIndex: 100,
            marginTop: 6,
            padding: "12px 14px",
            borderRadius: 12,
            background: "var(--color-background-primary, #1b1c22)",
            border: "1px solid rgba(255,255,255,0.08)",
            fontSize: 13,
            color: "#9CA3AF",
          }}
        >
          No players found for &quot;{query}&quot;
        </div>
      )}
    </div>
  );
}
