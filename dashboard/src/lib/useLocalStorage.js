import { useState, useCallback } from "react";

// Persist a small piece of UI state in localStorage. Wrapped in try/catch so a
// private window / blocked storage never crashes the page — it just falls back
// to in-memory state. Used for the NFL PPR scoring preference (isolated from the
// NBA page by its own key).
export function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored === null ? initialValue : JSON.parse(stored);
    } catch {
      return initialValue;
    }
  });

  const set = useCallback(
    (next) => {
      setValue((prev) => {
        const resolved = typeof next === "function" ? next(prev) : next;
        try {
          window.localStorage.setItem(key, JSON.stringify(resolved));
        } catch {
          /* storage unavailable — keep in-memory value only */
        }
        return resolved;
      });
    },
    [key],
  );

  return [value, set];
}
