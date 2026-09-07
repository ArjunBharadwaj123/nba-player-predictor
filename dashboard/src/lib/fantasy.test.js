import { describe, it, expect, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { fantasyPoints, RECEPTION_MULTIPLIERS, SCORING_FORMAT_LABELS } from "./fantasy";
import { useLocalStorage } from "./useLocalStorage";

describe("client fantasy scoring", () => {
  it("one reception adds 1 point under PPR", () => {
    expect(fantasyPoints({ receptions: 1 }, "WR", "ppr")).toBeCloseTo(1.0);
  });
  it("one reception adds 0.5 points under Half PPR", () => {
    expect(fantasyPoints({ receptions: 1 }, "WR", "half_ppr")).toBeCloseTo(0.5);
  });
  it("one reception adds 0 points under No PPR", () => {
    expect(fantasyPoints({ receptions: 1 }, "WR", "no_ppr")).toBeCloseTo(0.0);
  });
  it("identical components score correctly under every format", () => {
    const s = { receiving_yards: 100, receiving_tds: 1, receptions: 6 };
    expect(fantasyPoints(s, "WR", "no_ppr")).toBeCloseTo(16.0);
    expect(fantasyPoints(s, "WR", "half_ppr")).toBeCloseTo(19.0);
    expect(fantasyPoints(s, "WR", "ppr")).toBeCloseTo(22.0);
  });
  it("kicker points are unchanged across formats", () => {
    const k = { field_goals_made: 2, extra_points_made: 3 };
    const ppr = fantasyPoints(k, "K", "ppr");
    expect(fantasyPoints(k, "K", "half_ppr")).toBe(ppr);
    expect(fantasyPoints(k, "K", "no_ppr")).toBe(ppr);
  });
  it("exposes exact reception multipliers + labels", () => {
    expect(RECEPTION_MULTIPLIERS).toEqual({ ppr: 1.0, half_ppr: 0.5, no_ppr: 0.0 });
    expect(SCORING_FORMAT_LABELS.ppr).toBe("PPR");
  });
});

describe("scoring-format persistence (useLocalStorage)", () => {
  beforeEach(() => window.localStorage.clear());

  it("defaults to PPR when nothing is stored", () => {
    const { result } = renderHook(() => useLocalStorage("nfl_scoring_format", "ppr"));
    expect(result.current[0]).toBe("ppr");
  });

  it("persists the selected format to localStorage", () => {
    const { result } = renderHook(() => useLocalStorage("nfl_scoring_format", "ppr"));
    act(() => result.current[1]("half_ppr"));
    expect(result.current[0]).toBe("half_ppr");
    expect(JSON.parse(window.localStorage.getItem("nfl_scoring_format"))).toBe("half_ppr");
  });

  it("restores the saved preference on remount", () => {
    window.localStorage.setItem("nfl_scoring_format", JSON.stringify("no_ppr"));
    const { result } = renderHook(() => useLocalStorage("nfl_scoring_format", "ppr"));
    expect(result.current[0]).toBe("no_ppr");
  });
});
