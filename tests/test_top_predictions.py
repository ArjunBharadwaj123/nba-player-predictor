"""Combined Top-10 ranking/order + limit."""
import analytics.top_predictions as tp


def test_build_top_predictions_orders_and_limits(monkeypatch):
    fake = [
        {"sport": "NFL", "player": "A", "stat": "x", "projection": 1, "baseline": 0,
         "confidence_score": 0.5},
        {"sport": "NFL", "player": "B", "stat": "x", "projection": 1, "baseline": 0,
         "confidence_score": 2.1},
        {"sport": "NBA", "player": "C", "stat": "y", "projection": 1, "baseline": 0,
         "confidence_score": 1.4},
    ]
    monkeypatch.setattr(tp, "nfl_top", lambda today: fake)
    monkeypatch.setattr(tp, "nba_top", lambda today: [])
    out = tp.build_top_predictions(limit=2)
    assert out["count"] == 2
    # Sorted by confidence_score desc.
    assert [p["player"] for p in out["predictions"]] == ["B", "C"]
    assert out["predictions"][0]["confidence_score"] >= out["predictions"][1]["confidence_score"]


def test_empty_when_no_games(monkeypatch):
    monkeypatch.setattr(tp, "nfl_top", lambda today: [])
    monkeypatch.setattr(tp, "nba_top", lambda today: [])
    out = tp.build_top_predictions()
    assert out["count"] == 0
    assert out["predictions"] == []


def test_nba_top_empty_without_nba_api(monkeypatch):
    # nba_api isn't installed in CI -> NBA yields no fabricated picks.
    from datetime import date
    assert tp.nba_top(date(2026, 9, 7)) == []
