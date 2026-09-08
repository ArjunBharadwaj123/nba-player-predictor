"""No game -> no prediction (both the resolver returning None and unknown players)."""
import pytest

from conftest import requires_artifacts


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@requires_artifacts
def test_predict_404_when_no_upcoming_game(client, monkeypatch):
    # Force the schedule resolver to find no upcoming game.
    import nfl.scraping.current_context as cc
    monkeypatch.setattr(cc, "get_next_game_context", lambda pid, today=None: None)
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.post("/nfl/predict", json={"player_id": qb["id"], "scoring_format": "ppr"})
    assert r.status_code == 404
    assert "no upcoming game" in r.json()["detail"].lower()


@requires_artifacts
def test_predict_ok_with_explicit_context(client):
    # An explicit opponent bypasses schedule resolution -> prediction proceeds.
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.post("/nfl/predict", json={
        "player_id": qb["id"], "scoring_format": "ppr",
        "context": {"opponent_team": "KC", "home_away": 1, "game_total": 47}})
    assert r.status_code == 200
    assert "passing_yards" in r.json()["predictions"]


def test_predict_unknown_player_404(client):
    assert client.post("/nfl/predict", json={"player_id": "nope"}).status_code == 404
