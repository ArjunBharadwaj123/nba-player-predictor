"""Existing NBA endpoints remain functional after the multi-sport refactor."""
import pytest

ROOT_MODELS = None


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def _nba_ready():
    from pathlib import Path
    root = Path(__file__).parent.parent
    return (root / "nba" / "models" / "saved" / "pts_model.pkl").exists()


nba = pytest.mark.skipif(not _nba_ready(), reason="NBA model artifacts missing")


@nba
def test_nba_health_still_at_root(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "models_loaded" in r.json()


@nba
def test_nba_players_still_at_root(client):
    r = client.get("/players")
    assert r.status_code == 200
    assert "players" in r.json()


@nba
def test_nba_and_nfl_coexist(client):
    # Both roots respond; NFL is namespaced under /nfl, NBA stays at root.
    assert client.get("/health").status_code == 200
    assert client.get("/nfl/health").status_code in (200, 503)


@nba
def test_nba_probability_validation_unchanged(client):
    r = client.get("/probability", params={
        "stat": "not_a_stat", "threshold": 10, "direction": "over",
        "player_name": "x"})
    assert r.status_code == 422
