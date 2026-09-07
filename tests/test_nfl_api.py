"""NFL API routes, response serialization, and position-specific validation.
Requires built artifacts (skipped otherwise)."""
import json
import math

import pytest

from conftest import requires_artifacts


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def _assert_json_safe(obj):
    """No NaN/Infinity survive serialization."""
    text = json.dumps(obj)
    assert "NaN" not in text and "Infinity" not in text


@requires_artifacts
def test_health(client):
    r = client.get("/nfl/health")
    assert r.status_code == 200
    assert r.json()["supported_positions"] == ["QB", "RB", "WR", "TE", "K"]


@requires_artifacts
def test_players_filter(client):
    r = client.get("/nfl/players", params={"position": "QB"})
    assert r.status_code == 200
    players = r.json()["players"]
    assert all(p["position"] == "QB" for p in players)
    assert len(players) > 0


@requires_artifacts
def test_players_invalid_filter_422(client):
    assert client.get("/nfl/players", params={"position": "P"}).status_code == 422


@requires_artifacts
def test_predict_serialization_and_shape(client):
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.post("/nfl/predict", json={"player_id": qb["id"], "scoring_format": "ppr"})
    assert r.status_code == 200
    body = r.json()
    _assert_json_safe(body)
    assert "passing_yards" in body["predictions"]
    assert body["scoring_format"] == "ppr"
    # every prediction is a finite number
    for v in body["predictions"].values():
        assert isinstance(v, (int, float)) and math.isfinite(v)


@requires_artifacts
def test_predict_invalid_scoring_format_422(client):
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.post("/nfl/predict", json={"player_id": qb["id"], "scoring_format": "super"})
    assert r.status_code == 422


@requires_artifacts
def test_probability_rejects_wrong_position_stat(client):
    wr = client.get("/nfl/players", params={"position": "WR"}).json()["players"][0]
    # passing_yards does not apply to a WR
    r = client.get("/nfl/probability", params={
        "player_id": wr["id"], "stat": "passing_yards", "threshold": 200})
    assert r.status_code == 422


@requires_artifacts
def test_probability_valid_position_stat(client):
    wr = client.get("/nfl/players", params={"position": "WR"}).json()["players"][0]
    r = client.get("/nfl/probability", params={
        "player_id": wr["id"], "stat": "receiving_yards", "threshold": 60})
    assert r.status_code == 200
    assert 0 <= r.json()["probability"] <= 1


@requires_artifacts
def test_fantasy_probability_reports_scoring_format(client):
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.get("/nfl/probability", params={
        "player_id": qb["id"], "stat": "fantasy_points", "threshold": 18,
        "scoring_format": "half_ppr"})
    assert r.status_code == 200
    assert r.json()["scoring_format"] == "half_ppr"


@requires_artifacts
def test_score_endpoint_recomputes_without_models(client):
    # /score only takes component predictions -> proves no model rerun needed.
    payload = {"position": "WR", "scoring_format": "ppr",
               "predictions": {"receiving_yards": 100, "receptions": 6,
                               "receiving_tds": 1, "targets": 9}}
    ppr = client.post("/nfl/score", json=payload).json()
    payload["scoring_format"] = "no_ppr"
    noppr = client.post("/nfl/score", json=payload).json()
    assert ppr["point_estimate"] == pytest.approx(noppr["point_estimate"] + 6.0)


@requires_artifacts
def test_score_invalid_format_422(client):
    r = client.post("/nfl/score", json={"position": "WR", "scoring_format": "x",
                                        "predictions": {}})
    assert r.status_code == 422


@requires_artifacts
def test_recent_endpoint(client):
    qb = client.get("/nfl/players", params={"position": "QB"}).json()["players"][0]
    r = client.get(f"/nfl/players/{qb['id']}/recent", params={"n": 3})
    assert r.status_code == 200
    assert "recent_games" in r.json()


@requires_artifacts
def test_unknown_player_404(client):
    assert client.post("/nfl/predict", json={"player_id": "nope"}).status_code == 404
