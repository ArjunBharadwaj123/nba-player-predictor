"""Admin API auth + stats shape. Requires built artifacts for /stats content."""
import os

import pytest

from conftest import requires_artifacts


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_verify_503_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    assert client.get("/admin/verify", headers={"X-Admin-Password": "x"}).status_code == 503


def test_verify_401_wrong_password(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    assert client.get("/admin/verify", headers={"X-Admin-Password": "nope"}).status_code == 401
    # Missing header also rejected.
    assert client.get("/admin/verify").status_code == 401


def test_verify_200_correct_password(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    r = client.get("/admin/verify", headers={"X-Admin-Password": "secret123"})
    assert r.status_code == 200 and r.json()["ok"] is True


@requires_artifacts
def test_stats_shape(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    r = client.get("/admin/stats", headers={"X-Admin-Password": "secret123"})
    assert r.status_code == 200
    body = r.json()
    assert "nfl" in body and "nba" in body
    assert body["nfl"]["sport"] == "NFL"
    assert isinstance(body["nfl"]["targets"], list) and len(body["nfl"]["targets"]) > 0


def test_top_predictions_requires_auth(client, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret123")
    assert client.get("/admin/top-predictions").status_code == 401
    r = client.get("/admin/top-predictions", headers={"X-Admin-Password": "secret123"})
    assert r.status_code == 200
    assert "predictions" in r.json()
