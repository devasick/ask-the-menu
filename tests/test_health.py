from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_ok(pg_url: str) -> None:
    with TestClient(create_app(Settings(_env_file=None, database_url=pg_url))) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "db": "ok"}


def test_health_reports_unavailable_db() -> None:
    url = "postgresql+asyncpg://nobody:nothing@127.0.0.1:1/none"
    with TestClient(create_app(Settings(_env_file=None, database_url=url))) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["db"] == "unavailable"


def test_cors_only_allows_configured_origins() -> None:
    settings = Settings(_env_file=None, cors_origins=["https://menunow.de"])
    with TestClient(create_app(settings)) as client:
        ok = client.options(
            "/health",
            headers={"Origin": "https://menunow.de", "Access-Control-Request-Method": "GET"},
        )
        bad = client.options(
            "/health",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
    assert ok.headers.get("access-control-allow-origin") == "https://menunow.de"
    assert "access-control-allow-origin" not in bad.headers
