"""Tests for the OpenEnv FastAPI app.

The OpenEnv create_app framework handles HTTP/WebSocket endpoints.
We verify the app imports correctly and the health endpoint works.
"""

from fastapi.testclient import TestClient

from patronet.app import app


class TestOpenEnvApp:
  def test_health_returns_healthy(self):
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

  def test_schema_endpoint_exists(self):
    client = TestClient(app)
    response = client.get("/schema")
    assert response.status_code == 200

  def test_reset_endpoint_exists(self):
    client = TestClient(app)
    response = client.post("/reset")
    assert response.status_code == 200
