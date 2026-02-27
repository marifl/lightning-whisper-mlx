"""Tests for the FastAPI server module."""
import pytest
from fastapi.testclient import TestClient

from lightning_whisper_mlx.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_models(client):
    """GET /api/models returns all available models with their quantization options."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    # Must contain all 11 models from lightning.py
    assert "tiny" in data
    assert "distil-large-v3" in data
    # Standard models have base + 4bit + 8bit
    assert set(data["tiny"]) == {"base", "4bit", "8bit"}
    # Distilled models only have base
    assert set(data["distil-small.en"]) == {"base"}
