"""Tests for the FastAPI server module."""
import io

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


def test_transcribe_creates_job(client):
    """POST /api/transcribe with an audio file returns a job_id and queued status."""
    # Create a tiny valid WAV file (44 bytes -- header only, no samples)
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "tiny"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_transcribe_rejects_invalid_model(client):
    """POST /api/transcribe with invalid model name returns 422."""
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "nonexistent-model"},
    )
    assert resp.status_code == 422
