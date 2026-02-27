"""Tests for the FastAPI server module."""
import io
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from lightning_whisper_mlx.server import app


@pytest.fixture
def client():
    return TestClient(app)


def _noop_transcription(*args, **kwargs):
    """Stub that marks the job as completed with dummy data."""
    from lightning_whisper_mlx.server import _jobs, _jobs_lock, JobStatus

    job_id = args[0]
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["result"] = {
            "text": "test transcription",
            "segments": [],
            "language": "en",
        }


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


@patch("lightning_whisper_mlx.server._run_transcription", _noop_transcription)
def test_transcribe_creates_job(client):
    """POST /api/transcribe with an audio file returns a job_id and queued status."""
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


@patch("lightning_whisper_mlx.server._run_transcription", _noop_transcription)
def test_transcribe_accepts_base_quant(client):
    """POST /api/transcribe normalizes quant='base' to None."""
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "tiny", "quant": "base"},
    )
    assert resp.status_code == 202


@patch("lightning_whisper_mlx.server._run_transcription", _noop_transcription)
def test_get_job_returns_status(client):
    """GET /api/jobs/{id} returns job status after creation."""
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    create_resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "tiny"},
    )
    job_id = create_resp.json()["job_id"]

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("queued", "processing", "completed", "failed")


def test_get_job_not_found(client):
    """GET /api/jobs/{id} with invalid ID returns 404."""
    resp = client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404
