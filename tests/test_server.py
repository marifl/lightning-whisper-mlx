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


def _noop_tts(*args, **kwargs):
    """Stub that marks the TTS job as completed with a dummy file path."""
    from lightning_whisper_mlx.server import _jobs, _jobs_lock, JobStatus

    job_id = args[0]
    # Create a minimal WAV file at the expected path
    import wave
    audio_path = args[1]
    with wave.open(audio_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)  # 0.1s silence
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["result"] = {"audio_path": audio_path}


@patch("lightning_whisper_mlx.server._run_tts", _noop_tts)
def test_tts_creates_job(client):
    """POST /api/tts with text returns a job_id and queued status."""
    resp = client.post("/api/tts", data={"text": "Hello world"})
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_tts_rejects_empty_text(client):
    """POST /api/tts with empty text returns 422."""
    resp = client.post("/api/tts", data={"text": ""})
    assert resp.status_code == 422


def test_tts_rejects_missing_text(client):
    """POST /api/tts without text field returns 422."""
    resp = client.post("/api/tts")
    assert resp.status_code == 422


@patch("lightning_whisper_mlx.server._run_tts", _noop_tts)
def test_tts_audio_download(client):
    """GET /api/tts-jobs/{id}/audio returns WAV file after job completes."""
    create_resp = client.post("/api/tts", data={"text": "Test audio"})
    job_id = create_resp.json()["job_id"]

    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.json()["status"] == "completed"

    audio_resp = client.get(f"/api/tts-jobs/{job_id}/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"] == "audio/wav"
    assert len(audio_resp.content) > 0


def test_tts_audio_not_found(client):
    """GET /api/tts-jobs/{id}/audio with invalid ID returns 404."""
    resp = client.get("/api/tts-jobs/nonexistent-id/audio")
    assert resp.status_code == 404


# --- Speaker API tests ---

def _make_wav_bytes() -> bytes:
    """Minimal valid WAV header."""
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def tmp_speakers(tmp_path, monkeypatch):
    """Redirect speaker storage to temp dir."""
    monkeypatch.setattr("lightning_whisper_mlx.speakers.SPEAKERS_DIR", tmp_path)
    return tmp_path


def test_list_speakers_empty(client):
    """GET /api/speakers returns empty list initially."""
    resp = client.get("/api/speakers")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_speaker(client):
    """POST /api/speakers creates a speaker and returns metadata."""
    resp = client.post(
        "/api/speakers",
        files={"ref_audio": ("voice.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Alice", "ref_text": "Hello, I am Alice."},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Alice"
    assert "id" in data


def test_create_and_list_speakers(client):
    """Created speakers appear in list."""
    client.post(
        "/api/speakers",
        files={"ref_audio": ("voice.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Alice", "ref_text": "ref A"},
    )
    resp = client.get("/api/speakers")
    assert len(resp.json()) == 1
    assert resp.json()[0]["name"] == "Alice"


def test_delete_speaker(client):
    """DELETE /api/speakers/{id} removes the speaker."""
    create_resp = client.post(
        "/api/speakers",
        files={"ref_audio": ("voice.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Alice", "ref_text": "ref A"},
    )
    sid = create_resp.json()["id"]
    del_resp = client.delete(f"/api/speakers/{sid}")
    assert del_resp.status_code == 200

    resp = client.get("/api/speakers")
    assert resp.json() == []


def test_delete_speaker_not_found(client):
    """DELETE /api/speakers/{id} with bad ID returns 404."""
    resp = client.delete("/api/speakers/nonexistent")
    assert resp.status_code == 404


# --- Dialog TTS tests ---

import json as json_mod


def _noop_dialog_tts(*args, **kwargs):
    """Stub that marks the dialog TTS job as completed with a dummy WAV."""
    from lightning_whisper_mlx.server import _jobs, _jobs_lock, JobStatus
    import wave

    job_id = args[0]
    audio_path = args[1]
    with wave.open(audio_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 2400)
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["result"] = {"audio_path": audio_path}


@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_creates_job(client):
    """POST /api/tts/dialog with segments returns a job_id."""
    create_resp = client.post(
        "/api/speakers",
        files={"ref_audio": ("v.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Alice", "ref_text": "Hello."},
    )
    sid = create_resp.json()["id"]

    resp = client.post(
        "/api/tts/dialog",
        content=json_mod.dumps({
            "segments": [{"speaker_id": sid, "text": "Hello world."}],
        }),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_rejects_empty_segments(client):
    """POST /api/tts/dialog with empty segments returns 422."""
    resp = client.post(
        "/api/tts/dialog",
        content=json_mod.dumps({"segments": []}),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_rejects_missing_speaker(client):
    """POST /api/tts/dialog with nonexistent speaker_id returns 422."""
    resp = client.post(
        "/api/tts/dialog",
        content=json_mod.dumps({
            "segments": [{"speaker_id": "nonexistent", "text": "Hello."}],
        }),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_full_flow(client):
    """Full flow: create speakers -> submit dialog -> poll -> download audio."""
    # Create two speakers
    s1 = client.post(
        "/api/speakers",
        files={"ref_audio": ("v.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Alice", "ref_text": "Hello."},
    ).json()
    s2 = client.post(
        "/api/speakers",
        files={"ref_audio": ("v.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
        data={"name": "Bob", "ref_text": "Hi."},
    ).json()

    # Submit dialog
    resp = client.post(
        "/api/tts/dialog",
        content=json_mod.dumps({
            "segments": [
                {"speaker_id": s1["id"], "text": "[warmly] Hello Bob!"},
                {"speaker_id": s2["id"], "text": "[curious] Hey Alice."},
            ],
            "steps": 4,
            "pause_between_ms": 200,
        }),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll — should complete (mocked)
    job_resp = client.get(f"/api/jobs/{job_id}")
    assert job_resp.json()["status"] == "completed"

    # Download audio
    audio_resp = client.get(f"/api/tts-jobs/{job_id}/audio")
    assert audio_resp.status_code == 200
    assert audio_resp.headers["content-type"] == "audio/wav"
