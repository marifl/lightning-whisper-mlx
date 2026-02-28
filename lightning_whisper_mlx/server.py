"""FastAPI REST API for lightning-whisper-mlx.

Run with: uvicorn lightning_whisper_mlx.server:app --reload
"""
from __future__ import annotations

import tempfile
import threading
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .lightning import models

app = FastAPI(title="Lightning Whisper MLX", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    """Return available models and their quantization options."""
    return {name: list(variants.keys()) for name, variants in models.items()}


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# In-memory job store (non-goal v1: no persistence)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _run_transcription(job_id: str, audio_path: str, model: str,
                        quant: str | None, batch_size: int,
                        diarize: bool, hf_token: str | None,
                        correct: bool, correct_backend: str | None,
                        anthropic_api_key: str | None) -> None:
    """Run transcription in a background thread."""
    import os
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.processing

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        from .lightning import LightningWhisperMLX
        whisper = LightningWhisperMLX(model, batch_size=batch_size, quant=quant)

        transcribe_kwargs: dict[str, Any] = {
            "diarize": diarize,
            "correct": correct,
        }
        # Only pass correct_backend when explicitly provided;
        # otherwise let transcribe() use its own default ("anthropic").
        if correct and correct_backend:
            transcribe_kwargs["correct_backend"] = correct_backend

        result = whisper.transcribe(audio_path, **transcribe_kwargs)

        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.completed
            _jobs[job_id]["result"] = result
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.failed
            _jobs[job_id]["error"] = str(e)
    finally:
        # Clean up temp file
        Path(audio_path).unlink(missing_ok=True)


def _run_tts(job_id: str, output_path: str, text: str,
             steps: int, speed: float, seed: int | None) -> None:
    """Run TTS generation in a background thread."""
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.processing

        from .tts import LightningTTSMLX
        tts = LightningTTSMLX()
        tts.generate(text=text, output_path=output_path, steps=steps, speed=speed, seed=seed)

        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.completed
            _jobs[job_id]["result"] = {"audio_path": output_path}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.failed
            _jobs[job_id]["error"] = str(e)
        Path(output_path).unlink(missing_ok=True)


@app.post("/api/transcribe", status_code=202)
async def transcribe(
    file: UploadFile,
    model: str = Form(default="distil-large-v3"),
    quant: str | None = Form(default=None),
    batch_size: int = Form(default=12),
    diarize: bool = Form(default=False),
    hf_token: str | None = Form(default=None),
    correct: bool = Form(default=False),
    correct_backend: str | None = Form(default=None),
    anthropic_api_key: str | None = Form(default=None),
) -> dict[str, str]:
    """Upload audio and start a transcription job."""
    # Normalize 'base' to no quantization for consistency with /api/models
    if quant == "base":
        quant = None

    # Validate model
    if model not in models:
        raise HTTPException(status_code=422, detail=f"Invalid model: {model}. Available: {list(models.keys())}")

    if quant and quant not in ("4bit", "8bit"):
        raise HTTPException(status_code=422, detail="quant must be '4bit', '8bit', or null")

    if quant and "distil" in model:
        raise HTTPException(status_code=422, detail=f"Quantization not supported for distilled model '{model}'")

    # Save uploaded file to temp location (streamed in chunks)
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": JobStatus.queued,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_transcription,
        args=(job_id, tmp_path, model, quant, batch_size,
              diarize, hf_token, correct, correct_backend, anthropic_api_key),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Poll job status and result."""
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": job["result"],
            "error": job["error"],
        }


@app.post("/api/tts", status_code=202)
async def text_to_speech(
    text: str = Form(),
    steps: int = Form(default=8),
    speed: float = Form(default=1.0),
    seed: int | None = Form(default=None),
) -> dict[str, str]:
    """Submit text for speech synthesis."""
    if not text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    job_id = str(uuid.uuid4())
    output_path = str(Path(tempfile.gettempdir()) / f"tts_{job_id}.wav")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": JobStatus.queued,
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_tts,
        args=(job_id, output_path, text, steps, speed, seed),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/tts-jobs/{job_id}/audio")
def get_tts_audio(job_id: str) -> FileResponse:
    """Download the generated audio file."""
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = _jobs[job_id]

    result = job.get("result")
    if not result or not result.get("audio_path"):
        raise HTTPException(status_code=404, detail="Audio not available")

    audio_path = Path(result["audio_path"])
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    cleanup = BackgroundTask(audio_path.unlink, missing_ok=True)
    return FileResponse(audio_path, media_type="audio/wav", filename=f"tts_{job_id}.wav",
                        background=cleanup)
