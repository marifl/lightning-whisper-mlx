"""FastAPI REST API for lightning-whisper-mlx.

Run with: uvicorn lightning_whisper_mlx.server:app --reload
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

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
