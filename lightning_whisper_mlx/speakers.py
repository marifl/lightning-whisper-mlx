"""Speaker library: CRUD for reference-audio speaker profiles."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPEAKERS_DIR = Path("./speakers")


def _ensure_dir() -> None:
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)


def create_speaker(name: str, audio_bytes: bytes, ref_text: str) -> dict[str, Any]:
    """Create a new speaker profile. Returns metadata dict."""
    _ensure_dir()
    sid = str(uuid.uuid4())
    speaker_dir = SPEAKERS_DIR / sid
    speaker_dir.mkdir()

    (speaker_dir / "ref_audio.wav").write_bytes(audio_bytes)

    meta = {
        "id": sid,
        "name": name,
        "ref_text": ref_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (speaker_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def list_speakers() -> list[dict[str, Any]]:
    """List all speaker profiles."""
    _ensure_dir()
    speakers = []
    for meta_path in sorted(SPEAKERS_DIR.glob("*/metadata.json")):
        speakers.append(json.loads(meta_path.read_text()))
    return speakers


def get_speaker(speaker_id: str) -> dict[str, Any] | None:
    """Get a single speaker by ID, or None if not found."""
    meta_path = SPEAKERS_DIR / speaker_id / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def get_speaker_audio_path(speaker_id: str) -> Path | None:
    """Get the path to a speaker's reference audio, or None."""
    audio_path = SPEAKERS_DIR / speaker_id / "ref_audio.wav"
    return audio_path if audio_path.exists() else None


def delete_speaker(speaker_id: str) -> bool:
    """Delete a speaker profile. Returns True if deleted, False if not found."""
    speaker_dir = SPEAKERS_DIR / speaker_id
    if not speaker_dir.exists():
        return False
    shutil.rmtree(speaker_dir)
    return True
