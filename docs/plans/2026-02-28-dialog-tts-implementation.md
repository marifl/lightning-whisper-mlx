# Multi-Speaker Dialog TTS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement lifelike multi-speaker dialogue TTS with ElevenLabs V3 format compatibility, speaker library management, and a dedicated dialog UI — all running locally on Apple Silicon via F5-TTS MLX.

**Architecture:** Server-side speaker library stored at `./speakers/{uuid}/` with CRUD API. Dialog TTS endpoint accepts segments with speaker references, strips ElevenLabs-style `[tags]`, chunks text at punctuation boundaries (max 135 UTF-8 bytes), generates audio per-chunk with per-speaker reference audio, inserts silence between speaker changes, and concatenates to a single WAV. Frontend has a speaker manager in Settings and a dedicated `/tts/dialog` route with visual editor + JSON import.

**Tech Stack:** Python/FastAPI (backend), Next.js/React/TypeScript (frontend), F5-TTS-MLX, ffmpeg, numpy, soundfile.

**Repos:**
- Backend: `/Users/marcusifland/prj/lightning-whisper-mlx/`
- Frontend: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/`

---

## Task 1: Speaker Storage Utilities

**Files:**
- Create: `lightning_whisper_mlx/speakers.py`
- Test: `tests/test_speakers.py`

**Step 1: Write the failing tests**

```python
# tests/test_speakers.py
"""Tests for speaker CRUD utilities."""
import json
from pathlib import Path

import pytest

from lightning_whisper_mlx.speakers import (
    create_speaker,
    list_speakers,
    get_speaker,
    delete_speaker,
    SPEAKERS_DIR,
)


@pytest.fixture(autouse=True)
def tmp_speakers(tmp_path, monkeypatch):
    """Point SPEAKERS_DIR to a temp directory for every test."""
    monkeypatch.setattr("lightning_whisper_mlx.speakers.SPEAKERS_DIR", tmp_path)
    return tmp_path


def _make_wav_bytes() -> bytes:
    """Minimal valid WAV header (16kHz mono, 0.1s silence)."""
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


class TestCreateSpeaker:
    def test_creates_directory_and_metadata(self, tmp_speakers):
        speaker = create_speaker("Alice", _make_wav_bytes(), "Hello, I am Alice.")
        assert speaker["name"] == "Alice"
        assert speaker["ref_text"] == "Hello, I am Alice."
        sid = speaker["id"]
        assert (tmp_speakers / sid / "metadata.json").exists()
        assert (tmp_speakers / sid / "ref_audio.wav").exists()

    def test_metadata_has_required_fields(self, tmp_speakers):
        speaker = create_speaker("Bob", _make_wav_bytes(), "Test text.")
        meta = json.loads((tmp_speakers / speaker["id"] / "metadata.json").read_text())
        assert set(meta.keys()) == {"id", "name", "ref_text", "created_at"}


class TestListSpeakers:
    def test_empty_when_no_speakers(self, tmp_speakers):
        assert list_speakers() == []

    def test_returns_all_speakers(self, tmp_speakers):
        create_speaker("Alice", _make_wav_bytes(), "ref A")
        create_speaker("Bob", _make_wav_bytes(), "ref B")
        speakers = list_speakers()
        assert len(speakers) == 2
        names = {s["name"] for s in speakers}
        assert names == {"Alice", "Bob"}


class TestGetSpeaker:
    def test_returns_speaker_by_id(self, tmp_speakers):
        created = create_speaker("Alice", _make_wav_bytes(), "ref text")
        speaker = get_speaker(created["id"])
        assert speaker["name"] == "Alice"

    def test_returns_none_for_missing(self, tmp_speakers):
        assert get_speaker("nonexistent") is None


class TestDeleteSpeaker:
    def test_deletes_speaker_directory(self, tmp_speakers):
        created = create_speaker("Alice", _make_wav_bytes(), "ref text")
        sid = created["id"]
        assert delete_speaker(sid) is True
        assert not (tmp_speakers / sid).exists()

    def test_returns_false_for_missing(self, tmp_speakers):
        assert delete_speaker("nonexistent") is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_speakers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lightning_whisper_mlx.speakers'`

**Step 3: Write minimal implementation**

```python
# lightning_whisper_mlx/speakers.py
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_speakers.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
git add lightning_whisper_mlx/speakers.py tests/test_speakers.py
git commit -m "feat: add speaker library CRUD utilities"
```

---

## Task 2: Speaker REST API Endpoints

**Files:**
- Modify: `lightning_whisper_mlx/server.py`
- Test: `tests/test_server.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_server.py::test_list_speakers_empty tests/test_server.py::test_create_speaker -v`
Expected: FAIL — 404 (endpoints don't exist yet)

**Step 3: Add Speaker API endpoints to server.py**

Add these endpoints to `lightning_whisper_mlx/server.py` (after the TTS endpoints):

```python
# --- Speaker API ---

from .speakers import create_speaker as _create_speaker, list_speakers as _list_speakers, delete_speaker as _delete_speaker


@app.get("/api/speakers")
def get_speakers() -> list[dict]:
    """List all speaker profiles."""
    return _list_speakers()


@app.post("/api/speakers", status_code=201)
async def create_speaker_endpoint(
    name: str = Form(),
    ref_text: str = Form(),
    ref_audio: UploadFile = ...,
) -> dict:
    """Create a new speaker profile with reference audio."""
    audio_bytes = await ref_audio.read()
    return _create_speaker(name, audio_bytes, ref_text)


@app.delete("/api/speakers/{speaker_id}")
def delete_speaker_endpoint(speaker_id: str) -> dict:
    """Delete a speaker profile."""
    if not _delete_speaker(speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": True}
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_server.py -v`
Expected: All tests PASS (old + new)

**Step 5: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
git add lightning_whisper_mlx/server.py tests/test_server.py
git commit -m "feat: add Speaker CRUD REST API endpoints"
```

---

## Task 3: Dialog TTS Utilities (Tag Stripping + Chunking)

**Files:**
- Create: `lightning_whisper_mlx/dialog.py`
- Test: `tests/test_dialog.py`

**Step 1: Write the failing tests**

```python
# tests/test_dialog.py
"""Tests for dialog TTS utilities."""
import pytest

from lightning_whisper_mlx.dialog import strip_tags, chunk_text


class TestStripTags:
    def test_strips_single_tag(self):
        assert strip_tags("[warmly] Hello there.") == "Hello there."

    def test_strips_multiple_tags(self):
        assert strip_tags("[curious] Mhm, [pause] bin da.") == "Mhm, bin da."

    def test_no_tags_unchanged(self):
        assert strip_tags("Just plain text.") == "Just plain text."

    def test_strips_and_trims(self):
        assert strip_tags("[tag]  Hello  [tag2]") == "Hello"

    def test_pause_tags_stripped(self):
        assert strip_tags("[long pause] Text [short pause] more") == "Text more"


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.")
        assert chunks == ["Hello world."]

    def test_splits_at_punctuation(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_bytes=40)
        assert len(chunks) >= 2
        # All chunks under max_bytes
        for c in chunks:
            assert len(c.encode("utf-8")) <= 40

    def test_respects_max_bytes(self):
        # German text with multi-byte chars
        text = "Geh auf Folie zwölf. Dann schauen wir uns das genauer an. Was siehst du dort?"
        chunks = chunk_text(text, max_bytes=50)
        for c in chunks:
            assert len(c.encode("utf-8")) <= 50

    def test_long_word_not_split(self):
        """A single word shorter than max_bytes stays intact."""
        text = "Donaudampfschifffahrtsgesellschaft."
        chunks = chunk_text(text, max_bytes=135)
        assert len(chunks) == 1

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# lightning_whisper_mlx/dialog.py
"""Dialog TTS utilities: tag stripping, text chunking, silence generation."""
from __future__ import annotations

import re

# Matches any [tag] including [long pause], [warmly], etc.
_TAG_RE = re.compile(r"\[.*?\]")

# Split points: after punctuation followed by whitespace
_SPLIT_RE = re.compile(r"(?<=[;:,.!?])\s+")


def strip_tags(text: str) -> str:
    """Remove all [bracket tags] from text and normalize whitespace."""
    cleaned = _TAG_RE.sub("", text)
    return " ".join(cleaned.split())


def chunk_text(text: str, max_bytes: int = 135) -> list[str]:
    """Split text into chunks of at most max_bytes UTF-8 bytes.

    Splits at punctuation boundaries (;:,.!?) to preserve natural phrasing.
    """
    text = text.strip()
    if not text:
        return []

    # If it fits in one chunk, return as-is
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    # Split at punctuation boundaries
    parts = _SPLIT_RE.split(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)

    return chunks
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_dialog.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
git add lightning_whisper_mlx/dialog.py tests/test_dialog.py
git commit -m "feat: add dialog TTS tag stripping and text chunking"
```

---

## Task 4: Dialog TTS Pipeline (`_run_dialog_tts`)

**Files:**
- Modify: `lightning_whisper_mlx/server.py`
- Test: `tests/test_server.py` (append)

This adds the `POST /api/tts/dialog` endpoint with the full pipeline: tag stripping → chunking → per-speaker generation → silence insertion → WAV concatenation.

**Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
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
        wf.writeframes(b"\x00\x00" * 2400)  # 0.1s silence
    with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["result"] = {"audio_path": audio_path}


@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_creates_job(client):
    """POST /api/tts/dialog with segments returns a job_id."""
    # First create a speaker
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_server.py::test_dialog_tts_creates_job -v`
Expected: FAIL — 404 or AttributeError

**Step 3: Add dialog TTS endpoint and pipeline to server.py**

Add to the imports at top of `server.py`:

```python
from pydantic import BaseModel
```

Add these models and endpoints:

```python
# --- Dialog TTS ---

class DialogSegment(BaseModel):
    speaker_id: str
    text: str

class DialogRequest(BaseModel):
    segments: list[DialogSegment]
    steps: int = 8
    speed: float = 1.0
    pause_between_ms: int = 300
    model: str | None = None


def _run_dialog_tts(job_id: str, output_path: str, segments: list[dict],
                    steps: int, speed: float, pause_between_ms: int,
                    model: str | None) -> None:
    """Run multi-speaker dialog TTS in a background thread."""
    try:
        import numpy as np
        import soundfile as sf

        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.processing

        from .dialog import strip_tags, chunk_text
        from .speakers import get_speaker, get_speaker_audio_path
        from .tts import LightningTTSMLX

        tts = LightningTTSMLX(model=model) if model else LightningTTSMLX()

        all_audio: list[np.ndarray] = []
        total_segments = len(segments)
        prev_speaker_id = None

        for i, seg in enumerate(segments):
            speaker = get_speaker(seg["speaker_id"])
            ref_audio_path = str(get_speaker_audio_path(seg["speaker_id"]))
            ref_text = speaker["ref_text"]

            # Strip tags and chunk
            clean_text = strip_tags(seg["text"])
            chunks = chunk_text(clean_text)

            # Insert silence between speaker changes
            if prev_speaker_id is not None and seg["speaker_id"] != prev_speaker_id:
                silence_samples = int(24000 * pause_between_ms / 1000)
                all_audio.append(np.zeros(silence_samples, dtype=np.float32))

            # Generate each chunk
            for chunk in chunks:
                percent = int((i / total_segments) * 100)
                _update_progress(job_id, "generating",
                                 f"Generating segment {i + 1}/{total_segments}...",
                                 percent)

                chunk_path = str(Path(output_path).parent / f"chunk_{job_id}_{i}_{id(chunk)}.wav")
                tts.generate(
                    text=chunk,
                    output_path=chunk_path,
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    steps=steps,
                    speed=speed,
                )
                # Read generated chunk
                chunk_audio, _ = sf.read(chunk_path, dtype="float32")
                all_audio.append(chunk_audio)
                Path(chunk_path).unlink(missing_ok=True)

            prev_speaker_id = seg["speaker_id"]

        # Concatenate all audio
        if all_audio:
            final_audio = np.concatenate(all_audio)
            sf.write(output_path, final_audio, 24000)

        _update_progress(job_id, "completed", "Done", 100)
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.completed
            _jobs[job_id]["result"] = {"audio_path": output_path}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = JobStatus.failed
            _jobs[job_id]["error"] = str(e)
        Path(output_path).unlink(missing_ok=True)


@app.post("/api/tts/dialog", status_code=202)
async def dialog_tts(req: DialogRequest) -> dict[str, str]:
    """Submit multi-speaker dialog for TTS generation."""
    if not req.segments:
        raise HTTPException(status_code=422, detail="segments must not be empty")

    # Validate all speaker_ids exist
    from .speakers import get_speaker
    for seg in req.segments:
        if get_speaker(seg.speaker_id) is None:
            raise HTTPException(status_code=422,
                                detail=f"Speaker not found: {seg.speaker_id}")

    job_id = str(uuid.uuid4())
    output_path = str(Path(tempfile.gettempdir()) / f"dialog_{job_id}.wav")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": JobStatus.queued,
            "result": None,
            "error": None,
            "progress": None,
        }

    segments_data = [{"speaker_id": s.speaker_id, "text": s.text} for s in req.segments]

    thread = threading.Thread(
        target=_run_dialog_tts,
        args=(job_id, output_path, segments_data, req.steps, req.speed,
              req.pause_between_ms, req.model),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}
```

Add `soundfile` to `pyproject.toml` under `tts` extras:

```toml
tts = ["f5-tts-mlx", "soundfile"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/test_server.py -v`
Expected: All tests PASS (old + new)

**Step 5: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
git add lightning_whisper_mlx/server.py tests/test_server.py pyproject.toml
git commit -m "feat: add POST /api/tts/dialog endpoint with multi-speaker pipeline"
```

---

## Task 5: Frontend — Speaker API Client Functions

**Files:**
- Modify: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/lib/api.ts`

**Step 1: Add Speaker types and API functions**

Append to `lib/api.ts`:

```typescript
// --- Speaker API ---

export interface Speaker {
  id: string
  name: string
  ref_text: string
  created_at: string
}

export async function fetchSpeakers(): Promise<Speaker[]> {
  const res = await fetch(`${_apiBase}/api/speakers`)
  if (!res.ok) throw new Error(`Failed to fetch speakers: ${res.status}`)
  return res.json()
}

export async function createSpeaker(name: string, refAudio: File, refText: string): Promise<Speaker> {
  const form = new FormData()
  form.append("name", name)
  form.append("ref_text", refText)
  form.append("ref_audio", refAudio)

  const res = await fetch(`${_apiBase}/api/speakers`, { method: "POST", body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Failed to create speaker: ${res.status}`)
  }
  return res.json()
}

export async function deleteSpeaker(speakerId: string): Promise<void> {
  const res = await fetch(`${_apiBase}/api/speakers/${speakerId}`, { method: "DELETE" })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Failed to delete speaker: ${res.status}`)
  }
}

// --- Dialog TTS API ---

export interface DialogSegment {
  speaker_id: string
  text: string
}

export interface DialogTtsParams {
  segments: DialogSegment[]
  steps?: number
  speed?: number
  pause_between_ms?: number
  model?: string
}

export async function startDialogTts(params: DialogTtsParams): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${_apiBase}/api/tts/dialog`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Failed to start dialog TTS: ${res.status}`)
  }
  return res.json()
}
```

**Step 2: Verify build**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm build`
Expected: Build succeeds with zero errors

**Step 3: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
git add lib/api.ts
git commit -m "feat: add speaker and dialog TTS API client functions"
```

---

## Task 6: Frontend — Speaker Management in Settings

**Files:**
- Modify: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/app/(app)/settings/page.tsx`

**Step 1: Add Speakers section (section 5) to Settings page**

Add between the TTS Defaults section and the Save button. Add state for speakers, add/delete handlers, and the UI:

```typescript
// New imports at top:
import { fetchSpeakers, createSpeaker, deleteSpeaker } from "@/lib/api"
import type { Speaker } from "@/lib/api"
import { Trash2, Plus, Upload as UploadIcon, User } from "lucide-react"

// New state inside SettingsPage():
const [speakers, setSpeakers] = useState<Speaker[]>([])
const [newSpeakerName, setNewSpeakerName] = useState("")
const [newSpeakerRefText, setNewSpeakerRefText] = useState("")
const [newSpeakerFile, setNewSpeakerFile] = useState<File | null>(null)
const [speakerLoading, setSpeakerLoading] = useState(false)

// New useEffect to load speakers:
useEffect(() => {
  fetchSpeakers().then(setSpeakers).catch(() => {})
}, [])

// Handler functions:
const handleAddSpeaker = async () => {
  if (!newSpeakerName.trim() || !newSpeakerFile || !newSpeakerRefText.trim()) return
  setSpeakerLoading(true)
  try {
    const speaker = await createSpeaker(newSpeakerName.trim(), newSpeakerFile, newSpeakerRefText.trim())
    setSpeakers(prev => [...prev, speaker])
    setNewSpeakerName("")
    setNewSpeakerRefText("")
    setNewSpeakerFile(null)
  } catch (e) {
    // Error handled silently in v1
  } finally {
    setSpeakerLoading(false)
  }
}

const handleDeleteSpeaker = async (id: string) => {
  try {
    await deleteSpeaker(id)
    setSpeakers(prev => prev.filter(s => s.id !== id))
  } catch (e) {
    // Error handled silently in v1
  }
}
```

The JSX for section 5 (place after TTS Defaults SectionWrapper, before Save button):

```tsx
{/* Speakers */}
<SectionWrapper id="section-speakers" number={5} title="Speakers" description="Reference voices for multi-speaker dialog TTS">
  <div className="space-y-4">
    {/* Speaker list */}
    {speakers.length > 0 ? (
      <div className="space-y-2">
        {speakers.map(speaker => (
          <div key={speaker.id} className="flex items-center justify-between p-3 border-2 border-border rounded-sm">
            <div className="flex items-center gap-3 min-w-0">
              <User className="h-4 w-4 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <p className="font-semibold text-sm truncate">{speaker.name}</p>
                <p className="text-xs text-muted-foreground truncate">{speaker.ref_text}</p>
              </div>
            </div>
            <button
              onClick={() => handleDeleteSpeaker(speaker.id)}
              className="p-1.5 text-muted-foreground hover:text-destructive transition-colors shrink-0"
              aria-label={`Delete ${speaker.name}`}
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        ))}
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">No speakers added yet. Add a speaker to use multi-speaker dialog TTS.</p>
    )}

    {/* Add speaker form */}
    <div className="space-y-3 p-4 border-2 border-dashed border-border rounded-sm">
      <Label className="text-xs uppercase tracking-widest text-muted-foreground font-bold block">
        Add Speaker
      </Label>
      <div className="space-y-2">
        <input
          type="text"
          value={newSpeakerName}
          onChange={(e) => setNewSpeakerName(e.target.value)}
          className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
          placeholder="Speaker name (e.g. Alice)"
        />
        <textarea
          value={newSpeakerRefText}
          onChange={(e) => setNewSpeakerRefText(e.target.value)}
          className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50 resize-y"
          placeholder="Reference text (transcript of the uploaded audio)"
          rows={2}
        />
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 px-3 py-2 border-2 border-border rounded-sm cursor-pointer hover:border-foreground transition-colors text-sm font-medium">
            <UploadIcon className="h-4 w-4" />
            {newSpeakerFile ? newSpeakerFile.name : "Choose audio file"}
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => setNewSpeakerFile(e.target.files?.[0] ?? null)}
            />
          </label>
        </div>
        <p className="text-xs text-muted-foreground">Mono WAV, 5–15 seconds, clear without background noise.</p>
      </div>
      <Button
        onClick={handleAddSpeaker}
        disabled={!newSpeakerName.trim() || !newSpeakerFile || !newSpeakerRefText.trim() || speakerLoading}
        className="w-full"
      >
        <Plus className="h-4 w-4" />
        {speakerLoading ? "Adding..." : "Add Speaker"}
      </Button>
    </div>
  </div>
</SectionWrapper>
```

**Step 2: Verify build**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm build`
Expected: Build succeeds with zero errors

**Step 3: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
git add app/\(app\)/settings/page.tsx
git commit -m "feat: add speaker management section to Settings page"
```

---

## Task 7: Frontend — Dialog Page (`/tts/dialog`)

**Files:**
- Create: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/app/(app)/tts/dialog/page.tsx`

**Step 1: Create the dialog page**

```tsx
"use client"

import { useState, useRef, useEffect } from "react"
import { Loader2, Volume2, Download, Plus, Trash2, MessageSquare, FileJson } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { ProgressBar } from "@/components/ui/progress-bar"
import { SectionWrapper } from "@/components/ui/section-wrapper"
import { fetchSpeakers, startDialogTts, pollJobStatus, getTtsAudioUrl } from "@/lib/api"
import type { Speaker, JobProgress } from "@/lib/api"
import { useSettings } from "@/lib/settings"

interface DialogLine {
  id: string
  speakerId: string
  text: string
}

const statusVariant: Record<string, "ghost" | "warning" | "success" | "error"> = {
  idle: "ghost",
  queued: "warning",
  processing: "warning",
  completed: "success",
  failed: "error",
}

export default function DialogPage() {
  const { settings, loaded } = useSettings()
  const [speakers, setSpeakers] = useState<Speaker[]>([])
  const [lines, setLines] = useState<DialogLine[]>([
    { id: crypto.randomUUID(), speakerId: "", text: "" },
  ])
  const [steps, setSteps] = useState(8)
  const [speed, setSpeed] = useState(1.0)
  const [pauseMs, setPauseMs] = useState(300)

  const [status, setStatus] = useState<"idle" | "queued" | "processing" | "completed" | "failed">("idle")
  const [audioJobId, setAudioJobId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<JobProgress | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const [mode, setMode] = useState<"editor" | "json">("editor")
  const [jsonInput, setJsonInput] = useState("")

  useEffect(() => {
    if (!loaded) return
    setSteps(settings.ttsDefaultSteps)
    setSpeed(settings.ttsDefaultSpeed)
  }, [loaded]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchSpeakers().then(setSpeakers).catch(() => {})
  }, [])

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const isRunning = status === "queued" || status === "processing"
  const audioUrl = audioJobId ? getTtsAudioUrl(audioJobId) : null

  const addLine = () => {
    setLines(prev => [...prev, { id: crypto.randomUUID(), speakerId: "", text: "" }])
  }

  const removeLine = (id: string) => {
    setLines(prev => prev.length > 1 ? prev.filter(l => l.id !== id) : prev)
  }

  const updateLine = (id: string, field: "speakerId" | "text", value: string) => {
    setLines(prev => prev.map(l => l.id === id ? { ...l, [field]: value } : l))
  }

  const parseJsonImport = () => {
    try {
      const data = JSON.parse(jsonInput)
      const inputs: { text: string; voice_id: string }[] = data.inputs ?? data.segments ?? []
      if (!inputs.length) return

      const newLines: DialogLine[] = inputs.map(input => ({
        id: crypto.randomUUID(),
        speakerId: speakers.find(s => s.name === input.voice_id || s.id === input.voice_id)?.id ?? "",
        text: input.text,
      }))
      setLines(newLines)
      setMode("editor")
    } catch {
      setError("Invalid JSON format")
    }
  }

  const handleGenerate = async () => {
    const validLines = lines.filter(l => l.speakerId && l.text.trim())
    if (!validLines.length) return

    if (pollRef.current) clearInterval(pollRef.current)

    setStatus("queued")
    setError(null)
    setAudioJobId(null)
    setProgress(null)

    try {
      const { job_id } = await startDialogTts({
        segments: validLines.map(l => ({ speaker_id: l.speakerId, text: l.text })),
        steps,
        speed,
        pause_between_ms: pauseMs,
        model: settings.ttsModel,
      })

      pollRef.current = setInterval(async () => {
        try {
          const job = await pollJobStatus(job_id)
          setStatus(job.status)
          setProgress(job.progress ?? null)

          if (job.status === "completed") {
            setAudioJobId(job_id)
            if (pollRef.current) clearInterval(pollRef.current)
          } else if (job.status === "failed") {
            setError(job.error ?? "Dialog TTS generation failed")
            if (pollRef.current) clearInterval(pollRef.current)
          }
        } catch {
          setError("Lost connection to server")
          if (pollRef.current) clearInterval(pollRef.current)
        }
      }, 1000)
    } catch (e) {
      setStatus("failed")
      setError(e instanceof Error ? e.message : "Failed to start dialog TTS")
    }
  }

  if (!loaded) return null

  return (
    <div className="space-y-12">
      <SectionWrapper id="dialog-input" number={1} title="Dialog" description="Multi-speaker dialogue input" showConnector>
        {/* Mode toggle */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setMode("editor")}
            className={`px-3 py-1.5 text-sm font-semibold border-2 rounded-sm transition-colors ${
              mode === "editor"
                ? "border-foreground bg-foreground text-background"
                : "border-border hover:border-foreground"
            }`}
          >
            <MessageSquare className="h-3 w-3 inline mr-1.5 -mt-0.5" />
            Visual Editor
          </button>
          <button
            onClick={() => setMode("json")}
            className={`px-3 py-1.5 text-sm font-semibold border-2 rounded-sm transition-colors ${
              mode === "json"
                ? "border-foreground bg-foreground text-background"
                : "border-border hover:border-foreground"
            }`}
          >
            <FileJson className="h-3 w-3 inline mr-1.5 -mt-0.5" />
            JSON Import
          </button>
        </div>

        {mode === "editor" ? (
          <div className="space-y-3">
            {speakers.length === 0 && (
              <p className="text-sm text-muted-foreground p-3 border-2 border-dashed border-border rounded-sm">
                No speakers found. Add speakers in Settings first.
              </p>
            )}

            {lines.map((line, idx) => (
              <div key={line.id} className="flex gap-2 items-start">
                <select
                  value={line.speakerId}
                  onChange={(e) => updateLine(line.id, "speakerId", e.target.value)}
                  className="w-40 shrink-0 rounded-sm border-2 border-border bg-background px-2 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
                >
                  <option value="">Speaker...</option>
                  {speakers.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
                <div className="flex-1 space-y-1">
                  <input
                    type="text"
                    value={line.text}
                    onChange={(e) => updateLine(line.id, "text", e.target.value)}
                    placeholder="Dialog text..."
                    className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
                  />
                  <p className={`text-xs ${
                    line.text.length > 135 ? "text-destructive font-semibold" : "text-muted-foreground"
                  }`}>
                    {line.text.length} / 135
                  </p>
                </div>
                <button
                  onClick={() => removeLine(line.id)}
                  className="p-2 text-muted-foreground hover:text-destructive transition-colors shrink-0"
                  disabled={lines.length <= 1}
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}

            <Button variant="outline" onClick={addLine} className="w-full">
              <Plus className="h-4 w-4" />
              Add Line
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <textarea
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              placeholder={`Paste ElevenLabs-compatible JSON:\n{\n  "inputs": [\n    {"text": "[warmly] Hello.", "voice_id": "Alice"},\n    {"text": "[curious] Hi!", "voice_id": "Bob"}\n  ]\n}`}
              rows={8}
              className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-mono placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-foreground/50 resize-y"
            />
            <Button onClick={parseJsonImport} disabled={!jsonInput.trim()} className="w-full">
              Import & Switch to Editor
            </Button>
          </div>
        )}
      </SectionWrapper>

      <SectionWrapper id="dialog-params" number={2} title="Parameters" description="Generation settings">
        <div className="grid grid-cols-3 gap-4">
          <div className="space-y-2">
            <Label htmlFor="dialog-steps">Diffusion Steps</Label>
            <input
              id="dialog-steps"
              type="number"
              min={1} max={64}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="dialog-speed">Speed</Label>
            <input
              id="dialog-speed"
              type="number"
              min={0.5} max={2.0} step={0.1}
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="dialog-pause">Pause (ms)</Label>
            <input
              id="dialog-pause"
              type="number"
              min={0} max={5000} step={100}
              value={pauseMs}
              onChange={(e) => setPauseMs(Number(e.target.value))}
              className="w-full rounded-sm border-2 border-border bg-background px-3 py-2 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-foreground/50"
            />
            <p className="text-xs text-muted-foreground">Between speaker changes</p>
          </div>
        </div>
      </SectionWrapper>

      <SectionWrapper id="dialog-generate" number={3} title="Generate" description="Run dialog TTS and download audio">
        <div className="space-y-4">
          <Button
            onClick={handleGenerate}
            disabled={!lines.some(l => l.speakerId && l.text.trim()) || isRunning}
            className="w-full h-12 text-base"
          >
            {isRunning ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                {status === "queued" ? "Queued..." : "Generating Dialog..."}
              </>
            ) : (
              <>
                <Volume2 className="h-5 w-5" />
                Generate Dialog
              </>
            )}
          </Button>

          {progress && isRunning && (
            <ProgressBar
              phase={progress.phase}
              detail={progress.detail}
              percent={progress.percent}
            />
          )}

          {status !== "idle" && (
            <div className="flex items-center gap-2">
              <Badge variant={statusVariant[status]}>
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </Badge>
            </div>
          )}

          {error && (
            <div className="p-3 border-2 border-destructive rounded-sm bg-destructive/10 text-sm text-destructive">
              {error}
            </div>
          )}

          {audioUrl && status === "completed" && (
            <div className="space-y-3 p-4 border-2 border-border rounded-sm bg-card">
              <audio controls className="w-full" src={audioUrl} />
              <div className="flex gap-2">
                <Button variant="outline" size="sm" asChild>
                  <a href={audioUrl} download={`dialog_${audioJobId}.wav`}>
                    <Download className="h-4 w-4" />
                    Download WAV
                  </a>
                </Button>
              </div>
            </div>
          )}
        </div>
      </SectionWrapper>
    </div>
  )
}
```

**Step 2: Verify build**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm build`
Expected: Build succeeds with zero errors

**Step 3: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
git add app/\(app\)/tts/dialog/page.tsx
git commit -m "feat: add /tts/dialog page with visual editor and JSON import"
```

---

## Task 8: Frontend — Sidebar Navigation Update

**Files:**
- Modify: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/components/layout/sidebar.tsx`

**Step 1: Update sidebar to add Dialog sub-item and routing support**

Update the `AppPage` type and `ttsItems` array:

```typescript
// Change AppPage type to include dialog:
export type AppPage = "stt" | "tts" | "tts-dialog" | "settings"

// Update ttsItems:
const ttsItems: { id: string; label: string; icon: React.ReactNode; page?: AppPage }[] = [
  { id: "tts-input", label: "Generate", icon: <Volume2 className="h-4 w-4" /> },
  { id: "tts-dialog", label: "Dialog", icon: <MessageSquare className="h-4 w-4" />, page: "tts-dialog" },
]
```

Add `MessageSquare` to the lucide-react import.

Update the TTS items rendering in `NavContent` to use `page` for routing — when a TTS item has a `page` property, clicking it navigates to that page instead of scrolling:

```typescript
{ttsItems.map(item => (
  <button
    key={item.id}
    onClick={() => item.page ? handleClick(item.page) : handleClick("tts", item.id)}
    className={cn(
      "flex items-center gap-2 w-full text-left px-3 rounded-sm transition-colors text-sm font-semibold border-2 border-transparent",
      linkClass,
      activePage !== "tts" && activePage !== "tts-dialog" && "opacity-50",
      (item.page ? activePage === item.page : activePage === "tts" && activeSection === item.id)
        ? "bg-sidebar-active text-sidebar-active-foreground border-sidebar-active"
        : "hover:border-border"
    )}
  >
    {item.icon}
    {item.label}
  </button>
))}
```

Update the TTS group header highlight to also match `tts-dialog`:

```typescript
activePage === "tts" || activePage === "tts-dialog" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
```

And its `aria-current`:

```typescript
aria-current={activePage === "tts" || activePage === "tts-dialog" ? "page" : undefined}
```

**Step 2: Update layout.tsx to handle the new route**

Modify `/Users/marcusifland/prj/lightning-whisper-mlx-ui/app/(app)/layout.tsx` to map the `/tts/dialog` pathname to the `tts-dialog` AppPage value. The `pathname` → `AppPage` mapping needs to include:

```typescript
// In the pathname-to-page mapping:
"/tts/dialog" → "tts-dialog"
```

And the `onNavigate` handler needs to route `tts-dialog` to `/tts/dialog`.

**Step 3: Verify build**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm build`
Expected: Build succeeds with zero errors

**Step 4: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
git add components/layout/sidebar.tsx app/\(app\)/layout.tsx
git commit -m "feat: add Dialog sub-item to sidebar navigation"
```

---

## Task 9: Integration Test — Full Round-Trip

**Files:**
- Modify: `tests/test_server.py` (append)

**Step 1: Write a test that exercises the full flow via API**

```python
@patch("lightning_whisper_mlx.server._run_dialog_tts", _noop_dialog_tts)
def test_dialog_tts_full_flow(client):
    """Full flow: create speakers → submit dialog → poll → download audio."""
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
```

**Step 2: Run all tests**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
git add tests/test_server.py
git commit -m "test: add dialog TTS integration test for full API flow"
```

---

## Task 10: Final Verification & Build Check

**Step 1: Run all backend tests**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Build frontend**

Run: `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm build`
Expected: Build succeeds with zero errors

**Step 3: Manual smoke test (optional)**

Start both servers and verify:
1. `cd /Users/marcusifland/prj/lightning-whisper-mlx && uv run uvicorn lightning_whisper_mlx.server:app --reload`
2. `cd /Users/marcusifland/prj/lightning-whisper-mlx-ui && pnpm dev`
3. Go to Settings → Speakers section → Add a speaker
4. Go to TTS → Dialog → Add lines, assign speakers, generate
5. Verify audio plays back
