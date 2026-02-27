# Speaker Diarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-speaker detection via pyannote-audio with standalone functions and integrated `transcribe(diarize=True)` convenience parameter.

**Architecture:** Two pure functions in `diarize.py` — `diarize_audio()` wraps pyannote Pipeline, `assign_speakers()` does temporal overlap assignment. Integrated into `transcribe()` via optional `diarize` parameter. Import guard pattern for optional `pyannote-audio` dependency.

**Tech Stack:** pyannote-audio (diarization), PyTorch (pyannote dependency), pytest

---

## Context

**Segment format:** `transcribe_audio()` returns `{"text": str, "segments": list, "language": str}` where each segment is `[start_seek, end_seek, text]`. Seek values are mel frame positions. Convert to seconds: `seconds = seek * HOP_LENGTH / SAMPLE_RATE` = `seek * 160 / 16000` = `seek * 0.01`.

**Constants (from `audio.py`):** `HOP_LENGTH = 160`, `SAMPLE_RATE = 16000`.

**pyannote API:**
```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token="HF_TOKEN")
diarization = pipeline("audio.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # turn.start, turn.end (seconds), speaker (str like "SPEAKER_00")
```

---

## Task 1: Add `diarize` optional dependency to pyproject.toml

**Files:**
- Modify: `pyproject.toml:16-18`

**Step 1: Add diarize extra**

Edit `pyproject.toml` optional-dependencies to:

```toml
[project.optional-dependencies]
tts = ["f5-tts-mlx"]
diarize = ["pyannote-audio"]
dev = ["pytest"]
```

**Step 2: Verify uv resolves it**

Run: `uv lock --check`
Expected: Lock file up to date (or updates cleanly)

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyannote-audio as optional diarize dependency"
```

---

## Task 2: Write `assign_speakers` tests (pure logic, no pyannote)

**Files:**
- Create: `tests/test_diarize.py`

**Step 1: Write the test file**

```python
import pytest

from lightning_whisper_mlx.audio import HOP_LENGTH, SAMPLE_RATE


def _seek(seconds: float) -> int:
    """Convert seconds to seek position (mel frames)."""
    return int(seconds * SAMPLE_RATE / HOP_LENGTH)


class TestAssignSpeakers:
    """assign_speakers must match segments to speaker turns by temporal overlap."""

    def test_two_speakers_clean_turns(self):
        """Each segment falls entirely within one speaker turn."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(0.0), _seek(3.0), "Hello from speaker A."],
            [_seek(3.0), _seek(6.0), "Hello from speaker B."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 6.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello from speaker A."
        assert result[0]["start"] == pytest.approx(0.0, abs=0.02)
        assert result[0]["end"] == pytest.approx(3.0, abs=0.02)
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_single_speaker(self):
        """All segments assigned to the sole speaker."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(0.0), _seek(2.0), "First sentence."],
            [_seek(2.0), _seek(4.0), "Second sentence."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert all(s["speaker"] == "SPEAKER_00" for s in result)

    def test_overlap_assigns_dominant_speaker(self):
        """Segment spanning two turns gets the speaker with more overlap."""
        from lightning_whisper_mlx.diarize import assign_speakers

        # Segment 1.0-4.0s: speaker A covers 1.0-2.0 (1s), speaker B covers 2.0-4.0 (2s)
        segments = [
            [_seek(1.0), _seek(4.0), "Overlapping segment."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 5.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["speaker"] == "SPEAKER_01"  # 2s > 1s overlap

    def test_segment_in_silence_gap(self):
        """Segment falling in a gap between speaker turns gets speaker None."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(5.0), _seek(7.0), "Nobody is speaking here."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 10.0, "end": 13.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["speaker"] is None

    def test_empty_segments(self):
        """Empty segments list returns empty list."""
        from lightning_whisper_mlx.diarize import assign_speakers

        result = assign_speakers([], [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}])
        assert result == []

    def test_empty_speaker_turns(self):
        """No speaker turns means all segments get speaker None."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [[_seek(0.0), _seek(2.0), "Some text."]]
        result = assign_speakers(segments, [])

        assert result[0]["speaker"] is None

    def test_output_has_seconds_not_seeks(self):
        """Output start/end must be in seconds, not seek positions."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [[_seek(1.5), _seek(3.5), "Test."]]
        speaker_turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["start"] == pytest.approx(1.5, abs=0.02)
        assert result[0]["end"] == pytest.approx(3.5, abs=0.02)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (diarize module doesn't exist yet)

**Step 3: Commit**

```bash
git add tests/test_diarize.py
git commit -m "test: add assign_speakers unit tests with mock data"
```

---

## Task 3: Implement `assign_speakers` function

**Files:**
- Create: `lightning_whisper_mlx/diarize.py`

**Step 1: Create diarize.py with assign_speakers**

```python
from typing import Optional

from .audio import HOP_LENGTH, SAMPLE_RATE


def _seek_to_seconds(seek: int) -> float:
    """Convert a mel-frame seek position to seconds."""
    return seek * HOP_LENGTH / SAMPLE_RATE


def assign_speakers(
    segments: list,
    speaker_turns: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcription segments by temporal overlap.

    Each input segment is [start_seek, end_seek, text] (seek in mel frames).
    Each speaker_turn is {"speaker": str, "start": float, "end": float} (seconds).

    Returns list of dicts with keys: start, end, text, speaker (seconds).
    Speaker is None if no speaker turn overlaps the segment.
    """
    result = []
    for seg in segments:
        start_seek, end_seek, text = seg
        seg_start = _seek_to_seconds(start_seek)
        seg_end = _seek_to_seconds(end_seek)

        best_speaker = None
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        result.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "speaker": best_speaker,
        })

    return result
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: All 7 tests PASS

**Step 3: Commit**

```bash
git add lightning_whisper_mlx/diarize.py
git commit -m "feat: implement assign_speakers with temporal overlap logic"
```

---

## Task 4: Implement `diarize_audio` function

**Files:**
- Modify: `lightning_whisper_mlx/diarize.py`

**Step 1: Add diarize_audio to diarize.py**

Add before `assign_speakers`:

```python
import os


def diarize_audio(
    audio_path: str,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """Run speaker diarization on an audio file using pyannote-audio.

    Requires:
    - pyannote-audio installed: uv sync --extra diarize
    - HF_TOKEN environment variable set (huggingface.co/settings/tokens)

    Returns list of dicts: [{"speaker": "SPEAKER_00", "start": 0.2, "end": 1.5}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote-audio is required for diarization. Install it with:\n"
            "  pip install lightning-whisper-mlx[diarize]\n"
            "  # or: uv sync --extra diarize"
        ) from e

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable required for pyannote diarization models.\n"
            "Get a token at https://huggingface.co/settings/tokens"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarization = pipeline(audio_path, num_speakers=num_speakers,
                           min_speakers=min_speakers, max_speakers=max_speakers)

    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })

    return speaker_turns
```

**Step 2: Add import guard test to test_diarize.py**

Append to `tests/test_diarize.py`:

```python
class TestDiarizeAudioImportGuard:
    """diarize_audio must raise ImportError with install instructions when pyannote missing."""

    def test_missing_hf_token_raises(self, monkeypatch):
        """Missing HF_TOKEN raises EnvironmentError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Only test if pyannote is actually installed
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            pytest.skip("pyannote-audio not installed")

        from lightning_whisper_mlx.diarize import diarize_audio

        with pytest.raises(EnvironmentError, match="HF_TOKEN"):
            diarize_audio("dummy.wav")
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_diarize.py -v`
Expected: All tests PASS (import guard test skips if pyannote not installed)

**Step 4: Commit**

```bash
git add lightning_whisper_mlx/diarize.py tests/test_diarize.py
git commit -m "feat: implement diarize_audio with pyannote-audio pipeline"
```

---

## Task 5: Integrate into transcribe() and __init__.py

**Files:**
- Modify: `lightning_whisper_mlx/lightning.py:88-90`
- Modify: `lightning_whisper_mlx/__init__.py`

**Step 1: Add diarize parameter to transcribe()**

Replace the `transcribe` method in `lightning.py`:

```python
    def transcribe(self, audio_path, language=None, diarize=False):
        result = transcribe_audio(audio_path, path_or_hf_repo=f'./mlx_models/{self.name}', language=language, batch_size=self.batch_size)
        if diarize:
            from .diarize import diarize_audio, assign_speakers
            speaker_turns = diarize_audio(audio_path)
            result["segments"] = assign_speakers(result["segments"], speaker_turns)
        return result
```

**Step 2: Add diarization exports to __init__.py**

Replace `__init__.py`:

```python
from .lightning import LightningWhisperMLX


def __getattr__(name):
    if name == "LightningTTSMLX":
        from .tts import LightningTTSMLX
        return LightningTTSMLX
    if name in ("diarize_audio", "assign_speakers"):
        from . import diarize
        return getattr(diarize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All existing tests PASS + diarize tests PASS

**Step 4: Commit**

```bash
git add lightning_whisper_mlx/lightning.py lightning_whisper_mlx/__init__.py
git commit -m "feat: integrate diarization into transcribe() and public API"
```

---

## Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update module dependency graph**

Change the graph in CLAUDE.md to:

```
__init__.py → lightning.py → transcribe.py → audio.py
                                            → load_models.py → whisper.py
                                            → decoding.py → tokenizer.py
                                            → timing.py
                                            → tokenizer.py
                            → diarize.py → pyannote.audio (external, optional)
           → tts.py → f5_tts_mlx (external, optional)
```

**Step 2: Update Public API section**

Add bullet point:

```
- **`diarize_audio()`** / **`assign_speakers()`** (`diarize.py`) — Speaker diarization. Lazy-imported via `__getattr__`. Requires optional `pyannote-audio` dependency. Install with `uv sync --extra diarize`. Requires `HF_TOKEN` env var.
```

**Step 3: Update Setup & Development section**

Add to the uv sync block:

```bash
uv sync --extra diarize   # + pyannote-audio (speaker diarization)
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with diarization module"
```

---

## Execution Sequence

```
Task 1: Add diarize dependency to pyproject.toml
Task 2: Write assign_speakers tests (TDD: tests first)
Task 3: Implement assign_speakers (make tests pass)
Task 4: Implement diarize_audio + import guard test
Task 5: Integrate into transcribe() and __init__.py
Task 6: Update CLAUDE.md
```

Tasks are strictly sequential. Each depends on the previous.
