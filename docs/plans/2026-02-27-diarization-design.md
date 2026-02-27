# Speaker Diarization Design

**Goal:** Add multi-speaker detection to lightning-whisper-mlx via pyannote-audio, with both standalone functions and integrated `transcribe()` convenience parameter.

**Approach:** Thin wrapper — two pure functions in `diarize.py` + convenience param on `transcribe()`.

---

## Data Flow

```
Audio file
    │
    ├──► transcribe_audio() → {"text", "segments": [[start_seek, end_seek, text], ...], "language"}
    │
    ├──► diarize_audio(path) → [{"speaker": "SPEAKER_00", "start": 0.2, "end": 1.5}, ...]
    │
    └──► assign_speakers(segments, speaker_turns) → [{"start": 0.0, "end": 2.5, "text": "...", "speaker": "SPEAKER_00"}, ...]
```

Segments use seek positions (mel frames) internally. `assign_speakers()` converts to seconds via `seek * HOP_LENGTH / SAMPLE_RATE` before computing overlap.

**Output format:** When `diarize=True`, segments become dicts with `start`, `end`, `text`, `speaker` keys. When `diarize=False`, output stays unchanged (backward compatible).

---

## Module & API

### New: `lightning_whisper_mlx/diarize.py`

```python
def diarize_audio(
    audio_path: str,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """Run pyannote speaker diarization.
    Returns [{"speaker": "SPEAKER_00", "start": 0.2, "end": 1.5}, ...]
    """

def assign_speakers(
    segments: list,
    speaker_turns: list[dict],
) -> list[dict]:
    """Pure function. For each segment, find speaker turn with maximum temporal overlap.
    Converts seek-based segments to second-based dicts with speaker labels.
    No pyannote dependency — just math on timestamps.
    """
```

- `diarize_audio`: reads `HF_TOKEN` from env, loads `pyannote/speaker-diarization-3.1`, runs pipeline, converts `Annotation` to plain dicts
- `assign_speakers`: pure function, no external dependencies

### Modified: `lightning_whisper_mlx/lightning.py`

```python
def transcribe(self, audio_path, language=None, diarize=False):
    result = transcribe_audio(...)
    if diarize:
        from .diarize import diarize_audio, assign_speakers
        speaker_turns = diarize_audio(audio_path)
        result["segments"] = assign_speakers(result["segments"], speaker_turns)
    return result
```

### Modified: `lightning_whisper_mlx/__init__.py`

Re-export `diarize_audio` and `assign_speakers` via `__getattr__` for standalone use.

### Modified: `pyproject.toml`

```toml
[project.optional-dependencies]
tts = ["f5-tts-mlx"]
diarize = ["pyannote-audio"]
dev = ["pytest"]
```

---

## Error Handling

1. **`pyannote-audio` not installed:** `ImportError` with `"pyannote-audio is required for diarization. Install with: uv sync --extra diarize"`
2. **`HF_TOKEN` not set:** `EnvironmentError` with `"HF_TOKEN environment variable required for pyannote diarization models. Get a token at https://huggingface.co/settings/tokens"`
3. **No speakers detected:** `assign_speakers` returns segments with `speaker: None`. No crash.

---

## Testing Strategy

### Unit tests (`tests/test_diarize.py`) — no pyannote needed

Test `assign_speakers` with mock data:
- Two speakers, clean turns → correct assignment
- Single speaker → all segments same speaker
- Overlap → assigned to speaker with greater overlap duration
- Silence gap → segment gets `speaker: None`
- Empty inputs → no crash
- Seek-to-seconds conversion correctness

### Import guard test
Verify `ImportError` with install instructions when `pyannote-audio` not installed.

### E2E test (`@pytest.mark.slow`) — requires pyannote + HF_TOKEN
Generate multi-speaker audio with TTS (two calls, concatenate), transcribe with `diarize=True`, verify at least 2 distinct speakers in output.

---

## Files Changed

| Action | File | Description |
|--------|------|-------------|
| Create | `lightning_whisper_mlx/diarize.py` | diarize_audio() + assign_speakers() |
| Modify | `lightning_whisper_mlx/lightning.py` | Add diarize param to transcribe() |
| Modify | `lightning_whisper_mlx/__init__.py` | Re-export diarization functions |
| Modify | `pyproject.toml` | Add diarize optional dependency |
| Create | `tests/test_diarize.py` | Unit tests for assign_speakers |
| Modify | `CLAUDE.md` | Update architecture docs |
