# TTS Integration & Test Suite Redesign

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add text-to-speech via F5-TTS-MLX wrapper, migrate to uv + pyproject.toml, and replace tautological tests with real behavior tests including a TTS→STT roundtrip.

**Architecture:** 3 workstreams: (1) migrate build system to uv + pyproject.toml, (2) add TTS wrapper module, (3) rewrite test suite with reference values and E2E roundtrip.

**Tech Stack:** uv (package manager), pyproject.toml (build config), f5-tts-mlx (TTS backend), pytest (testing)

---

## Problem: Current Tests Are Tautological

The existing 52 tests validate type, shape, and existence — not correctness. Examples:

| Test | Problem |
|------|---------|
| `assert SAMPLE_RATE == 16000` | Tests a constant against itself |
| `assert isinstance(result, mx.array)` | Would pass even if output is garbage |
| `assert result.shape[-1] == 80` | Shape check, content could be random |
| `assert isinstance(tokenizer.eot, int)` | Existence check, not value correctness |
| `assert result.ndim == 2` | Structural check, not behavioral |

**Core issue:** Almost every test would still pass if the functions returned wrong but correctly-shaped data.

---

## Design

### 1. Build System: setup.py → pyproject.toml + uv

**Delete:** `setup.py`

**Create:** `pyproject.toml`

```toml
[project]
name = "lightning-whisper-mlx"
version = "0.0.10"
description = "High-performance Whisper speech-to-text and F5 text-to-speech for Apple Silicon"
requires-python = ">=3.10"
dependencies = [
    "huggingface_hub",
    "mlx",
    "numba",
    "numpy",
    "tqdm",
    "tiktoken==0.3.3",
    "scipy",
]

[project.optional-dependencies]
tts = ["f5-tts-mlx"]
dev = ["pytest"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["lightning_whisper_mlx"]
```

**Workflow:**
```bash
uv sync                  # base dependencies
uv sync --extra dev      # + pytest
uv sync --extra tts      # + f5-tts-mlx
uv sync --all-extras     # everything
uv run pytest            # run tests
uv run pytest -m slow    # include E2E tests
```

### 2. TTS Module: `lightning_whisper_mlx/tts.py`

Eigenständige Klasse `LightningTTSMLX` als Wrapper um `f5-tts-mlx`.

**Public API:**

```python
from lightning_whisper_mlx import LightningTTSMLX

tts = LightningTTSMLX(model="lucasnewman/f5-tts-mlx")

# Basic generation
tts.generate(
    text="Hallo, das ist ein Test.",
    output_path="output.wav",
)

# With voice cloning
tts.generate(
    text="Guten Morgen allerseits.",
    output_path="output.wav",
    ref_audio="reference_voice.wav",
    ref_text="Das ist die Referenzstimme.",
)
```

**Implementation details:**

- `__init__` stores model name/config, lazy-loads on first `generate()` call
- `generate()` delegates to `f5_tts_mlx.generate.generate()`
- Handles audio format requirements (F5 expects 24kHz mono WAV)
- Returns path to generated audio file
- Import guard: raises `ImportError` with install instructions if `f5-tts-mlx` not installed

**Parameters for `generate()`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Text to synthesize |
| `output_path` | `str` | required | Path for output WAV file |
| `ref_audio` | `Optional[str]` | `None` | Reference audio for voice cloning |
| `ref_text` | `Optional[str]` | `None` | Transcript of reference audio |
| `steps` | `int` | `8` | Diffusion steps (more = better quality, slower) |
| `speed` | `float` | `1.0` | Speech speed multiplier |
| `seed` | `Optional[int]` | `None` | Reproducibility seed |

**Module dependency graph update:**

```
__init__.py → lightning.py → transcribe.py → ...  (STT path, unchanged)
           → tts.py → f5_tts_mlx (external)       (TTS path, new)
```

### 3. Test Suite Redesign

**Delete all existing tests.** Replace with:

#### 3a. Unit Tests with Reference Values (`tests/test_audio.py`)

Pre-computed reference values (verified against codebase):

| Input | Measurement | Reference Value |
|-------|------------|-----------------|
| 440Hz sine, 1s, 16kHz | Peak mel bin (80 mels) | **11** |
| Silence (zeros) | All mel values | **-1.5** (uniform) |
| 1000Hz sine, 1s, 16kHz | Peak STFT bin | **25** |

```python
class TestMelSpectrogram:
    def test_440hz_energy_concentration(self):
        """A 440Hz tone must produce peak energy at mel bin 11."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000, dtype=np.float32))
        mel = log_mel_spectrogram(audio, n_mels=80)
        avg_energy = np.array(mel).mean(axis=0)
        assert np.argmax(avg_energy) == 11

    def test_silence_produces_uniform_floor(self):
        """All-zeros input must produce uniform -1.5 across all mel bins."""
        mel = log_mel_spectrogram(np.zeros(16000, dtype=np.float32), n_mels=80)
        mel_np = np.array(mel)
        assert np.allclose(mel_np, -1.5, atol=0.01)

    def test_1000hz_stft_peak(self):
        """1000Hz tone STFT must peak at frequency bin 25."""
        audio = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 16000, dtype=np.float32))
        freqs = stft(mx.array(audio), hanning(N_FFT), nperseg=N_FFT, noverlap=HOP_LENGTH)
        avg_mag = np.abs(np.array(freqs)).mean(axis=0)
        assert np.argmax(avg_mag) == 25
```

#### 3b. Unit Tests with Known Token IDs (`tests/test_tokenizer.py`)

```python
class TestTokenizerKnownValues:
    def test_known_special_token_ids(self):
        """Whisper special tokens must have exact known IDs."""
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.eot == 50257
        assert tok.sot == 50258
        assert tok.timestamp_begin == 50364
        assert tok.no_timestamps == 50363

    def test_english_language_token(self):
        """English language token must be sot + 1 + index('en')."""
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.language_token == 50259  # <|en|>

    def test_german_language_token(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.language_token == 50261  # <|de|>

    def test_encode_known_text(self):
        """Known text must produce known token sequence."""
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.encode(" hello") == [7751]
```

#### 3c. Decoding Tests with Behavioral Assertions (`tests/test_decoding.py`)

```python
class TestCompressionRatio:
    def test_known_value(self):
        """compression_ratio for a known string must match pre-computed value."""
        # Pre-compute: compression_ratio("hello world") == X.XXX
        # Assert exact match (deterministic function)

class TestGreedyDecoder:
    def test_sum_logprobs_accumulates_correctly(self):
        """After decoding, sum_logprobs must equal sum of per-token log probs."""
        # Feed known logits → verify sum_logprobs matches manual calculation

class TestSuppressBlank:
    def test_prevents_empty_transcription(self):
        """At sample_begin, blank/EOT tokens must have -inf logit."""
        # This existing test is actually good — keep it
```

#### 3d. E2E Roundtrip: TTS → STT (`tests/test_roundtrip.py`)

```python
@pytest.mark.slow
class TestTTSToSTTRoundtrip:
    def test_german_roundtrip(self):
        """Generate German speech with F5-TTS, transcribe with Whisper, verify content."""
        tts = LightningTTSMLX()
        tts.generate(
            text="Die Sonne scheint heute besonders hell.",
            output_path=str(tmp_path / "test.wav"),
            seed=42,
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe(
            str(tmp_path / "test.wav"),
            language="de",
        )

        text = result["text"].lower()
        # Fuzzy match: core words must appear
        assert "sonne" in text
        assert "scheint" in text
        assert "hell" in text

    def test_english_roundtrip(self):
        """Same roundtrip in English."""
        # "The quick brown fox" → TTS → STT → verify keywords
```

**Test markers:**
- Default `pytest` → runs unit tests only (~fast, no model download)
- `pytest -m slow` → includes E2E roundtrip (downloads tiny model + F5 model)

---

## Files Changed

| Action | File | Description |
|--------|------|-------------|
| Delete | `setup.py` | Replaced by pyproject.toml |
| Create | `pyproject.toml` | Build config with uv support |
| Create | `lightning_whisper_mlx/tts.py` | LightningTTSMLX wrapper class |
| Modify | `lightning_whisper_mlx/__init__.py` | Re-export LightningTTSMLX (lazy) |
| Rewrite | `tests/test_audio.py` | Real reference-value tests |
| Rewrite | `tests/test_tokenizer.py` | Known token ID tests |
| Rewrite | `tests/test_decoding.py` | Behavioral tests |
| Create | `tests/test_roundtrip.py` | E2E TTS→STT roundtrip |
| Modify | `tests/conftest.py` | Update fixtures, add slow marker |
| Delete | `.venv/` | Will be recreated by uv sync |

---

## Execution Sequence

```
Task 1: Migrate setup.py → pyproject.toml
Task 2: Verify uv sync + uv run pytest works with existing tests
Task 3: Create lightning_whisper_mlx/tts.py (LightningTTSMLX)
Task 4: Update __init__.py to re-export LightningTTSMLX
Task 5: Rewrite tests/test_audio.py with reference values
Task 6: Rewrite tests/test_tokenizer.py with known token IDs
Task 7: Rewrite tests/test_decoding.py with behavioral assertions
Task 8: Create tests/test_roundtrip.py (E2E TTS→STT)
Task 9: Run full test suite, fix issues, commit
```

Tasks 5-7 can run in parallel. Task 8 depends on Task 3 (TTS module).
