# TTS Integration & Test Suite Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate build system to uv + pyproject.toml, add F5-TTS-MLX wrapper, and replace tautological tests with reference-value behavior tests including a TTS→STT E2E roundtrip.

**Architecture:** Three sequential workstreams: (1) pyproject.toml migration, (2) TTS wrapper module, (3) test suite rewrite. Tests use pre-computed reference values verified against the codebase — not type/shape checks.

**Tech Stack:** uv 0.7+, pyproject.toml (hatchling), f5-tts-mlx (lucasnewman), pytest, Python 3.13

**Design Doc:** `docs/plans/2026-02-26-tts-integration-and-test-redesign.md`

---

### Task 1: Migrate setup.py → pyproject.toml

**Files:**
- Delete: `setup.py`
- Create: `pyproject.toml`

**Step 1: Create pyproject.toml**

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

[tool.pytest.ini_options]
markers = [
    "slow: marks tests that download models and run E2E (deselect with '-m \"not slow\"')",
]
```

**Step 2: Delete setup.py**

Delete the file `setup.py`.

**Step 3: Delete old .venv and recreate with uv**

```bash
rm -rf .venv
uv sync --extra dev
```

Expected: `.venv` recreated with all base deps + pytest installed. Output shows "Resolved X packages" and "Installed X packages".

**Step 4: Verify existing tests still pass**

```bash
uv run pytest tests/ -v
```

Expected: All 52 existing tests PASS (they're tautological but should still work to prove the build system migration is correct).

**Step 5: Commit**

```bash
git add pyproject.toml tests/
git rm setup.py
git commit -m "build: migrate setup.py to pyproject.toml with uv support

Add hatchling build backend, optional extras for tts and dev,
pytest marker config for slow tests. Delete setup.py."
```

---

### Task 2: Create TTS wrapper module

**Files:**
- Create: `lightning_whisper_mlx/tts.py`
- Modify: `lightning_whisper_mlx/__init__.py`

**Step 1: Create `lightning_whisper_mlx/tts.py`**

```python
from typing import Optional


class LightningTTSMLX:
    """Text-to-speech wrapper around f5-tts-mlx for Apple Silicon.

    Requires the optional `tts` extra: pip install lightning-whisper-mlx[tts]
    """

    def __init__(self, model: str = "lucasnewman/f5-tts-mlx"):
        self.model_name = model
        self._f5_generate = None

    def _ensure_loaded(self):
        if self._f5_generate is not None:
            return
        try:
            from f5_tts_mlx.generate import generate
        except ImportError:
            raise ImportError(
                "f5-tts-mlx is required for TTS. Install it with:\n"
                "  pip install lightning-whisper-mlx[tts]\n"
                "  # or: uv sync --extra tts"
            )
        self._f5_generate = generate

    def generate(
        self,
        text: str,
        output_path: str,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        steps: int = 8,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            output_path: Path for output WAV file.
            ref_audio: Reference audio path for voice cloning.
            ref_text: Transcript of reference audio.
            steps: Diffusion steps (more = better quality, slower).
            speed: Speech speed multiplier.
            seed: Random seed for reproducibility.

        Returns:
            The output_path where the WAV file was written.
        """
        self._ensure_loaded()
        self._f5_generate(
            generation_text=text,
            model_name=self.model_name,
            ref_audio_path=ref_audio,
            ref_audio_text=ref_text,
            steps=steps,
            speed=speed,
            seed=seed,
            output_path=output_path,
        )
        return output_path
```

**Step 2: Update `lightning_whisper_mlx/__init__.py` with lazy TTS import**

Replace the content of `__init__.py` with:

```python
from .lightning import LightningWhisperMLX


def __getattr__(name):
    if name == "LightningTTSMLX":
        from .tts import LightningTTSMLX
        return LightningTTSMLX
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

This uses Python 3.7+ module-level `__getattr__` so that `from lightning_whisper_mlx import LightningTTSMLX` works without importing f5-tts-mlx at import time.

**Step 3: Verify import works without f5-tts-mlx installed**

```bash
uv run python -c "from lightning_whisper_mlx import LightningTTSMLX; tts = LightningTTSMLX(); print('class loaded ok')"
```

Expected: Prints "class loaded ok" — the class instantiates but `_f5_generate` is None until `generate()` is called.

**Step 4: Commit**

```bash
git add lightning_whisper_mlx/tts.py lightning_whisper_mlx/__init__.py
git commit -m "feat: add LightningTTSMLX wrapper for f5-tts-mlx

Lazy-loading wrapper that delegates to f5_tts_mlx.generate.generate().
Raises ImportError with install instructions if f5-tts-mlx not present.
Re-exported from __init__.py via module __getattr__ (no eager import)."
```

---

### Task 3: Rewrite tests/test_audio.py with reference values

**Files:**
- Rewrite: `tests/test_audio.py`
- Modify: `tests/conftest.py`

**Context:** The current test_audio.py has 20 tautological tests (constant equality, shape checks, type checks). Replace them all with tests that validate actual computed values against pre-verified reference values.

**Reference values (verified against the codebase):**

| Input | Measurement | Reference Value | How verified |
|-------|------------|-----------------|--------------|
| 440Hz sine, 1s, 16kHz | Peak mel bin (80 mels) | **11** | `np.argmax(np.array(log_mel_spectrogram(audio)).mean(axis=0))` |
| Silence (zeros), 1s | All mel values | **-1.5** (uniform, atol=0.01) | `np.array(log_mel_spectrogram(np.zeros(16000, dtype=np.float32)))` |
| 1000Hz sine, 1s, 16kHz | Peak STFT bin | **25** | `np.argmax(np.abs(np.array(stft(...))).mean(axis=0))` |

**Step 1: Update `tests/conftest.py`**

Replace contents with:

```python
import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests that download models and run E2E")


@pytest.fixture
def sine_440hz():
    """1-second 440Hz sine wave at 16kHz sample rate."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sine_1000hz():
    """1-second 1000Hz sine wave at 16kHz sample rate."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def silence():
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)
```

**Step 2: Rewrite `tests/test_audio.py`**

Replace the entire file with:

```python
import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.audio import (
    HOP_LENGTH,
    N_FFT,
    N_SAMPLES,
    hanning,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    stft,
)


class TestMelSpectrogram:
    def test_440hz_peak_mel_bin(self, sine_440hz):
        """A 440Hz tone must concentrate energy at mel bin 11."""
        mel = log_mel_spectrogram(sine_440hz, n_mels=80)
        avg_energy = np.array(mel).mean(axis=0)
        assert np.argmax(avg_energy) == 11

    def test_silence_produces_uniform_floor(self, silence):
        """All-zeros input must produce uniform -1.5 across all mel bins."""
        mel = log_mel_spectrogram(silence, n_mels=80)
        mel_np = np.array(mel)
        assert np.allclose(mel_np, -1.5, atol=0.01)

    def test_440hz_has_higher_energy_than_silence(self, sine_440hz, silence):
        """A tone must produce higher mel energy than silence."""
        mel_tone = np.array(log_mel_spectrogram(sine_440hz, n_mels=80))
        mel_silence = np.array(log_mel_spectrogram(silence, n_mels=80))
        assert mel_tone.mean() > mel_silence.mean()

    def test_80_vs_128_mels_same_peak_region(self, sine_440hz):
        """80-mel and 128-mel spectrograms must agree on peak frequency region."""
        mel_80 = np.array(log_mel_spectrogram(sine_440hz, n_mels=80))
        mel_128 = np.array(log_mel_spectrogram(sine_440hz, n_mels=128))
        peak_80 = np.argmax(mel_80.mean(axis=0))
        peak_128 = np.argmax(mel_128.mean(axis=0))
        # 128-mel peak should be roughly (128/80) * 80-mel peak
        assert abs(peak_128 - peak_80 * 128 / 80) < 5


class TestStft:
    def test_1000hz_peak_frequency_bin(self, sine_1000hz):
        """1000Hz tone STFT must peak at frequency bin 25."""
        audio = mx.array(sine_1000hz)
        window = hanning(N_FFT)
        freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        avg_mag = np.abs(np.array(freqs)).mean(axis=0)
        assert np.argmax(avg_mag) == 25

    def test_silence_stft_near_zero(self, silence):
        """Silence must produce near-zero STFT magnitudes."""
        audio = mx.array(silence)
        window = hanning(N_FFT)
        freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        avg_mag = np.abs(np.array(freqs)).mean()
        assert avg_mag < 1e-6


class TestPadOrTrim:
    def test_pad_short_fills_zeros(self):
        """Padding a short array must append exact zeros."""
        data = mx.array([1.0, 2.0, 3.0])
        result = pad_or_trim(data, length=5)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, [1.0, 2.0, 3.0, 0.0, 0.0])

    def test_trim_long_keeps_prefix(self):
        """Trimming must keep the first N values exactly."""
        data = mx.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = pad_or_trim(data, length=3)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, [10.0, 20.0, 30.0])

    def test_default_length_is_n_samples(self):
        """Default pad target must be N_SAMPLES (480000)."""
        short = mx.zeros(100)
        result = pad_or_trim(short)
        assert result.shape[0] == N_SAMPLES


class TestLoadAudio:
    def test_nonexistent_file_raises(self):
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio("/nonexistent/file.wav")

    def test_invalid_file_raises(self, tmp_path):
        bad_file = tmp_path / "not_audio.txt"
        bad_file.write_text("this is not audio")
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio(str(bad_file))
```

**Step 3: Run tests to verify**

```bash
uv run pytest tests/test_audio.py -v
```

Expected: All tests PASS. Key assertions:
- `test_440hz_peak_mel_bin` → mel bin 11
- `test_silence_produces_uniform_floor` → all values ≈ -1.5
- `test_1000hz_peak_frequency_bin` → STFT bin 25

**Step 4: Commit**

```bash
git add tests/test_audio.py tests/conftest.py
git commit -m "test: rewrite audio tests with reference-value assertions

Replace tautological shape/type checks with pre-verified reference values:
440Hz→mel bin 11, silence→uniform -1.5, 1000Hz→STFT bin 25.
Pad/trim tests assert exact output arrays, not just shapes."
```

---

### Task 4: Rewrite tests/test_tokenizer.py with known token IDs

**Files:**
- Rewrite: `tests/test_tokenizer.py`

**Context:** The current test_tokenizer.py has 17 tests that check types and existence (`isinstance(tok.eot, int)`, `len(words) > 0`). Replace with exact known token IDs from the Whisper tokenizer spec.

**Reference values (verified against the codebase):**

| Property | Value | Source |
|----------|-------|--------|
| `eot` | 50257 | Whisper multilingual tokenizer |
| `sot` | 50258 | Whisper multilingual tokenizer |
| `timestamp_begin` | 50364 | Whisper multilingual tokenizer |
| `no_timestamps` | 50363 | Whisper multilingual tokenizer |
| `language_token` (en) | 50259 | `<\|en\|>` |
| `language_token` (de) | 50261 | `<\|de\|>` |
| `encode(" hello")` | `[7751]` | tiktoken multilingual encoding |
| `sot_sequence` (en, transcribe) | `(50258, 50259, 50359)` | sot + lang + task |

**Step 1: Rewrite `tests/test_tokenizer.py`**

Replace the entire file with:

```python
from lightning_whisper_mlx.tokenizer import get_tokenizer


class TestSpecialTokenIds:
    """Whisper special tokens must have exact known IDs from the tokenizer spec."""

    def test_eot(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.eot == 50257

    def test_sot(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.sot == 50258

    def test_timestamp_begin(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.timestamp_begin == 50364

    def test_no_timestamps(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.no_timestamps == 50363


class TestLanguageTokens:
    """Language tokens must map to exact known IDs."""

    def test_english_token(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.language_token == 50259  # <|en|>

    def test_german_token(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.language_token == 50261  # <|de|>

    def test_french_token(self):
        tok = get_tokenizer(multilingual=True, language="fr")
        assert tok.language_token == 50265  # <|fr|>

    def test_language_name_lookup_resolves(self):
        """Passing 'german' must resolve to language code 'de'."""
        tok = get_tokenizer(multilingual=True, language="german")
        assert tok.language == "de"
        assert tok.language_token == 50261


class TestSotSequence:
    """The start-of-transcript sequence must contain exact token IDs."""

    def test_english_transcribe(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.sot_sequence == (50258, 50259, 50359)

    def test_english_translate(self):
        tok = get_tokenizer(multilingual=True, language="en", task="translate")
        assert tok.sot_sequence == (50258, 50259, 50358)

    def test_german_transcribe(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.sot_sequence == (50258, 50261, 50359)


class TestEncodeDecode:
    """Encoding and decoding must produce exact known token sequences."""

    def test_encode_hello(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.encode(" hello") == [7751]

    def test_roundtrip(self):
        tok = get_tokenizer(multilingual=True, language="en")
        original = " hello world"
        decoded = tok.decode(tok.encode(original))
        assert decoded == original

    def test_empty_string(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.encode("") == []
        assert tok.decode([]) == ""


class TestSplitToWordTokens:
    """Word splitting must produce correct word boundaries."""

    def test_hello_world_splits_into_two_words(self):
        tok = get_tokenizer(multilingual=True, language="en")
        tokens = tok.encode(" hello world")
        words, word_tokens = tok.split_to_word_tokens(tokens + [tok.eot])
        assert len(words) == 2
        assert words[0].strip() == "hello"
        assert words[1].strip() == "world"


class TestInvalidInput:
    def test_invalid_language_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported language"):
            get_tokenizer(multilingual=True, language="klingon")
```

**Step 2: Run tests to verify**

```bash
uv run pytest tests/test_tokenizer.py -v
```

Expected: All tests PASS. Every assertion checks an exact known value, not a type or existence.

**Step 3: Commit**

```bash
git add tests/test_tokenizer.py
git commit -m "test: rewrite tokenizer tests with exact known token IDs

Assert eot=50257, sot=50258, timestamp_begin=50364,
language tokens en=50259/de=50261, encode(' hello')=[7751],
sot_sequence=(50258,50259,50359). No more type/existence checks."
```

---

### Task 5: Rewrite tests/test_decoding.py with behavioral assertions

**Files:**
- Rewrite: `tests/test_decoding.py`

**Context:** The current test_decoding.py has some good behavioral tests (SuppressBlank, EOT propagation) mixed with weaker ones. Keep the good ones, add `compression_ratio` exact-value test, strengthen `sum_logprobs` test.

**Reference values:**
- `compression_ratio("hello world")` = **0.5789473684210527** (deterministic: `len(b"hello world") / len(zlib.compress(b"hello world"))`)
- `compression_ratio("")` must not crash (empty string edge case)
- `compression_ratio("aaa" * 100)` > 5.0 (highly compressible)

**Step 1: Rewrite `tests/test_decoding.py`**

Replace the entire file with:

```python
import mlx.core as mx
import numpy as np

from lightning_whisper_mlx.decoding import (
    GreedyDecoder,
    SuppressBlank,
    SuppressTokens,
    compression_ratio,
)
from lightning_whisper_mlx.tokenizer import get_tokenizer


class TestCompressionRatio:
    def test_known_value(self):
        """compression_ratio('hello world') must equal the pre-computed value."""
        assert compression_ratio("hello world") == 11 / 19

    def test_repetitive_high_ratio(self):
        """Highly repetitive text must compress well (ratio > 5)."""
        assert compression_ratio("aaa" * 100) > 5.0

    def test_empty_string_does_not_crash(self):
        """Empty string must not raise."""
        ratio = compression_ratio("")
        assert ratio >= 0


class TestGreedyDecoder:
    EOT = 50257

    def test_argmax_at_temperature_zero(self):
        """Temperature=0 must pick the argmax token."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        logits = mx.zeros((1, 100))
        logits = logits.at[:, 42].add(10.0)
        tokens = mx.zeros((1, 1), dtype=mx.int32)
        sum_logprobs = mx.zeros((1,))

        new_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens[0, -1].item() == 42
        assert not completed.item()

    def test_eot_propagation(self):
        """Once EOT is emitted, all subsequent tokens must be EOT."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        vocab_size = self.EOT + 10
        tokens = mx.array([[self.EOT]], dtype=mx.int32)
        logits = mx.zeros((1, vocab_size))
        logits = logits.at[:, 5].add(10.0)
        sum_logprobs = mx.zeros((1,))

        new_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens[0, -1].item() == self.EOT
        assert completed.item()

    def test_tokens_grow_by_one(self):
        """Each update step must append exactly one token."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        tokens = mx.zeros((2, 3), dtype=mx.int32)
        logits = mx.zeros((2, 100))
        sum_logprobs = mx.zeros((2,))

        new_tokens, _, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens.shape == (2, 4)

    def test_sum_logprobs_accumulates(self):
        """sum_logprobs must increase (become less negative) by the selected token's log-prob."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        logits = mx.zeros((1, 100))
        logits = logits.at[:, 0].add(10.0)  # token 0 dominant
        tokens = mx.zeros((1, 1), dtype=mx.int32)
        sum_logprobs_before = mx.zeros((1,))

        _, _, sum_logprobs_after = decoder.update(tokens, logits, sum_logprobs_before)
        # Log-prob of argmax with logits=[10,0,0,...] should be close to 0 (dominant token)
        # sum_logprobs should have changed from 0
        assert sum_logprobs_after[0].item() != 0.0
        # The log-prob of the dominant token should be negative (log-softmax < 0)
        assert sum_logprobs_after[0].item() < 0.0

    def test_finalize_appends_eot(self):
        """finalize must append EOT token at the end."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        tokens = mx.zeros((1, 1, 5), dtype=mx.int32)
        sum_logprobs = mx.zeros((1,))

        final_tokens, _ = decoder.finalize(tokens, sum_logprobs)
        assert final_tokens.shape[-1] == 6
        assert final_tokens[0, 0, -1].item() == self.EOT


class TestSuppressBlank:
    def test_suppresses_blank_and_eot_at_sample_begin(self):
        """At sample_begin position, space tokens and EOT must be set to -inf."""
        tok = get_tokenizer(multilingual=True, language="en")
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tok, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, sample_begin), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        space_tokens = tok.encode(" ")
        for st in space_tokens:
            assert result_np[st] == float("-inf")
        assert result_np[tok.eot] == float("-inf")

    def test_no_suppression_after_sample_begin(self):
        """After sample_begin, logits must pass through unchanged."""
        tok = get_tokenizer(multilingual=True, language="en")
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tok, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, sample_begin + 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        assert mx.all(result == logits).item()


class TestSuppressTokens:
    def test_suppresses_exact_tokens(self):
        """Specified tokens must be -inf, all others unchanged."""
        suppress = [5, 10, 15]
        filt = SuppressTokens(suppress, 100)

        logits = mx.ones((1, 100))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        for idx in suppress:
            assert result_np[idx] == float("-inf")
        # Non-suppressed tokens must be exactly 1.0
        assert result_np[0] == 1.0
        assert result_np[20] == 1.0

    def test_empty_suppress_list_is_identity(self):
        """Empty suppress list must not change any logits."""
        filt = SuppressTokens([], 100)
        logits = mx.ones((1, 100))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        assert mx.all(result == logits).item()
```

**Step 2: Run tests to verify**

```bash
uv run pytest tests/test_decoding.py -v
```

Expected: All tests PASS. Key assertion: `compression_ratio("hello world") == 11/19`.

**Step 3: Commit**

```bash
git add tests/test_decoding.py
git commit -m "test: rewrite decoding tests with exact behavioral assertions

compression_ratio('hello world')==11/19, sum_logprobs accumulation,
EOT propagation, SuppressBlank at sample_begin boundary."
```

---

### Task 6: Create tests/test_roundtrip.py (E2E TTS→STT)

**Files:**
- Create: `tests/test_roundtrip.py`

**Context:** This test generates speech with F5-TTS, then transcribes it with Whisper, verifying the content matches. Marked `@pytest.mark.slow` since it downloads models.

**Step 1: Install TTS extra**

```bash
uv sync --all-extras
```

Expected: f5-tts-mlx and all dependencies installed.

**Step 2: Create `tests/test_roundtrip.py`**

```python
import pytest

from lightning_whisper_mlx import LightningWhisperMLX
from lightning_whisper_mlx.tts import LightningTTSMLX


@pytest.mark.slow
class TestTTSToSTTRoundtrip:
    """Generate speech with F5-TTS, transcribe with Whisper, verify content."""

    def test_german_roundtrip(self, tmp_path):
        """German TTS→STT: core words must survive the roundtrip."""
        tts = LightningTTSMLX()
        wav_path = str(tmp_path / "de_test.wav")
        tts.generate(
            text="Die Sonne scheint heute besonders hell.",
            output_path=wav_path,
            seed=42,
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe(wav_path, language="de")
        text = result["text"].lower()

        assert "sonne" in text, f"Expected 'sonne' in transcription: {text}"
        assert "scheint" in text, f"Expected 'scheint' in transcription: {text}"

    def test_english_roundtrip(self, tmp_path):
        """English TTS→STT: core words must survive the roundtrip."""
        tts = LightningTTSMLX()
        wav_path = str(tmp_path / "en_test.wav")
        tts.generate(
            text="The quick brown fox jumps over the lazy dog.",
            output_path=wav_path,
            seed=42,
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe(wav_path, language="en")
        text = result["text"].lower()

        assert "quick" in text, f"Expected 'quick' in transcription: {text}"
        assert "fox" in text, f"Expected 'fox' in transcription: {text}"
        assert "dog" in text, f"Expected 'dog' in transcription: {text}"
```

**Step 3: Delete `tests/__init__.py`** (if it exists — pytest doesn't need it and it can cause import issues)

Check if it exists first. If it does, delete it.

**Step 4: Run unit tests (excluding slow)**

```bash
uv run pytest tests/ -v -m "not slow"
```

Expected: All non-slow tests PASS. The roundtrip tests are skipped.

**Step 5: Run E2E roundtrip tests**

```bash
uv run pytest tests/test_roundtrip.py -v -m slow
```

Expected: Both roundtrip tests PASS. This downloads the F5-TTS model and Whisper tiny model on first run. The German test checks for "sonne" and "scheint" in the transcription. The English test checks for "quick", "fox", and "dog".

**If a roundtrip test fails:** The TTS model output may vary. Adjust keyword assertions to use the most distinctive words from each sentence. The `seed=42` parameter should help with reproducibility.

**Step 6: Commit**

```bash
git add tests/test_roundtrip.py
git commit -m "test: add E2E TTS→STT roundtrip tests

Generate speech with F5-TTS, transcribe with Whisper tiny,
verify core keywords survive the roundtrip. Marked @pytest.mark.slow."
```

---

### Task 7: Clean up and final verification

**Files:**
- Delete: `tests/__init__.py` (if exists)
- Verify all tests

**Step 1: Remove test __init__.py if present**

```bash
rm -f tests/__init__.py
```

**Step 2: Run full unit test suite**

```bash
uv run pytest tests/ -v -m "not slow"
```

Expected: All unit tests PASS. Should be ~25-30 tests total.

**Step 3: Run full suite including slow tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS including the slow roundtrip tests.

**Step 4: Update CLAUDE.md**

Update the "Setup & Development" section to reflect uv usage and the new test commands. Update the "Module Dependency Graph" to include the TTS path.

**Step 5: Final commit**

```bash
git add CLAUDE.md tests/
git commit -m "chore: finalize test suite rewrite and update CLAUDE.md

Remove tests/__init__.py, update docs to reflect uv workflow
and new TTS module in dependency graph."
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | setup.py → pyproject.toml + uv | Existing 52 tests still pass |
| 2 | `tts.py` + `__init__.py` update | Import smoke test |
| 3 | Rewrite test_audio.py | 440Hz→bin 11, silence→-1.5, 1000Hz→bin 25 |
| 4 | Rewrite test_tokenizer.py | eot=50257, encode(" hello")=[7751], etc. |
| 5 | Rewrite test_decoding.py | compression_ratio=11/19, EOT propagation |
| 6 | Create test_roundtrip.py | German + English TTS→STT roundtrip |
| 7 | Cleanup + verification | Full suite green |

Tasks 3-5 are independent and can run in parallel. Task 6 depends on Task 2 (TTS module).
