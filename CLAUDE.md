# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lightning Whisper MLX is a high-performance Whisper speech-to-text implementation optimized for Apple Silicon using the MLX framework. It achieves ~10x speedup over Whisper CPP through batched decoding, distilled models, and quantization.

## Setup & Development

```bash
uv sync                  # base dependencies
uv sync --extra dev      # + pytest
uv sync --extra tts      # + f5-tts-mlx
uv sync --all-extras     # everything
```

**Runtime dependency:** `ffmpeg` must be in PATH (used by `audio.py:load_audio` to decode audio files via subprocess).

**Tests:**
```bash
uv run pytest tests/ -v                 # unit tests only (~1s, slow tests excluded by default)
uv run pytest tests/ -v -m ''          # all tests incl E2E roundtrip (~40s, downloads models)
```

No linter, formatter, or CI configured. `tiktoken` is pinned to `0.3.3`. Build system: `pyproject.toml` with hatchling backend.

## Architecture

### Module Dependency Graph

```
__init__.py → lightning.py → transcribe.py → audio.py
                                            → load_models.py → whisper.py
                                            → decoding.py → tokenizer.py
                                            → timing.py
                                            → tokenizer.py
           → tts.py → f5_tts_mlx (external, optional)
```

### Public API

Two public classes re-exported from `__init__.py`:

- **`LightningWhisperMLX`** (`lightning.py`) — STT. Constructor downloads Whisper weights from HuggingFace Hub; `transcribe()` delegates to `transcribe_audio()`.
- **`LightningTTSMLX`** (`tts.py`) — TTS. Lazy-loaded via `__getattr__` to avoid requiring the optional `f5-tts-mlx` dependency. Wraps `f5_tts_mlx.generate.generate()`. Install with `uv sync --extra tts`.

### Model Registry (`lightning.py:models`)

Top-level dict maps model names to HF repo IDs. Structure: `models[name]["base"|"4bit"|"8bit"]`. Distilled models (`distil-*`) all point to `mustafaaljadery/distil-whisper-mlx` and only have a `"base"` key — quantization is not supported for distilled models and raises `ValueError`.

### MLX vs PyTorch Model

- `whisper.py` — **MLX model** used at runtime. `Whisper` class binds `decode` and `detect_language` from `decoding.py` as class-level attributes (mixin-style: `Whisper.decode = decode_function`), so `model.decode(mel, options)` dispatches to `decoding.py:decode()`.

### Transcription Pipeline (`transcribe.py`)

`transcribe_audio()` orchestrates end-to-end transcription:
1. `log_mel_spectrogram()` converts audio → mel (forced to CPU via `mx.set_default_device(mx.cpu)`)
2. Audio split into 30-second (`N_FRAMES=3000`) segments, grouped into batches of `batch_size`
3. Batched segments encoded through `AudioEncoder`, then decoded via `model.decode()` → `DecodingTask`
4. **Fallback logic** (`decode_with_fallback`): if compression ratio or avg log-prob thresholds fail, re-decodes individual segments at temperature=1.0
5. Results assembled with token-level timestamps

`ModelHolder` is a class-level singleton cache — calls `load_model()` only when model path changes.

The main transcription loop starts at a **negative seek** (`seek = -3000`) and advances by `N_FRAMES` per batch item, which is an intentional offset strategy.

### Decoding (`decoding.py`)

- `DecodingTask` owns the full decode lifecycle: `__init__` sets up tokenizer, inference engine, decoder, and logit filter chain; `run()` orchestrates encoder forward pass → autoregressive `_main_loop()` → ranking
- `Inference` class wraps the decoder forward pass with **KV-cache** management (stores/reuses key-value tensors across autoregressive steps)
- `GreedyDecoder` is the only implemented `TokenDecoder` (beam search raises `NotImplementedError`)
- **LogitFilter chain** applied each step: `SuppressBlank` → `SuppressTokens` → `ApplyTimestampRules` (enforces timestamp token pairing rules, monotonicity)
- `DecodingOptions` is a frozen dataclass; `DecodingResult` carries output tokens, text, probabilities

### Audio Processing (`audio.py`)

Custom STFT via `mx.fft.rfft` and `mx.as_strided` (no scipy/librosa dependency for spectrograms). Mel filter banks pre-computed in `assets/mel_filters.npz` (supports 80 and 128 mel bins). Key constants: `SAMPLE_RATE=16000`, `CHUNK_LENGTH=30s`, `HOP_LENGTH=160`, `N_FRAMES=3000`.

### Model Loading (`load_models.py`)

`load_model()` reads `config.json` + `weights.npz` from disk or HF Hub. Quantized models detected by `quantization` key in config; quantization applied via `nn.quantize()` with a class predicate checking for `{param_path}.scales` in the weight dict.

### Tokenizer (`tokenizer.py`)

`Tokenizer` wraps `tiktoken.Encoding` with Whisper special tokens (sot, eot, language tokens, timestamp tokens). Two vocab files in `assets/`: `gpt2.tiktoken` (English-only) and `multilingual.tiktoken`. `get_tokenizer()` is `@lru_cache`d.

### Word Timestamps (`timing.py`)

`add_word_timestamps()` → `find_alignment()` extracts cross-attention weights from `model.forward_with_cross_qk()`, applies median filtering (`scipy.signal.medfilt`), then DTW alignment (`dtw_cpu` is `@numba.jit` compiled). `merge_punctuations()` merges punctuation tokens into adjacent words.

## Key Conventions

- Models download to `./mlx_models/{model_name}/` relative to CWD (created at init time)
- Distilled models: different `hf_hub_download` path strategy (`filename` includes subdirectory, `local_dir="./"`) vs standard models (`filename` is just the file, `local_dir` is the model directory)
- All inference runs in fp16 by default (`DecodingOptions.fp16 = True`)
- `transcribe()` returns `dict` with keys: `text`, `segments` (list of `[start_seek, end_seek, text]`), `language`

## Gotchas

- **`batch_size` defaults to 12** in both `transcribe_audio()` and `LightningWhisperMLX.__init__()`.
- **`word_timestamps` is intentionally deferred.** The parameter is accepted by `transcribe_audio()` and the full implementation exists in `timing.py` (`add_word_timestamps`, `find_alignment`, DTW alignment), but is never called. Activation was deferred because it requires an extra `forward_with_cross_qk()` pass per 30-second window (~30-50% overhead). Segment-level timestamps (from Whisper's timestamp tokens) are sufficient for current use cases including diarization. See `docs/plans/` for activation notes.
- **`./mlx_models/` is created relative to CWD.** Running from different directories will create model caches in different locations. This is not user-home-based.

## Supported Models

Models: `tiny`, `small`, `distil-small.en`, `base`, `medium`, `distil-medium.en`, `large`, `large-v2`, `distil-large-v2`, `large-v3`, `distil-large-v3`

Quantization: `None`, `"4bit"`, `"8bit"` (raises `ValueError` for `distil-*` models)
