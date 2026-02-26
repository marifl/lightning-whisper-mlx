# Lightning Whisper MLX: Dead Code Cleanup, Bug Fixes & Feature Roadmap

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform lightning-whisper-mlx from a buggy prototype with dead code into a production-grade Whisper product with multi-speaker detection and LLM-based text correction.

**Architecture:** 4-phase approach: (1) remove dead code & fix critical bugs (**done**), (2) add test infrastructure, (3) integrate speaker diarization via pyannote-audio (segment-level alignment), (4) add LLM-based post-processing pipeline. Word-level timestamps deferred — segment-level timestamps are sufficient for diarization and current use cases.

**Tech Stack:** Python 3.10+, MLX, pyannote-audio (diarization), mlx-lm or Anthropic API (LLM correction), pytest

---

## Audit Summary: Current State of the Codebase

### Dead Code Inventory

| # | What | Where | Lines | Impact |
|---|------|-------|-------|--------|
| D1 | `torch_whisper.py` (entire file) | `lightning_whisper_mlx/torch_whisper.py` | 308 | PyTorch reference model. Never imported anywhere. Causes `torch` to be a required dependency (~500MB). |
| D2 | `torch` dependency | `setup.py:15` | 1 | Only used by D1. Wastes 500MB+ install size. |
| D3 | `more-itertools` dependency | `setup.py:17` | 1 | Listed in install_requires but never imported in any module. |
| D4 | `import tqdm` | `transcribe.py:9` | 1 | Imported but never called. Was likely used for progress bars at some point. |
| D5 | `import time` | `transcribe.py:10` | 1 | Imported but never used. |
| D6 | Temperature branching in GreedyDecoder | `decoding.py:263-266` | 4 | The if/else for temperature==0 vs sampling is immediately overwritten by line 268 `next_tokens = mx.argmax(logits, axis=-1)`. Temperature-based sampling is completely dead. |
| D7 | `word_timestamps` parameter path | `transcribe.py:74,203-204` | ~15 | Parameter accepted, `add_word_timestamps` imported (line 23), warning issued (line 203), but the function is **never actually called**. The entire word-timestamp pipeline in `timing.py` (330 lines) is unreachable. |
| D8 | `hallucination_silence_threshold` parameter | `transcribe.py:78` | 1 | Accepted as parameter, documented in docstring, but never referenced in function body. |
| D9 | `_get_end()` helper | `transcribe.py:44-48` | 5 | Defined but never called. Accesses `s["words"]` which would require word_timestamps to be active. |
| D10 | `word_anomaly_score`, `is_segment_anomaly`, `next_words_segment` | `transcribe.py:295-318` | 24 | Nested functions inside `format_output()`. Defined but only useful for word_timestamps path (which is dead). Currently just dead weight inside the hot path. |
| D11 | Beam search commented-out code | `decoding.py:436-438` | 3 | Commented-out `BeamSearchDecoder` instantiation. |
| D12 | `segment["words"] = []` in format_output | `transcribe.py:392` | 1 | Sets "words" key on segments, but word_timestamps is never called so this key is meaningless. |

### Bugs

| # | What | Where | Severity | Detail |
|---|------|-------|----------|--------|
| B1 | `batch_size: 6` is a type annotation, not a default | `transcribe.py:79` | **HIGH** | Python treats `batch_size: 6` as annotating batch_size with literal type 6, not as `batch_size: int = 6`. Callers must always pass batch_size explicitly. README falsely claims default is 12. |
| B2 | Temperature sampling never works | `decoding.py:268` | **HIGH** | `next_tokens = mx.argmax(logits, axis=-1)` on line 268 unconditionally overwrites the temperature-aware branching on lines 263-266. Temperature fallback in `decode_with_fallback` (temperature=1.0) is supposed to use sampling but silently uses greedy. |
| B3 | Distilled model quantization is broken | `lightning.py:67-88` | **MEDIUM** | When `quant="4bit"` and model is distilled, code downloads base (unquantized) weights but saves them to a `-4-bit` directory. `load_model` will then try to apply quantization to weights that don't have `.scales` entries, meaning the quantization predicate in `load_models.py:36-38` won't match anything. Silently runs unquantized. |

### Implementation Gaps for Target Product

| # | Feature | Status | Priority for Product |
|---|---------|--------|---------------------|
| G1 | Word-level timestamps | Code exists in `timing.py` but never called | **DEFERRED** - segment-level timestamps sufficient for diarization; word-level adds ~30-50% overhead per window via extra forward pass |
| G2 | Multi-speaker detection (diarization) | Not present | **CRITICAL** - core requested feature |
| G3 | LLM-based text correction | Not present | **CRITICAL** - core requested feature |
| G4 | Beam search decoding | `NotImplementedError` | **LOW** - greedy works well for Whisper |
| G5 | Speculative decoding | README says "Coming Soon" - no code | **LOW** - optimization, not needed for MVP |
| G6 | Test suite | Only manual `test.py` | **HIGH** - needed before any safe refactoring |
| G7 | Progress reporting | `tqdm` imported but unused | **MEDIUM** - UX improvement |
| G8 | Proper error handling | Missing throughout | **MEDIUM** - production readiness |

---

## Priority Assessment & Phased Execution Plan

### Phase 1: Dead Code Removal & Bug Fixes (Foundation) — COMPLETE
**Why first:** Nothing else is safe to build on top of a codebase with silent bugs and 500MB of unnecessary dependencies.

### ~~Phase 2: Activate Word-Level Timestamps~~ — DEFERRED
**Reason deferred:** Word-level timestamps require an extra forward pass per 30-second window (`forward_with_cross_qk`), adding ~30-50% overhead. Segment-level timestamps (already produced by Whisper's timestamp tokens) are sufficient for diarization and current use cases. The code in `timing.py` remains intact and can be activated later if word-level precision is needed.

### Phase 2: Test Infrastructure (was Phase 3)
**Why next:** Before adding complex new features (diarization, LLM), we need regression safety nets.

### Phase 3: Multi-Speaker Detection (Diarization) (was Phase 4)
**Why third:** Core requested feature. Uses **segment-level alignment** — assigns the dominant speaker per transcription segment based on temporal overlap with pyannote diarization output. No dependency on word-level timestamps.

### Phase 4: LLM-Based Text Correction (was Phase 5)
**Why fourth:** Post-processing layer that operates on completed transcription+diarization output.

---

## Phase 1: Dead Code Removal & Bug Fixes

### Task 1.1: Remove `torch_whisper.py` and `torch` dependency

**Files:**
- Delete: `lightning_whisper_mlx/torch_whisper.py`
- Modify: `setup.py:15` (remove `"torch"` from install_requires)
- Modify: `setup.py:17` (remove `"more-itertools"` from install_requires)

**Step 1: Delete torch_whisper.py**

```bash
git rm lightning_whisper_mlx/torch_whisper.py
```

**Step 2: Verify no imports reference it**

Run: `grep -r "torch_whisper" lightning_whisper_mlx/`
Expected: No matches

**Step 3: Remove torch and more-itertools from setup.py**

Edit `setup.py` install_requires to:
```python
install_requires=[
    'huggingface_hub',
    "mlx",
    "numba",
    "numpy",
    "tqdm",
    "tiktoken==0.3.3",
    "scipy"
]
```

Note: We keep `tqdm` because we will wire it up for progress reporting later.

**Step 4: Commit**

```bash
git add -A && git commit -m "chore: remove dead torch_whisper.py and unused deps (torch, more-itertools)"
```

---

### Task 1.2: Fix GreedyDecoder temperature bug (B2)

**Files:**
- Modify: `lightning_whisper_mlx/decoding.py:255-280`

**Step 1: Read the buggy code**

Current code at `decoding.py:260-268`:
```python
def update(self, tokens, logits, sum_logprobs):
    if self.temperature == 0:
        next_tokens = logits.argmax(axis=-1)
    else:
        next_tokens = mx.random.categorical(logits=logits / self.temperature)

    next_tokens = mx.argmax(logits, axis=-1)  # BUG: overwrites above
```

**Step 2: Fix by removing the duplicate line**

Replace with:
```python
def update(
    self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
) -> Tuple[mx.array, bool, mx.array]:
    if self.temperature == 0:
        next_tokens = logits.argmax(axis=-1)
    else:
        next_tokens = mx.random.categorical(logits=logits / self.temperature)

    logits = logits.astype(mx.float32)
```

The key change: **remove line 268** (`next_tokens = mx.argmax(logits, axis=-1)`). The rest of the method stays the same.

**Step 3: Commit**

```bash
git add lightning_whisper_mlx/decoding.py
git commit -m "fix: temperature-based sampling was silently overwritten by greedy argmax"
```

---

### Task 1.3: Fix batch_size type annotation bug (B1)

**Files:**
- Modify: `lightning_whisper_mlx/transcribe.py:79`

**Step 1: Fix the annotation**

Change line 79 from:
```python
    batch_size: 6,
```
to:
```python
    batch_size: int = 12,
```

This gives a proper type hint AND a sensible default (matching README's claim).

**Step 2: Commit**

```bash
git add lightning_whisper_mlx/transcribe.py
git commit -m "fix: batch_size was type-annotated as literal 6 instead of having a default value"
```

---

### Task 1.4: Remove unused imports in transcribe.py (D4, D5)

**Files:**
- Modify: `lightning_whisper_mlx/transcribe.py:9-10`

**Step 1: Remove unused imports**

Remove these lines from the import block:
```python
import tqdm
import time
```

Keep `tqdm` in setup.py for future use (Phase 2 progress bars).

**Step 2: Commit**

```bash
git add lightning_whisper_mlx/transcribe.py
git commit -m "chore: remove unused tqdm and time imports from transcribe.py"
```

---

### Task 1.5: Fix distilled model quantization logic (B3)

**Files:**
- Modify: `lightning_whisper_mlx/lightning.py:56-88`

**Step 1: Understand the problem**

When user requests `quant="4bit"` on a distilled model, the code:
1. Downloads base (unquantized) weights from `mustafaaljadery/distil-whisper-mlx`
2. Saves to `./mlx_models/distil-small.en-4-bit/`
3. `load_model` tries quantization but finds no `.scales` in weights - does nothing

**Step 2: Add clear error for unsupported quantization**

Replace the constructor logic to reject quantization for distilled models explicitly:

```python
def __init__(self, model, batch_size=12, quant=None):
    if quant and (quant != "4bit" and quant != "8bit"):
        raise ValueError("Quantization must be `4bit` or `8bit`")

    if model not in models:
        raise ValueError("Please select a valid model")

    if quant and "distil" in model:
        raise ValueError(
            f"Quantization is not supported for distilled model '{model}'. "
            "Distilled models are already optimized for speed. Use quant=None."
        )

    self.name = model
    self.batch_size = batch_size

    if quant:
        repo_id = models[model][quant]
    else:
        repo_id = models[model]['base']

    if "distil" in model:
        filename1 = f"./mlx_models/{self.name}/weights.npz"
        filename2 = f"./mlx_models/{self.name}/config.json"
        local_dir = "./"
    else:
        filename1 = "weights.npz"
        filename2 = "config.json"
        local_dir = f"./mlx_models/{self.name}"

    hf_hub_download(repo_id=repo_id, filename=filename1, local_dir=local_dir)
    hf_hub_download(repo_id=repo_id, filename=filename2, local_dir=local_dir)
```

**Step 3: Commit**

```bash
git add lightning_whisper_mlx/lightning.py
git commit -m "fix: reject quantization for distilled models instead of silently ignoring it"
```

---

### Task 1.6: Clean up dead code in transcribe.py (D7-D12)

**Files:**
- Modify: `lightning_whisper_mlx/transcribe.py`

**Step 1: Remove `_get_end()` function (lines 44-48)**

Delete the `_get_end()` helper function - it's never called.

**Step 2: Remove `hallucination_silence_threshold` parameter (line 78)**

Remove from the function signature and docstring.

**Step 3: Remove dead word_timestamps helper functions inside format_output**

Remove `word_anomaly_score()`, `is_segment_anomaly()`, and `next_words_segment()` from inside `format_output()`.

**Step 4: Remove `segment["words"] = []` from format_output** (line 392)

This sets a meaningless key since word_timestamps is never called.

**Step 5: Keep `word_timestamps` parameter and `add_word_timestamps` import**

We will activate this in Phase 2, so don't remove the parameter or the import.

**Step 6: Commit**

```bash
git add lightning_whisper_mlx/transcribe.py
git commit -m "chore: remove dead helpers (_get_end, hallucination_silence_threshold, word anomaly functions)"
```

---

## ~~Phase 2: Activate Word-Level Timestamps~~ — DEFERRED

> **Status:** Deferred. The code in `timing.py` (`add_word_timestamps`, `find_alignment`, `merge_punctuations`, DTW alignment) is fully implemented and preserved. Activation requires: (1) refactoring `all_segments` to keep rich segment dicts instead of `[start, end, text]` tuples, (2) calling `add_word_timestamps()` per batch item with the mel window, (3) fixing `new_segment()` closure bug with `seek`. See git history for the original task description.
>
> **Cost:** ~30-50% overhead per 30-second window due to extra `forward_with_cross_qk()` pass.
>
> **Trigger to revisit:** If word-level speaker alignment is needed for diarization accuracy, or for subtitle generation with per-word highlighting.

---

## Phase 2: Test Infrastructure

### Task 2.1: Set up pytest and basic test structure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_audio.py`
- Create: `tests/test_tokenizer.py`
- Create: `tests/test_decoding.py`
- Modify: `setup.py` (add test extras)

**Step 1: Create test directory and conftest**

```python
# tests/conftest.py
import pytest
import numpy as np
import mlx.core as mx

@pytest.fixture
def sample_audio():
    """Generate a 1-second sine wave at 440Hz as test audio."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)

@pytest.fixture
def sample_mel(sample_audio):
    """Generate mel spectrogram from sample audio."""
    from lightning_whisper_mlx.audio import log_mel_spectrogram
    return log_mel_spectrogram(mx.array(sample_audio))
```

**Step 2: Add unit tests for audio.py**

Test `pad_or_trim()`, `stft()`, `log_mel_spectrogram()`, `load_audio()` error handling.

**Step 3: Add unit tests for tokenizer.py**

Test `get_tokenizer()`, `encode()/decode()` round-trip, special token properties.

**Step 4: Add unit tests for decoding.py**

Test `GreedyDecoder` with temperature=0 and temperature>0, `SuppressBlank`, `SuppressTokens`, `compression_ratio()`.

**Step 5: Commit**

```bash
git add tests/ setup.py
git commit -m "test: add pytest infrastructure and unit tests for audio, tokenizer, decoding"
```

---

## Phase 3: Multi-Speaker Detection (Diarization)

### Task 3.1: Design diarization architecture

**Architecture decision:** Use `pyannote-audio` for speaker diarization. It provides pre-trained models for speaker segmentation and embedding. The pipeline will be:

1. Transcribe audio with segment-level timestamps (existing pipeline)
2. Run pyannote diarization to get speaker segments with timestamps
3. Align transcription segments to speaker segments via **temporal overlap** (dominant speaker per segment)
4. Output enriched transcript with `speaker` field per segment

**Alignment approach (segment-level):** For each transcription segment (1-10 seconds), compute the overlap with each pyannote speaker segment. Assign the speaker with the greatest overlap duration. This is less precise than word-level alignment at speaker boundaries mid-sentence, but sufficient for most use cases (meetings, interviews, podcasts).

**Files:**
- Create: `lightning_whisper_mlx/diarize.py`
- Modify: `lightning_whisper_mlx/lightning.py` (add `diarize` parameter)
- Modify: `setup.py` (add optional `pyannote-audio` dependency)

### Task 3.2: Implement diarization module

**Files:**
- Create: `lightning_whisper_mlx/diarize.py`

```python
# diarize.py - Speaker diarization using pyannote-audio
from typing import List, Dict, Optional, Tuple

def diarize_audio(
    audio_path: str,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> List[Dict]:
    """
    Perform speaker diarization on an audio file.

    Returns list of dicts with keys: speaker, start, end
    """
    ...

def assign_speakers_to_segments(
    segments: List[Dict],
    diarization: List[Dict],
) -> List[Dict]:
    """
    Assign speaker labels to transcription segments based on
    temporal overlap between diarization output and segment timestamps.

    For each segment, the speaker with the greatest overlap duration
    is assigned. This is segment-level alignment (not word-level).
    """
    ...
```

### Task 3.3: Integrate diarization into public API

**Files:**
- Modify: `lightning_whisper_mlx/lightning.py`

Add `diarize=False` parameter to `LightningWhisperMLX.transcribe()`. When enabled:
1. Run diarization after transcription
2. Assign speakers to segments via temporal overlap
3. Add `speaker` field to each segment in output

### Task 3.4: Add diarization tests

**Files:**
- Create: `tests/test_diarize.py`

Test speaker assignment logic with mock diarization output. Test edge cases: single speaker, overlapping speech, silence gaps, speaker change mid-segment.

**Step: Commit**

```bash
git add lightning_whisper_mlx/diarize.py tests/test_diarize.py lightning_whisper_mlx/lightning.py setup.py
git commit -m "feat: add multi-speaker detection via pyannote-audio diarization"
```

---

## Phase 4: LLM-Based Text Correction

### Task 4.1: Design LLM correction pipeline

**Architecture:** Post-processing layer that takes raw Whisper transcription and sends it through an LLM for:
- Punctuation restoration and normalization
- Grammar correction
- Domain-specific vocabulary correction (via user-provided glossary)
- Formatting (capitalization, number formatting)

Support multiple backends:
1. Local MLX model via `mlx-lm` (offline, fast on Apple Silicon)
2. Anthropic API (higher quality, requires network)
3. Custom callable (user provides their own function)

### Task 4.2: Implement correction module

**Files:**
- Create: `lightning_whisper_mlx/correct.py`

```python
# correct.py - LLM-based text correction
from typing import List, Dict, Optional, Callable

def correct_transcription(
    segments: List[Dict],
    *,
    backend: str = "local",  # "local", "anthropic", "custom"
    model: Optional[str] = None,
    glossary: Optional[List[str]] = None,
    custom_fn: Optional[Callable] = None,
    language: str = "en",
) -> List[Dict]:
    """
    Apply LLM-based correction to transcription segments.

    Processes segments in batches, preserving timestamps.
    Returns corrected segments with original text preserved
    in 'raw_text' field.
    """
    ...

def build_correction_prompt(
    text: str,
    language: str,
    glossary: Optional[List[str]] = None,
) -> str:
    """Build the correction prompt for the LLM."""
    ...
```

### Task 4.3: Integrate correction into public API

**Files:**
- Modify: `lightning_whisper_mlx/lightning.py`

Add `correct=False` and `glossary=None` parameters to `transcribe()`. When enabled, run correction as final post-processing step.

### Task 4.4: Add correction tests

**Files:**
- Create: `tests/test_correct.py`

Test prompt construction, glossary integration, segment preservation. Mock LLM calls for unit testing.

**Step: Commit**

```bash
git add lightning_whisper_mlx/correct.py tests/test_correct.py lightning_whisper_mlx/lightning.py
git commit -m "feat: add LLM-based text correction post-processing"
```

---

## Phase 5: Update Documentation & Dependencies

### Task 5.1: Update README.md

- Fix "default batch_size is 12" claim (now accurate after our fix)
- Remove "Coming Soon: Speculative Decoding" or update status
- Add diarization usage examples
- Add LLM correction usage examples
- Add new API parameters documentation

### Task 5.2: Update setup.py

Final install_requires:
```python
install_requires=[
    'huggingface_hub',
    "mlx",
    "numba",
    "numpy",
    "tqdm",
    "tiktoken>=0.3.3",
    "scipy",
]
extras_require={
    "diarize": ["pyannote-audio>=3.0"],
    "correct": ["mlx-lm"],
    "correct-api": ["anthropic"],
    "all": ["pyannote-audio>=3.0", "mlx-lm", "anthropic"],
    "dev": ["pytest", "pytest-cov"],
}
```

### Task 5.3: Update CLAUDE.md

Reflect all architectural changes, new modules, new dependencies, and removed dead code.

---

## Execution Sequence & Dependencies

```
Phase 1 (Foundation) — COMPLETE ✓
  Task 1.1 → Task 1.2 → Task 1.3 → Task 1.4 → Task 1.5 → Task 1.6

Word Timestamps — DEFERRED (code preserved in timing.py)

Phase 2 (Tests)
  Task 2.1 (depends on Phase 1 complete)

Phase 3 (Diarization) — segment-level alignment
  Task 3.1 → Task 3.2 → Task 3.3 → Task 3.4
  (depends on Phase 1 complete, can parallel with Phase 2)

Phase 4 (LLM Correction)
  Task 4.1 → Task 4.2 → Task 4.3 → Task 4.4
  (depends on Phase 1 complete, can parallel with Phase 3)

Phase 5 (Docs)
  Task 5.1-5.3 (depends on all phases complete)
```

## Estimated Effort

| Phase | Tasks | Complexity | Status |
|-------|-------|------------|--------|
| Phase 1: Dead Code & Bugs | 6 tasks | Low | **COMPLETE** |
| ~~Word Timestamps~~ | 1 task | Medium | **DEFERRED** |
| Phase 2: Tests | 1 task | Medium - writing comprehensive test suite | Pending |
| Phase 3: Diarization | 4 tasks | High - new feature, external dependency integration | Pending |
| Phase 4: LLM Correction | 4 tasks | High - new feature, multiple backend support | Pending |
| Phase 5: Documentation | 3 tasks | Low - documentation updates | Pending |
