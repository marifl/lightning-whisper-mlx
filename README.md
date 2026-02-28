# Lightning Whisper MLX

An incredibly fast implementation of Whisper optimized for Apple Silicon.

![Whisper Decoding Speed](./speed_image.png)

10x faster than Whisper CPP, 4x faster than current MLX Whisper implementation.

## Features

- **Batched Decoding** -> Higher Throughput
- **Distilled Models** -> Faster Decoding (less layers)
- **Quantized Models** -> Faster Memory Movement
- **Speaker Diarization** -> Multi-speaker detection via pyannote-audio
- **LLM Text Correction** -> Post-processing with local or API-based LLMs

## Installation

```bash
pip install lightning-whisper-mlx
```

Optional extras:

```bash
pip install lightning-whisper-mlx[diarize]      # + speaker diarization (pyannote-audio)
pip install lightning-whisper-mlx[correct]       # + local LLM correction (mlx-lm)
pip install lightning-whisper-mlx[correct-api]   # + API-based correction (anthropic)
pip install lightning-whisper-mlx[tts]           # + text-to-speech (f5-tts-mlx)
pip install lightning-whisper-mlx[server]        # + FastAPI server / web UI
```

## Usage

### Models

```
tiny, small, distil-small.en, base, medium, distil-medium.en, large, large-v2, distil-large-v2, large-v3, distil-large-v3
```

### Quantization

```
None, "4bit", "8bit"
```

> Quantization is not supported for distilled models (`distil-*`) and will raise a `ValueError`.

### Basic Transcription

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-medium.en", batch_size=12, quant=None)

result = whisper.transcribe(audio_path="/audio.mp3")
print(result['text'])
```

### Speaker Diarization

Identify who is speaking in multi-speaker audio. Requires `pyannote-audio` and a [HuggingFace token](https://huggingface.co/settings/tokens) with access to the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model.

```bash
pip install lightning-whisper-mlx[diarize]
export HF_TOKEN="hf_..."
```

```python
whisper = LightningWhisperMLX(model="distil-large-v3")

result = whisper.transcribe("/audio.mp3", diarize=True)

for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text']}")
```

You can constrain the number of speakers:

```python
result = whisper.transcribe("/audio.mp3", diarize=True, num_speakers=2)
# or: min_speakers=1, max_speakers=4
```

### LLM Text Correction

Improve transcription quality with LLM-based post-processing — fixes grammar, spelling, punctuation, and domain-specific terminology.

#### Using Anthropic API

```bash
pip install lightning-whisper-mlx[correct-api]
export ANTHROPIC_API_KEY="sk-..."
```

```python
result = whisper.transcribe("/audio.mp3", correct=True)
# Uses Claude Haiku by default
```

#### Using a local MLX model

```bash
pip install lightning-whisper-mlx[correct]
```

```python
result = whisper.transcribe("/audio.mp3", correct=True, correct_backend="local")
# Uses Llama-3.2-3B-Instruct-4bit by default (downloads on first run)
```

#### Using a custom function

```python
def my_corrector(text: str) -> str:
    # Your own correction logic
    return text.replace("gonna", "going to")

result = whisper.transcribe("/audio.mp3", correct=True, correct_backend="custom", correct_fn=my_corrector)
```

#### Domain glossary

Provide domain-specific terms the LLM should preserve unchanged:

```python
result = whisper.transcribe(
    "/audio.mp3",
    correct=True,
    glossary=["Kubernetes", "kubectl", "MLX"],
)
```

#### Combining diarization and correction

```python
result = whisper.transcribe(
    "/meeting.mp3",
    diarize=True,
    correct=True,
    glossary=["PyAnnote", "Whisper"],
)

for seg in result["segments"]:
    print(f"{seg['speaker']}: {seg['text']}")
    # Original uncorrected text available in seg['raw_text']
```

## API Reference

### `LightningWhisperMLX(model, batch_size=12, quant=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model name (see Models above) |
| `batch_size` | `int` | `12` | Batch size for decoding. Higher = faster but more memory |
| `quant` | `str\|None` | `None` | Quantization: `None`, `"4bit"`, or `"8bit"` |

### `whisper.transcribe(audio_path, ...)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_path` | `str` | required | Path to audio file |
| `language` | `str\|None` | `None` | Language code (auto-detected if None) |
| `diarize` | `bool` | `False` | Enable speaker diarization |
| `num_speakers` | `int\|None` | `None` | Exact number of speakers |
| `min_speakers` | `int\|None` | `None` | Minimum speakers |
| `max_speakers` | `int\|None` | `None` | Maximum speakers |
| `correct` | `bool` | `False` | Enable LLM text correction |
| `correct_backend` | `str` | `"anthropic"` | `"local"`, `"anthropic"`, or `"custom"` |
| `correct_model` | `str\|None` | `None` | Override the default LLM model |
| `glossary` | `list[str]\|None` | `None` | Domain terms to preserve |
| `correct_fn` | `callable\|None` | `None` | Custom correction function (for `"custom"` backend) |

## Notes

- The default `batch_size` is `12`. Higher is better for throughput but may cause memory issues on larger models. Scale down for `large` variants based on your unified memory.
- Diarization requires `HF_TOKEN` environment variable. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- LLM correction with the Anthropic backend requires `ANTHROPIC_API_KEY` environment variable.
- Pipeline order: transcription -> diarization (optional) -> correction (optional).

## Credits

- [Mustafa](https://github.com/mustafaaljadery) - Creator of Lightning Whisper MLX
- [Awni](https://github.com/awni) - Implementation of Whisper MLX (I built on top of this)
- [Vaibhav](https://github.com/Vaibhavs10) - Inspired me to build this (He created a version optimized for Cuda)
