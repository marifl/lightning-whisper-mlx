# Multi-Speaker Dialog TTS Design

**Goal:** Lifelike multi-speaker dialogue TTS on Apple Silicon, compatible with ElevenLabs V3 dialogue format, using F5-TTS MLX locally.

**Key Constraint:** F5-TTS quality degrades above ~135 characters per chunk. Text must be split and audio concatenated server-side.

---

## Decisions

| Decision | Choice |
|----------|--------|
| Dialogue format | ElevenLabs-compatible `[tag] Text...` with `voice_id` per input |
| Tag handling | Strip all `[tags]` before F5-TTS generation (F5 has no tag support) |
| Speaker management | Server-side API with persistent ref-audio storage |
| UI location | Dedicated `/tts/dialog` route |
| Chunking approach | Sequential generation with WAV concatenation |
| Storage | `./speakers/{id}/` on server filesystem |

---

## 1. Backend: Speaker API

### Endpoints

```
POST   /api/speakers           — Create speaker (name + ref_audio + ref_text)
GET    /api/speakers           — List all speakers
DELETE /api/speakers/{id}      — Delete speaker
```

### Storage Structure

```
./speakers/
  {uuid}/
    ref_audio.wav    (mono, 24kHz, converted on upload via ffmpeg)
    metadata.json    {"id", "name", "ref_text", "created_at"}
```

### Upload Processing

On `POST /api/speakers`, the uploaded audio is converted via ffmpeg:
```
ffmpeg -i input -ac 1 -ar 24000 -sample_fmt s16 ref_audio.wav
```

---

## 2. Backend: Dialog TTS Pipeline

### Endpoint

```
POST /api/tts/dialog    (status 202, async job)
```

### Request Body (JSON)

```json
{
  "segments": [
    {"speaker_id": "abc123", "text": "[warmly] Geh auf Folie zwölf."},
    {"speaker_id": "def456", "text": "[curious] Mhm, bin da. Was sehe ich?"}
  ],
  "steps": 8,
  "speed": 1.0,
  "pause_between_ms": 300,
  "model": "eamag/f5-tts-mlx-german"
}
```

### Processing Pipeline (`_run_dialog_tts`)

1. **Parse & Strip Tags:** Regex `\[.*?\]` removes all bracket tags from text
2. **Chunk per Segment:** Split at `;:,.!?` boundaries, max 135 UTF-8 bytes per chunk
3. **Resolve Speakers:** Load `ref_audio.wav` and `ref_text` for each speaker_id
4. **Sequential Generation:** For each chunk, call `f5_tts_mlx.generate()` with the speaker's ref_audio
5. **Insert Silence:** Between speaker changes, insert `pause_between_ms` of silence (`np.zeros`)
6. **Concatenate:** `np.concatenate()` all chunk WAVs + silence into final output
7. **Write:** `soundfile.write(output_path, audio, 24000)` as mono WAV
8. **Progress:** Report `"Generating segment {n}/{total}..."` with percent

### Tag Stripping

```python
import re
def strip_tags(text: str) -> str:
    return re.sub(r'\[.*?\]', '', text).strip()
```

### Chunk Splitting

Reuses F5-TTS's approach: split at `(?<=[;:,.!?])\s+`, max 135 UTF-8 bytes per chunk.

### Silence Insertion

```python
silence = np.zeros(int(24000 * pause_ms / 1000), dtype=np.float32)
```

Only inserted between segments (speaker changes), not between chunks of the same segment.

---

## 3. Frontend: Settings — Speakers Section

New section 5 on `/settings` page: **"Speakers"**

- Lists all speakers from `GET /api/speakers`
- Each card shows: name, filename, ref_text preview, delete button
- "Add Speaker" form: name input, file upload (accepts audio/*), ref_text textarea
- Requirements note: "Mono WAV, 5-15 seconds, clear without background noise"
- Upload calls `POST /api/speakers`, refreshes list
- Delete with confirmation

---

## 4. Frontend: Dialog Page (`/tts/dialog`)

Dedicated route with two input modes:

### Visual Editor (default)

- List of dialog lines, each with:
  - Speaker dropdown (populated from speaker library)
  - Text input with character counter (shows `n / 135`, red when over)
  - Remove line button
- "+ Add Line" button appends new empty line
- Parameters: steps, speed, pause_between_ms
- "Generate Dialog" button submits to `POST /api/tts/dialog`

### JSON Import

- Textarea for pasting ElevenLabs-compatible JSON
- Format: `{"inputs": [{"text": "...", "voice_id": "..."}]}`
- Parser maps `voice_id` to speaker names (configurable mapping)
- Strips tags, splits into dialog lines for the visual editor

### Progress & Playback

- Same polling mechanism as existing TTS (`/api/jobs/{id}`)
- Progress bar: "Generating segment 3/12... 25%"
- Audio player + download button on completion
- Same audio endpoint: `GET /api/tts-jobs/{id}/audio`

---

## 5. Sidebar Navigation Update

Add "Dialog" sub-item under TTS group in sidebar:

```
TTS
  ├── Text Input     → /tts
  └── Dialog         → /tts/dialog
```

---

## 6. ElevenLabs Compatibility

The system accepts ElevenLabs V3 dialogue JSON format:

```json
{
  "inputs": [
    {"text": "[warmly] Geh auf Folie zwölf. [long pause]", "voice_id": "vmVmHDKBkkCgbLVIOJRb"},
    {"text": "[curious] Mhm, bin da.", "voice_id": "g6xIsTj2HwM6VR4iXFCw"}
  ]
}
```

Processing:
- `voice_id` → mapped to local speaker via configurable mapping in Settings
- All `[tags]` stripped before F5-TTS generation
- `[long pause]` / `[short pause]` → converted to silence duration (3s / 1s)
- Text-only tags (`[warmly]`, `[curious]`) → stripped (no F5-TTS equivalent)

---

## Non-Goals (v1)

- Emotion/prosody control (F5-TTS doesn't support it)
- Parallel chunk generation (sequential first, optimize later)
- Real-time streaming (batch generation only)
- Drag-to-reorder dialog lines (nice-to-have, not MVP)
