# Lightning Whisper MLX — Web-UI Design

**Date:** 2026-02-27
**Status:** Approved

## Overview

Web-based UI for lightning-whisper-mlx exposing all STT features: transcription, speaker diarization, LLM text correction, and model selection. The design system is a 1:1 copy of [attachment.tools](https://attachment.tools) / Helperei_Forschung, differentiated only by the accent color (Teal instead of Salmon-Red).

## Architecture

```
lightning-whisper-mlx-ui/        (New repo — Next.js frontend)
  ├─ app/                        Next.js App Router
  ├─ components/ui/              shadcn/ui components (brutalist style)
  ├─ components/                 App-specific components
  ├─ lib/                        API client, utils
  └─ package.json

lightning-whisper-mlx/           (Existing repo — add FastAPI server)
  ├─ lightning_whisper_mlx/
  │   ├─ server.py               NEW: FastAPI REST API
  │   └─ ...existing modules...
  └─ pyproject.toml              Add fastapi + uvicorn deps
```

### Communication

Frontend ←HTTP→ FastAPI backend. The backend wraps `LightningWhisperMLX` and exposes REST endpoints.

## Design System

### Shared With Helperei (Identical)

Everything from `globals.css` is copied 1:1 except the two accent tokens. This includes:

| Token | Value | Notes |
|-------|-------|-------|
| `--background` | `#ffffff` / `#3a3a3a` | Light / Dark |
| `--foreground` | `#111111` / `#f5f5f5` | Light / Dark |
| `--border` | `#111111` | Same in both modes |
| `--shadow` | `#111111` | Brutalist shadow |
| `--primary` | `#111111` | Buttons, borders |
| `--radius` | `0.25rem` | 4px, `rounded-sm` |
| Font | Lexend | `--font-lexend` |
| Shadows | `4px 4px 0 var(--shadow)` | `.brutalist-shadow` |
| Borders | `border-2 border-border` | On all interactive elements |

Full token list: see Helperei's `app/globals.css`.

### Differentiated (Whisper UI)

| Token | Helperei | Whisper UI |
|-------|----------|------------|
| `--header-red` / `--header-teal` | `#e0524c` | `#0d9488` |
| `--salmon` | `#f4a59a` | `#8fd4cd` |

### Shared Components (Copy from Helperei)

All shadcn/ui New York style components:
- Button (default, outline, ghost, destructive, link, icon variants)
- Card (with `border-2`, `rounded-sm`, no shadow)
- Input, Textarea, Select
- Badge (default, secondary, destructive, outline, success, warning, error, ghost)
- Dialog, Popover, Tabs
- Sidebar (fixed left, brutalist shadow, collapsible)
- SelectionCard (for model selection)
- SectionHeader + SectionWrapper (numbered steps with connector lines)

### Tech Stack

- **Framework:** Next.js 16 (App Router)
- **UI:** React 19 + shadcn/ui (New York)
- **Styling:** Tailwind CSS 4 + PostCSS
- **Theme:** next-themes (light/dark)
- **Font:** Lexend (Google Fonts)
- **Icons:** Lucide React
- **Forms:** React Hook Form + Zod

## Pages & Layout

### Main Layout

```
┌──────────────────────────────────────────────────────┐
│  Sidebar (brutalist)             │  Main Content      │
│  w-64, border-2, brutalist-shadow│                    │
│                                  │  Section-based     │
│  Nav Items:                      │  workflow with      │
│  • Upload                        │  numbered steps    │
│  • Settings                      │  and connector     │
│  • Results                       │  lines             │
│  • History (future)              │                    │
│                                  │                    │
│  Theme Toggle                    │                    │
└──────────────────────────────────────────────────────┘
```

### Workflow Sections

**Section 1 — Audio Upload**
- Drag & drop zone with `border-2 border-dashed`
- File type validation (audio/*)
- File size display
- Upload progress indicator

**Section 2 — Model Settings**
- Model selection: SelectionCard grid (`tiny`, `small`, `base`, `medium`, `large`, `large-v2`, `large-v3` + distilled variants)
- Quantization toggle: `None` / `4bit` / `8bit` (disabled for distilled models)
- Feature toggles:
  - `[✓] Speaker Diarization` — shows HF_TOKEN input if needed
  - `[✓] LLM Text Correction` — shows backend selector (local/anthropic) + config

**Section 3 — Transcription**
- Start button (primary, full width)
- Progress bar during transcription
- Status badges (queued, processing, done, error)

**Section 4 — Results**
- Tabbed view: `Full Text` | `Segments` | `By Speaker`
- Full text: plain text output with copy button
- Segments: table with start/end times, text, speaker (if diarized)
- By Speaker: grouped by speaker with color-coded badges
- Corrected text shows diff (raw_text vs corrected)
- Export options: Copy, Download .txt, Download .srt

## FastAPI Backend

### Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `GET /api/models` | GET | List available models + supported quantizations |
| `POST /api/transcribe` | POST | Upload audio file, start transcription job |
| `GET /api/jobs/{job_id}` | GET | Poll job status + result |
| `GET /api/jobs/{job_id}/stream` | GET (SSE) | Stream progress updates |

### POST /api/transcribe

Request (multipart/form-data):
- `file`: audio file
- `model`: model name (default: `distil-large-v3`)
- `quant`: `null` | `4bit` | `8bit`
- `batch_size`: int (default: 12)
- `diarize`: bool (default: false)
- `hf_token`: string (for diarization)
- `correct`: bool (default: false)
- `correct_backend`: `local` | `anthropic` | null
- `anthropic_api_key`: string (for API correction)

Response:
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

### GET /api/jobs/{job_id}

Response:
```json
{
  "job_id": "uuid",
  "status": "processing" | "completed" | "failed",
  "progress": 0.65,
  "result": {
    "text": "...",
    "segments": [...],
    "language": "en"
  },
  "error": null
}
```

### Dependencies

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
server = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "python-multipart>=0.0.9"]
```

## Non-Goals (v1)

- User accounts / authentication
- Persistent job history (in-memory only)
- Real-time audio recording
- Multiple concurrent transcription jobs
- Deployment/containerization (Docker comes later)
