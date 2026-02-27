# Lightning Whisper MLX Web-UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full-stack web UI for lightning-whisper-mlx with a FastAPI backend and Next.js frontend sharing Helperei's brutalist design system (Teal variant).

**Architecture:** Two repos — `lightning-whisper-mlx` gains a `server.py` FastAPI module; `lightning-whisper-mlx-ui` is a new Next.js 16 project with shadcn/ui components copied from Helperei_Forschung and recolored with Teal accent. Frontend communicates with backend via REST API with SSE for progress streaming.

**Tech Stack:** Python (FastAPI, uvicorn), Next.js 16, React 19, Tailwind CSS 4, shadcn/ui (New York), Lexend font, Lucide icons, React Hook Form + Zod.

**Reference repos:**
- Backend: `../lightning-whisper-mlx/`
- Design source: `../Helperei_Forschung/`
- Frontend target: `../lightning-whisper-mlx-ui/` (to be created)

---

## Phase 1: FastAPI Backend (`lightning-whisper-mlx` repo)

### Task 1: Add server dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add the `server` optional dependency group**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
server = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "python-multipart>=0.0.9"]
```

**Step 2: Sync the new dependencies**

Run: `uv sync --extra server`
Expected: Dependencies install successfully.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add FastAPI server dependencies"
```

---

### Task 2: Write failing test for GET /api/models

**Files:**
- Create: `tests/test_server.py`

**Step 1: Write the test**

```python
"""Tests for the FastAPI server module."""
import pytest
from fastapi.testclient import TestClient

from lightning_whisper_mlx.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_list_models(client):
    """GET /api/models returns all available models with their quantization options."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    # Must contain all 11 models from lightning.py
    assert "tiny" in data
    assert "distil-large-v3" in data
    # Standard models have base + 4bit + 8bit
    assert set(data["tiny"]) == {"base", "4bit", "8bit"}
    # Distilled models only have base
    assert set(data["distil-small.en"]) == {"base"}
```

**Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_server.py::test_list_models -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lightning_whisper_mlx.server'`

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add failing test for GET /api/models endpoint"
```

---

### Task 3: Implement GET /api/models endpoint

**Files:**
- Create: `lightning_whisper_mlx/server.py`

**Step 1: Implement the minimal server**

```python
"""FastAPI REST API for lightning-whisper-mlx.

Run with: uvicorn lightning_whisper_mlx.server:app --reload
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from .lightning import models

app = FastAPI(title="Lightning Whisper MLX", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/models")
def list_models() -> dict[str, list[str]]:
    """Return available models and their quantization options."""
    return {name: list(variants.keys()) for name, variants in models.items()}
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_server.py::test_list_models -v`
Expected: PASS

**Step 3: Commit**

```bash
git add lightning_whisper_mlx/server.py
git commit -m "feat: add FastAPI server with GET /api/models endpoint"
```

---

### Task 4: Write failing test for POST /api/transcribe (job creation)

**Files:**
- Modify: `tests/test_server.py`

**Step 1: Write the test**

Append to `tests/test_server.py`:

```python
import io


def test_transcribe_creates_job(client):
    """POST /api/transcribe with an audio file returns a job_id and queued status."""
    # Create a tiny valid WAV file (44 bytes — header only, no samples)
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "tiny"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_transcribe_rejects_invalid_model(client):
    """POST /api/transcribe with invalid model name returns 422."""
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "nonexistent-model"},
    )
    assert resp.status_code == 422
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_server.py -k "transcribe" -v`
Expected: FAIL

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add failing tests for POST /api/transcribe"
```

---

### Task 5: Implement POST /api/transcribe and GET /api/jobs/{job_id}

**Files:**
- Modify: `lightning_whisper_mlx/server.py`

**Step 1: Add job management and transcription endpoint**

Add to `server.py` after the models endpoint:

```python
import tempfile
import threading
from pathlib import Path
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# In-memory job store (non-goal v1: no persistence)
_jobs: dict[str, dict[str, Any]] = {}


def _run_transcription(job_id: str, audio_path: str, model: str,
                        quant: str | None, batch_size: int,
                        diarize: bool, hf_token: str | None,
                        correct: bool, correct_backend: str | None,
                        anthropic_api_key: str | None) -> None:
    """Run transcription in a background thread."""
    import os
    try:
        _jobs[job_id]["status"] = JobStatus.processing

        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        from .lightning import LightningWhisperMLX
        whisper = LightningWhisperMLX(model, batch_size=batch_size, quant=quant)

        result = whisper.transcribe(
            audio_path,
            diarize=diarize,
            correct=correct,
            correct_backend=correct_backend if correct else None,
        )

        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["result"] = result
    except Exception as e:
        _jobs[job_id]["status"] = JobStatus.failed
        _jobs[job_id]["error"] = str(e)
    finally:
        # Clean up temp file
        Path(audio_path).unlink(missing_ok=True)


@app.post("/api/transcribe", status_code=202)
async def transcribe(
    file: UploadFile,
    model: str = Form(default="distil-large-v3"),
    quant: str | None = Form(default=None),
    batch_size: int = Form(default=12),
    diarize: bool = Form(default=False),
    hf_token: str | None = Form(default=None),
    correct: bool = Form(default=False),
    correct_backend: str | None = Form(default=None),
    anthropic_api_key: str | None = Form(default=None),
) -> dict[str, str]:
    """Upload audio and start a transcription job."""
    # Validate model
    if model not in models:
        raise HTTPException(status_code=422, detail=f"Invalid model: {model}. Available: {list(models.keys())}")

    if quant and quant not in ("4bit", "8bit"):
        raise HTTPException(status_code=422, detail="quant must be '4bit', '8bit', or null")

    if quant and "distil" in model:
        raise HTTPException(status_code=422, detail=f"Quantization not supported for distilled model '{model}'")

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": JobStatus.queued,
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_transcription,
        args=(job_id, tmp_path, model, quant, batch_size,
              diarize, hf_token, correct, correct_backend, anthropic_api_key),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """Poll job status and result."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }
```

**Step 2: Run the tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add lightning_whisper_mlx/server.py
git commit -m "feat: add POST /api/transcribe and GET /api/jobs/{job_id} endpoints"
```

---

### Task 6: Write and run test for GET /api/jobs/{job_id}

**Files:**
- Modify: `tests/test_server.py`

**Step 1: Write tests for job polling**

```python
def test_get_job_returns_status(client):
    """GET /api/jobs/{id} returns job status after creation."""
    wav_header = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    create_resp = client.post(
        "/api/transcribe",
        files={"file": ("test.wav", io.BytesIO(wav_header), "audio/wav")},
        data={"model": "tiny"},
    )
    job_id = create_resp.json()["job_id"]

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] in ("queued", "processing", "completed", "failed")


def test_get_job_not_found(client):
    """GET /api/jobs/{id} with invalid ID returns 404."""
    resp = client.get("/api/jobs/nonexistent-id")
    assert resp.status_code == 404
```

**Step 2: Run all server tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add job polling endpoint tests"
```

---

## Phase 2: Next.js Frontend Scaffold (`lightning-whisper-mlx-ui` repo)

### Task 7: Create Next.js project

**Files:**
- Create: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/` (entire project scaffold)

**Step 1: Scaffold the project**

```bash
cd /Users/marcusifland/prj
pnpm create next-app lightning-whisper-mlx-ui \
  --typescript --tailwind --eslint --app --src-dir=false \
  --import-alias "@/*" --use-pnpm
```

**Step 2: Verify it runs**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
pnpm dev
```

Expected: Next.js dev server starts on port 3000.

**Step 3: Init git and commit**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
git init
git add -A
git commit -m "feat: scaffold Next.js 16 project"
```

---

### Task 8: Install shared dependencies

**Files:**
- Modify: `package.json`

**Step 1: Install all required deps matching Helperei's stack**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui

# Radix primitives (only what we need)
pnpm add @radix-ui/react-dialog @radix-ui/react-label @radix-ui/react-popover \
  @radix-ui/react-select @radix-ui/react-slot @radix-ui/react-switch \
  @radix-ui/react-tabs @radix-ui/react-progress @radix-ui/react-checkbox \
  @radix-ui/react-dropdown-menu @radix-ui/react-tooltip

# UI utilities
pnpm add class-variance-authority clsx tailwind-merge lucide-react \
  next-themes @hookform/resolvers react-hook-form zod sonner

# Tailwind CSS 4 setup
pnpm add -D @tailwindcss/postcss tw-animate-css
```

**Step 2: Commit**

```bash
git add package.json pnpm-lock.yaml
git commit -m "feat: install shadcn/ui dependencies and Tailwind CSS 4"
```

---

### Task 9: Set up Tailwind CSS 4, Lexend font, and Teal theme

**Files:**
- Modify: `app/globals.css` — Replace entirely with Helperei's theme (Teal variant)
- Modify: `app/layout.tsx` — Add Lexend font, ThemeProvider
- Modify: `postcss.config.mjs` — Tailwind CSS 4 config
- Create: `components.json` — shadcn/ui config
- Create: `lib/utils.ts` — `cn()` helper

**Step 1: Replace `postcss.config.mjs`**

```js
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    "@tailwindcss/postcss": {},
  },
}
export default config
```

**Step 2: Replace `app/globals.css` entirely**

Copy from `/Users/marcusifland/prj/Helperei_Forschung/app/globals.css` and change:
- Line 41: `--header-red: #e0524c;` → `--header-teal: #0d9488;`
- Line 42: `--salmon: #f4a59a;` → `--salmon: #8fd4cd;`

Everything else is identical.

```css
@import "tailwindcss";

@custom-variant dark (&:is(.dark *));

:root {
  --background: #ffffff;
  --foreground: #111111;
  --card: #ffffff;
  --card-foreground: #111111;
  --popover: #ffffff;
  --popover-foreground: #111111;
  --primary: #111111;
  --primary-foreground: #ffffff;
  --secondary: #f4f4f4;
  --secondary-foreground: #111111;
  --muted: #f1f1f1;
  --muted-foreground: #6b6b6b;
  --accent: #f7f7f7;
  --accent-foreground: #111111;
  --destructive: #c62828;
  --destructive-foreground: #ffffff;
  --border: #111111;
  --input: #111111;
  --ring: #111111;
  --chart-1: oklch(0.4 0.15 240);
  --chart-2: oklch(0.45 0.12 200);
  --chart-3: oklch(0.5 0.1 180);
  --chart-4: oklch(0.55 0.08 160);
  --chart-5: oklch(0.6 0.06 140);
  --radius: 0.25rem;
  --sidebar: #ffffff;
  --sidebar-foreground: #111111;
  --sidebar-primary: #111111;
  --sidebar-primary-foreground: #ffffff;
  --sidebar-accent: #f1f1f1;
  --sidebar-accent-foreground: #111111;
  --sidebar-active: #111111;
  --sidebar-active-foreground: #ffffff;
  --sidebar-border: #111111;
  --sidebar-ring: #111111;
  --header-teal: #0d9488;
  --salmon: #8fd4cd;
  --shadow: #111111;
}

.dark {
  --background: #3a3a3a;
  --foreground: #f5f5f5;
  --card: #4a4a4a;
  --card-foreground: #f5f5f5;
  --popover: #4a4a4a;
  --popover-foreground: #f5f5f5;
  --primary: #111111;
  --primary-foreground: #ffffff;
  --muted: #5a5a5a;
  --muted-foreground: #c0c0c0;
  --secondary: #4a4a4a;
  --secondary-foreground: #f5f5f5;
  --accent: #5a5a5a;
  --accent-foreground: #f5f5f5;
  --destructive: #c62828;
  --destructive-foreground: #ffffff;
  --sidebar: #2a2a2a;
  --sidebar-foreground: #f5f5f5;
  --sidebar-accent: #4a4a4a;
  --sidebar-accent-foreground: #f5f5f5;
  --sidebar-active: #111111;
  --sidebar-active-foreground: #ffffff;
  --border: #111111;
  --input: #111111;
  --ring: #111111;
  --sidebar-border: #111111;
  --sidebar-ring: #111111;
  --shadow: #111111;
}

@theme inline {
  --font-sans: var(--font-lexend);
  --font-mono: "Geist Mono", "Geist Mono Fallback";
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-popover: var(--popover);
  --color-popover-foreground: var(--popover-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary: var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-accent: var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-destructive: var(--destructive);
  --color-destructive-foreground: var(--destructive-foreground);
  --color-border: var(--border);
  --color-input: var(--input);
  --color-ring: var(--ring);
  --color-chart-1: var(--chart-1);
  --color-chart-2: var(--chart-2);
  --color-chart-3: var(--chart-3);
  --color-chart-4: var(--chart-4);
  --color-chart-5: var(--chart-5);
  --radius-sm: calc(var(--radius) - 4px);
  --radius-md: calc(var(--radius) - 2px);
  --radius-lg: var(--radius);
  --radius-xl: calc(var(--radius) + 4px);
  --color-sidebar: var(--sidebar);
  --color-sidebar-foreground: var(--sidebar-foreground);
  --color-sidebar-primary: var(--sidebar-primary);
  --color-sidebar-primary-foreground: var(--sidebar-primary-foreground);
  --color-sidebar-accent: var(--sidebar-accent);
  --color-sidebar-accent-foreground: var(--sidebar-accent-foreground);
  --color-sidebar-active: var(--sidebar-active);
  --color-sidebar-active-foreground: var(--sidebar-active-foreground);
  --color-sidebar-border: var(--sidebar-border);
  --color-sidebar-ring: var(--sidebar-ring);
}

@layer base {
  * {
    @apply border-border outline-ring/50;
  }
  body {
    @apply bg-background text-foreground antialiased;
  }
}

.brutalist-shadow {
  box-shadow: 4px 4px 0 var(--shadow);
}

.dark .brutalist-shadow {
  box-shadow: 4px 4px 0 #111111;
}
```

**Step 3: Create `lib/utils.ts`**

```typescript
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

**Step 4: Create `components.json`**

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "",
    "css": "app/globals.css",
    "baseColor": "neutral",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib",
    "hooks": "@/hooks"
  },
  "iconLibrary": "lucide"
}
```

**Step 5: Create `components/theme-provider.tsx`**

Copy verbatim from `/Users/marcusifland/prj/Helperei_Forschung/components/theme-provider.tsx`:

```tsx
"use client"

import * as React from "react"
import {
  ThemeProvider as NextThemesProvider,
  type ThemeProviderProps,
} from "next-themes"

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>
}
```

**Step 6: Replace `app/layout.tsx`**

```tsx
import type React from "react"
import type { Metadata } from "next"
import { Lexend } from "next/font/google"
import "./globals.css"
import { ThemeProvider } from "@/components/theme-provider"

const lexend = Lexend({ subsets: ["latin"], variable: "--font-lexend" })

export const metadata: Metadata = {
  title: "Lightning Whisper MLX",
  description: "High-performance speech-to-text for Apple Silicon",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${lexend.variable} font-sans antialiased`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
```

**Step 7: Verify the app builds**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
pnpm build
```

Expected: Build succeeds.

**Step 8: Commit**

```bash
git add -A
git commit -m "feat: set up Tailwind CSS 4 with brutalist Teal theme"
```

---

### Task 10: Copy shared UI components from Helperei

**Files:**
- Create: `components/ui/button.tsx`
- Create: `components/ui/card.tsx`
- Create: `components/ui/badge.tsx`
- Create: `components/ui/input.tsx`
- Create: `components/ui/label.tsx`
- Create: `components/ui/select.tsx`
- Create: `components/ui/tabs.tsx`
- Create: `components/ui/dialog.tsx`
- Create: `components/ui/checkbox.tsx`
- Create: `components/ui/sheet.tsx`
- Create: `components/ui/tooltip.tsx`
- Create: `components/ui/dropdown-menu.tsx`
- Create: `components/ui/selection-card.tsx`
- Create: `components/ui/section-header.tsx`
- Create: `components/ui/section-wrapper.tsx`
- Create: `components/ui/copy-button.tsx`
- Create: `components/ui/theme-toggle.tsx`

**Step 1: Copy each file verbatim from Helperei**

Source: `/Users/marcusifland/prj/Helperei_Forschung/components/ui/`
Target: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/components/ui/`

Copy these files exactly as-is (no modifications needed — they use CSS variables so the Teal theme applies automatically):
- `button.tsx`, `card.tsx`, `badge.tsx`, `input.tsx`, `label.tsx`
- `select.tsx`, `tabs.tsx`, `dialog.tsx`, `checkbox.tsx`, `sheet.tsx`
- `tooltip.tsx`, `dropdown-menu.tsx`
- `selection-card.tsx`, `section-header.tsx`, `section-wrapper.tsx`
- `copy-button.tsx`, `theme-toggle.tsx`

**Step 2: Verify build still passes**

```bash
pnpm build
```

**Step 3: Commit**

```bash
git add components/ui/
git commit -m "feat: copy shared UI components from Helperei design system"
```

---

### Task 11: Create layout components (Sidebar + PageShell)

**Files:**
- Create: `components/layout/sidebar.tsx`
- Create: `components/layout/page-shell.tsx`

**Step 1: Create the Sidebar (adapted from Helperei)**

Create `components/layout/sidebar.tsx`:

```tsx
"use client"

import { useState, useEffect } from "react"
import { Menu, Mic, Settings, FileText, Sun, Moon } from "lucide-react"
import { useTheme } from "next-themes"
import { cn } from "@/lib/utils"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"

export type NavSection = "upload" | "settings" | "results"

const navItems: { id: NavSection; label: string; icon: React.ReactNode }[] = [
  { id: "upload", label: "Upload", icon: <Mic className="h-4 w-4" /> },
  { id: "settings", label: "Settings", icon: <Settings className="h-4 w-4" /> },
  { id: "results", label: "Results", icon: <FileText className="h-4 w-4" /> },
]

interface NavContentProps {
  activeSection: NavSection
  onNavigate: (section: NavSection) => void
  onClose?: () => void
  isMobile?: boolean
}

function NavContent({ activeSection, onNavigate, onClose, isMobile = false }: NavContentProps) {
  const { theme, setTheme } = useTheme()
  const linkClass = isMobile ? "py-2.5" : "py-2"

  return (
    <nav className={cn("space-y-6", isMobile ? "" : "p-4")}>
      <div>
        <h2 className={cn(
          "font-bold text-[11px] text-muted-foreground uppercase tracking-widest",
          isMobile ? "mb-3" : "mb-2"
        )}>
          Workflow
        </h2>
        <div className="space-y-1">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => {
                onNavigate(item.id)
                onClose?.()
              }}
              className={cn(
                "flex items-center gap-2 w-full text-left px-3 rounded-sm transition-colors text-sm font-semibold border-2 border-transparent",
                linkClass,
                activeSection === item.id
                  ? "bg-sidebar-active text-sidebar-active-foreground border-sidebar-active"
                  : "hover:border-border"
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <hr className="border-border border-t-2" />

      <button
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="flex items-center gap-2 w-full text-left px-3 py-2 rounded-sm transition-colors text-sm font-semibold border-2 border-transparent hover:border-border"
      >
        {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        {theme === "dark" ? "Light Mode" : "Dark Mode"}
      </button>
    </nav>
  )
}

interface AppSidebarProps {
  activeSection: NavSection
  onNavigate: (section: NavSection) => void
  collapsed: boolean
}

export function AppSidebar({ activeSection, onNavigate, collapsed }: AppSidebarProps) {
  return (
    <aside className={cn(
      "fixed left-6 top-6 bottom-6 bg-sidebar border-2 border-border brutalist-shadow transition-all duration-300 z-40 overflow-y-auto rounded-sm hidden md:block",
      collapsed ? "w-0 opacity-0 overflow-hidden" : "w-64"
    )}>
      <div className="p-4 border-b-2 border-border">
        <h1 className="font-bold text-lg">Lightning Whisper</h1>
        <p className="text-xs text-muted-foreground">Speech-to-Text for Apple Silicon</p>
      </div>
      <NavContent activeSection={activeSection} onNavigate={onNavigate} />
    </aside>
  )
}

interface MobileNavProps {
  activeSection: NavSection
  onNavigate: (section: NavSection) => void
}

export function MobileNav({ activeSection, onNavigate }: MobileNavProps) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 768px)")
    const handleChange = (e: MediaQueryListEvent) => {
      if (e.matches) setOpen(false)
    }
    mediaQuery.addEventListener("change", handleChange)
    return () => mediaQuery.removeEventListener("change", handleChange)
  }, [])

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button
          className="p-2 hover:bg-black/20 rounded-sm transition-colors"
          aria-label="Open navigation"
        >
          <Menu className="h-6 w-6" />
        </button>
      </SheetTrigger>
      <SheetContent side="left" className="w-[280px] sm:w-[320px] overflow-y-auto">
        <SheetHeader className="mb-6">
          <SheetTitle className="text-left font-bold text-xl">Navigation</SheetTitle>
        </SheetHeader>
        <NavContent
          activeSection={activeSection}
          onNavigate={onNavigate}
          onClose={() => setOpen(false)}
          isMobile
        />
      </SheetContent>
    </Sheet>
  )
}
```

**Step 2: Create `components/layout/page-shell.tsx`**

Copy from Helperei verbatim:

```tsx
import { cn } from "@/lib/utils"

interface PageShellProps {
  children: React.ReactNode
  collapsed: boolean
  className?: string
}

export function PageShell({ children, collapsed, className }: PageShellProps) {
  return (
    <main className={cn(
      "transition-all duration-300 min-h-screen",
      "mx-4 sm:mx-6",
      collapsed ? "md:ml-6" : "md:ml-[300px]",
      "md:mr-6",
      "pb-12 sm:pb-16",
      className
    )}>
      {children}
    </main>
  )
}
```

**Step 3: Verify build**

```bash
pnpm build
```

**Step 4: Commit**

```bash
git add components/layout/
git commit -m "feat: add sidebar and page shell layout components"
```

---

## Phase 3: Feature Sections

### Task 12: Create API client

**Files:**
- Create: `lib/api.ts`

**Step 1: Implement the API client**

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface ModelsResponse {
  [model: string]: string[]
}

export interface TranscribeParams {
  file: File
  model: string
  quant: string | null
  batchSize: number
  diarize: boolean
  hfToken: string | null
  correct: boolean
  correctBackend: string | null
}

export interface JobResponse {
  job_id: string
  status: "queued" | "processing" | "completed" | "failed"
  result: {
    text: string
    segments: Array<Record<string, unknown>>
    language: string
  } | null
  error: string | null
}

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE}/api/models`)
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.status}`)
  return res.json()
}

export async function startTranscription(params: TranscribeParams): Promise<{ job_id: string; status: string }> {
  const form = new FormData()
  form.append("file", params.file)
  form.append("model", params.model)
  if (params.quant) form.append("quant", params.quant)
  form.append("batch_size", String(params.batchSize))
  form.append("diarize", String(params.diarize))
  if (params.hfToken) form.append("hf_token", params.hfToken)
  form.append("correct", String(params.correct))
  if (params.correctBackend) form.append("correct_backend", params.correctBackend)

  const res = await fetch(`${API_BASE}/api/transcribe`, { method: "POST", body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Failed to start transcription: ${res.status}`)
  }
  return res.json()
}

export async function pollJob(jobId: string): Promise<JobResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`)
  if (!res.ok) throw new Error(`Failed to poll job: ${res.status}`)
  return res.json()
}
```

**Step 2: Commit**

```bash
git add lib/api.ts
git commit -m "feat: add API client for backend communication"
```

---

### Task 13: Build Section 1 — Audio Upload

**Files:**
- Create: `components/sections/audio-upload.tsx`

**Step 1: Implement the upload component**

```tsx
"use client"

import { useCallback, useState } from "react"
import { Upload, X, FileAudio } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface AudioUploadProps {
  file: File | null
  onFileChange: (file: File | null) => void
}

export function AudioUpload({ file, onFileChange }: AudioUploadProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped?.type.startsWith("audio/")) {
      onFileChange(dropped)
    }
  }, [onFileChange])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) onFileChange(selected)
  }, [onFileChange])

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  if (file) {
    return (
      <div className="flex items-center gap-4 p-4 border-2 border-border rounded-sm bg-muted">
        <FileAudio className="h-8 w-8 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-sm truncate">{file.name}</p>
          <p className="text-xs text-muted-foreground">{formatSize(file.size)}</p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={() => onFileChange(null)}>
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "relative flex flex-col items-center justify-center gap-3 p-8",
        "border-2 border-dashed border-border rounded-sm",
        "transition-all cursor-pointer",
        isDragging ? "bg-muted border-foreground" : "bg-background hover:bg-muted"
      )}
    >
      <Upload className="h-8 w-8 text-muted-foreground" />
      <div className="text-center">
        <p className="font-semibold text-sm">Drop audio file here</p>
        <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
      </div>
      <input
        type="file"
        accept="audio/*"
        onChange={handleFileInput}
        className="absolute inset-0 opacity-0 cursor-pointer"
      />
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add components/sections/audio-upload.tsx
git commit -m "feat: add audio upload component with drag & drop"
```

---

### Task 14: Build Section 2 — Model Settings

**Files:**
- Create: `components/sections/model-settings.tsx`

**Step 1: Implement the model settings component**

```tsx
"use client"

import { SelectionCard, SelectionCardGrid } from "@/components/ui/selection-card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

const MODEL_GROUPS = [
  {
    title: "Distilled (Fast)",
    models: ["distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3"],
  },
  {
    title: "Standard",
    models: ["tiny", "base", "small", "medium"],
  },
  {
    title: "Large",
    models: ["large", "large-v2", "large-v3"],
  },
]

const QUANT_OPTIONS = [
  { value: null, label: "None (fp16)" },
  { value: "4bit", label: "4-bit" },
  { value: "8bit", label: "8-bit" },
] as const

interface ModelSettingsProps {
  model: string
  onModelChange: (model: string) => void
  quant: string | null
  onQuantChange: (quant: string | null) => void
  diarize: boolean
  onDiarizeChange: (diarize: boolean) => void
  hfToken: string
  onHfTokenChange: (token: string) => void
  correct: boolean
  onCorrectChange: (correct: boolean) => void
  correctBackend: string
  onCorrectBackendChange: (backend: string) => void
  availableModels: Record<string, string[]> | null
}

export function ModelSettings({
  model, onModelChange,
  quant, onQuantChange,
  diarize, onDiarizeChange,
  hfToken, onHfTokenChange,
  correct, onCorrectChange,
  correctBackend, onCorrectBackendChange,
  availableModels,
}: ModelSettingsProps) {
  const isDistilled = model.startsWith("distil")
  const availableQuants = availableModels?.[model] ?? []

  return (
    <div className="space-y-6">
      {/* Model Selection */}
      {MODEL_GROUPS.map(group => (
        <div key={group.title}>
          <Label className="text-xs uppercase tracking-widest text-muted-foreground font-bold mb-2 block">
            {group.title}
          </Label>
          <SelectionCardGrid columns={2}>
            {group.models.map(m => (
              <SelectionCard
                key={m}
                selected={model === m}
                onClick={() => onModelChange(m)}
                title={m}
                disabled={availableModels !== null && !(m in availableModels)}
              />
            ))}
          </SelectionCardGrid>
        </div>
      ))}

      {/* Quantization */}
      <div>
        <Label className="text-xs uppercase tracking-widest text-muted-foreground font-bold mb-2 block">
          Quantization
        </Label>
        <SelectionCardGrid columns={3}>
          {QUANT_OPTIONS.map(opt => (
            <SelectionCard
              key={opt.label}
              selected={quant === opt.value}
              onClick={() => onQuantChange(opt.value)}
              title={opt.label}
              disabled={isDistilled && opt.value !== null}
              info={isDistilled && opt.value !== null ? "Not supported for distilled models" : undefined}
            />
          ))}
        </SelectionCardGrid>
      </div>

      {/* Feature Toggles */}
      <div className="space-y-3">
        <Label className="text-xs uppercase tracking-widest text-muted-foreground font-bold block">
          Post-Processing
        </Label>

        {/* Diarization toggle */}
        <SelectionCard
          selected={diarize}
          onClick={() => onDiarizeChange(!diarize)}
          title="Speaker Diarization"
          description="Identify who speaks when"
        />
        {diarize && (
          <div className="pl-4">
            <Label className="text-sm font-semibold mb-1 block">HF_TOKEN</Label>
            <Input
              type="password"
              placeholder="hf_..."
              value={hfToken}
              onChange={(e) => onHfTokenChange(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Required for pyannote-audio speaker diarization model.
            </p>
          </div>
        )}

        {/* Correction toggle */}
        <SelectionCard
          selected={correct}
          onClick={() => onCorrectChange(!correct)}
          title="LLM Text Correction"
          description="Clean up transcription with an LLM"
        />
        {correct && (
          <div className="pl-4">
            <Label className="text-xs uppercase tracking-widest text-muted-foreground font-bold mb-2 block">
              Backend
            </Label>
            <SelectionCardGrid columns={2}>
              <SelectionCard
                selected={correctBackend === "local"}
                onClick={() => onCorrectBackendChange("local")}
                title="Local (mlx-lm)"
                description="Runs on device"
              />
              <SelectionCard
                selected={correctBackend === "anthropic"}
                onClick={() => onCorrectBackendChange("anthropic")}
                title="Anthropic API"
                description="Cloud-based"
              />
            </SelectionCardGrid>
          </div>
        )}
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add components/sections/model-settings.tsx
git commit -m "feat: add model settings component with selection cards"
```

---

### Task 15: Build Section 3 — Transcription Control

**Files:**
- Create: `components/sections/transcription-control.tsx`

**Step 1: Implement the transcription control**

```tsx
"use client"

import { Loader2, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface TranscriptionControlProps {
  status: "idle" | "queued" | "processing" | "completed" | "failed"
  onStart: () => void
  disabled: boolean
  error: string | null
}

const statusVariant: Record<string, "ghost" | "warning" | "success" | "error"> = {
  idle: "ghost",
  queued: "warning",
  processing: "warning",
  completed: "success",
  failed: "error",
}

export function TranscriptionControl({ status, onStart, disabled, error }: TranscriptionControlProps) {
  const isRunning = status === "queued" || status === "processing"

  return (
    <div className="space-y-4">
      <Button
        onClick={onStart}
        disabled={disabled || isRunning}
        className="w-full h-12 text-base"
      >
        {isRunning ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            {status === "queued" ? "Queued..." : "Transcribing..."}
          </>
        ) : (
          <>
            <Play className="h-5 w-5" />
            Start Transcription
          </>
        )}
      </Button>

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
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add components/sections/transcription-control.tsx
git commit -m "feat: add transcription control component"
```

---

### Task 16: Build Section 4 — Results Display

**Files:**
- Create: `components/sections/results-display.tsx`

**Step 1: Implement the results component**

```tsx
"use client"

import { useState } from "react"
import { Copy, Download, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface Segment {
  start?: number
  end?: number
  text: string
  speaker?: string
  raw_text?: string
  [key: string]: unknown
}

interface ResultsDisplayProps {
  result: {
    text: string
    segments: Segment[] | [number, number, string][]
    language: string
  }
}

type TabId = "full-text" | "segments" | "by-speaker"

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const [activeTab, setActiveTab] = useState<TabId>("full-text")
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(result.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownloadTxt = () => {
    const blob = new Blob([result.text], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "transcription.txt"
    a.click()
    URL.revokeObjectURL(url)
  }

  // Normalize segments to dict format
  const segments: Segment[] = result.segments.map((seg) => {
    if (Array.isArray(seg)) {
      return { start: seg[0] / 100, end: seg[1] / 100, text: seg[2] as string }
    }
    return seg as Segment
  })

  // Group by speaker
  const speakers = new Map<string, Segment[]>()
  for (const seg of segments) {
    const speaker = seg.speaker ?? "Unknown"
    if (!speakers.has(speaker)) speakers.set(speaker, [])
    speakers.get(speaker)!.push(seg)
  }

  const tabs: { id: TabId; label: string }[] = [
    { id: "full-text", label: "Full Text" },
    { id: "segments", label: "Segments" },
    ...(speakers.size > 1 ? [{ id: "by-speaker" as TabId, label: "By Speaker" }] : []),
  ]

  return (
    <div className="space-y-4">
      {/* Language badge */}
      <Badge variant="secondary">{result.language.toUpperCase()}</Badge>

      {/* Tab bar */}
      <div className="flex gap-1 border-2 border-border rounded-sm p-1 bg-muted">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex-1 px-3 py-1.5 rounded-sm text-sm font-semibold transition-all",
              activeTab === tab.id
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="border-2 border-border rounded-sm p-4 bg-card min-h-[200px]">
        {activeTab === "full-text" && (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{result.text}</p>
        )}

        {activeTab === "segments" && (
          <div className="space-y-2">
            {segments.map((seg, i) => (
              <div key={i} className="flex gap-3 text-sm">
                {seg.start !== undefined && (
                  <span className="font-mono text-xs text-muted-foreground w-20 shrink-0 pt-0.5">
                    {formatTime(seg.start)} — {formatTime(seg.end ?? seg.start)}
                  </span>
                )}
                {seg.speaker && (
                  <Badge variant="outline" className="shrink-0">{seg.speaker}</Badge>
                )}
                <span className="flex-1">
                  {seg.text}
                  {seg.raw_text && seg.raw_text !== seg.text && (
                    <span className="block text-xs text-muted-foreground line-through mt-0.5">
                      {seg.raw_text}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        )}

        {activeTab === "by-speaker" && (
          <div className="space-y-6">
            {[...speakers.entries()].map(([speaker, segs]) => (
              <div key={speaker}>
                <Badge variant="default" className="mb-2">{speaker}</Badge>
                <div className="space-y-1 pl-2 border-l-2 border-border">
                  {segs.map((seg, i) => (
                    <p key={i} className="text-sm">
                      {seg.start !== undefined && (
                        <span className="font-mono text-xs text-muted-foreground mr-2">
                          {formatTime(seg.start)}
                        </span>
                      )}
                      {seg.text}
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={handleCopy}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button variant="outline" size="sm" onClick={handleDownloadTxt}>
          <Download className="h-4 w-4" />
          Download .txt
        </Button>
      </div>
    </div>
  )
}
```

**Step 2: Commit**

```bash
git add components/sections/results-display.tsx
git commit -m "feat: add results display with tabs, segments, speaker grouping"
```

---

### Task 17: Assemble main page

**Files:**
- Replace: `app/page.tsx`

**Step 1: Build the main page that ties everything together**

```tsx
"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { AppSidebar, MobileNav, type NavSection } from "@/components/layout/sidebar"
import { PageShell } from "@/components/layout/page-shell"
import { SectionWrapper } from "@/components/ui/section-wrapper"
import { AudioUpload } from "@/components/sections/audio-upload"
import { ModelSettings } from "@/components/sections/model-settings"
import { TranscriptionControl } from "@/components/sections/transcription-control"
import { ResultsDisplay } from "@/components/sections/results-display"
import { fetchModels, startTranscription, pollJob, type JobResponse } from "@/lib/api"

export default function Home() {
  // Navigation
  const [activeSection, setActiveSection] = useState<NavSection>("upload")

  // Upload state
  const [file, setFile] = useState<File | null>(null)

  // Model settings
  const [model, setModel] = useState("distil-large-v3")
  const [quant, setQuant] = useState<string | null>(null)
  const [diarize, setDiarize] = useState(false)
  const [hfToken, setHfToken] = useState("")
  const [correct, setCorrect] = useState(false)
  const [correctBackend, setCorrectBackend] = useState("anthropic")
  const [availableModels, setAvailableModels] = useState<Record<string, string[]> | null>(null)

  // Job state
  const [jobStatus, setJobStatus] = useState<"idle" | "queued" | "processing" | "completed" | "failed">("idle")
  const [result, setResult] = useState<JobResponse["result"]>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Fetch available models on mount
  useEffect(() => {
    fetchModels().then(setAvailableModels).catch(() => {})
  }, [])

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  const handleStart = useCallback(async () => {
    if (!file) return

    setJobStatus("queued")
    setError(null)
    setResult(null)

    try {
      const { job_id } = await startTranscription({
        file,
        model,
        quant,
        batchSize: 12,
        diarize,
        hfToken: diarize ? hfToken : null,
        correct,
        correctBackend: correct ? correctBackend : null,
      })

      // Poll for result
      pollRef.current = setInterval(async () => {
        try {
          const job = await pollJob(job_id)
          setJobStatus(job.status)

          if (job.status === "completed") {
            setResult(job.result)
            if (pollRef.current) clearInterval(pollRef.current)
          } else if (job.status === "failed") {
            setError(job.error ?? "Transcription failed")
            if (pollRef.current) clearInterval(pollRef.current)
          }
        } catch {
          setError("Lost connection to server")
          if (pollRef.current) clearInterval(pollRef.current)
        }
      }, 1000)
    } catch (e) {
      setJobStatus("failed")
      setError(e instanceof Error ? e.message : "Failed to start transcription")
    }
  }, [file, model, quant, diarize, hfToken, correct, correctBackend])

  // Auto-reset quant when switching to distilled model
  useEffect(() => {
    if (model.startsWith("distil") && quant !== null) {
      setQuant(null)
    }
  }, [model, quant])

  const sidebarCollapsed = false

  return (
    <>
      {/* Mobile header */}
      <header className="md:hidden flex items-center gap-3 p-4 border-b-2 border-border">
        <MobileNav activeSection={activeSection} onNavigate={setActiveSection} />
        <h1 className="font-bold text-lg">Lightning Whisper</h1>
      </header>

      {/* Desktop sidebar */}
      <AppSidebar
        activeSection={activeSection}
        onNavigate={setActiveSection}
        collapsed={sidebarCollapsed}
      />

      {/* Main content */}
      <PageShell collapsed={sidebarCollapsed}>
        <div className="max-w-3xl pt-6 md:pt-12 space-y-12">

          <SectionWrapper
            number={1}
            title="Audio Upload"
            description="Upload an audio file for transcription"
            isCompleted={file !== null}
            showConnector
          >
            <AudioUpload file={file} onFileChange={setFile} />
          </SectionWrapper>

          <SectionWrapper
            number={2}
            title="Model & Settings"
            description="Choose your Whisper model and post-processing options"
            showConnector
          >
            <ModelSettings
              model={model}
              onModelChange={setModel}
              quant={quant}
              onQuantChange={setQuant}
              diarize={diarize}
              onDiarizeChange={setDiarize}
              hfToken={hfToken}
              onHfTokenChange={setHfToken}
              correct={correct}
              onCorrectChange={setCorrect}
              correctBackend={correctBackend}
              onCorrectBackendChange={setCorrectBackend}
              availableModels={availableModels}
            />
          </SectionWrapper>

          <SectionWrapper
            number={3}
            title="Transcribe"
            description="Start the transcription process"
            isCompleted={jobStatus === "completed"}
            showConnector={jobStatus === "completed"}
          >
            <TranscriptionControl
              status={jobStatus}
              onStart={handleStart}
              disabled={!file}
              error={error}
            />
          </SectionWrapper>

          {result && (
            <SectionWrapper
              number={4}
              title="Results"
              description="Transcription output"
              isCompleted
            >
              <ResultsDisplay result={result} />
            </SectionWrapper>
          )}

        </div>
      </PageShell>
    </>
  )
}
```

**Step 2: Verify the full app builds**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
pnpm build
```

**Step 3: Commit**

```bash
git add app/page.tsx
git commit -m "feat: assemble main page with all workflow sections"
```

---

## Phase 4: Integration & Polish

### Task 18: Add .env.local template

**Files:**
- Create: `.env.example`

**Step 1: Create the file**

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Step 2: Add to .gitignore**

Ensure `.env.local` is in `.gitignore` (Next.js scaffold usually includes it).

**Step 3: Commit**

```bash
git add .env.example
git commit -m "feat: add .env.example with API URL config"
```

---

### Task 19: End-to-end smoke test

**Files:** None created; this is a verification task.

**Step 1: Start the FastAPI backend**

```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx
uv run uvicorn lightning_whisper_mlx.server:app --reload --port 8000
```

**Step 2: Start the Next.js frontend**

In a second terminal:
```bash
cd /Users/marcusifland/prj/lightning-whisper-mlx-ui
pnpm dev
```

**Step 3: Manual verification checklist**

- [ ] Open http://localhost:3000
- [ ] Verify brutalist design renders correctly (black borders, Lexend font)
- [ ] Verify dark mode toggle works
- [ ] Verify sidebar navigation
- [ ] Verify model list loads from backend (Section 2 shows models)
- [ ] Upload an audio file → verify file info displays
- [ ] Click "Start Transcription" → verify job queues
- [ ] Wait for completion → verify results display
- [ ] Test "Copy" and "Download .txt" buttons

---

### Task 20: Create README for the UI repo

**Files:**
- Create: `/Users/marcusifland/prj/lightning-whisper-mlx-ui/README.md`

**Step 1: Write the README**

```markdown
# Lightning Whisper MLX — Web UI

Web interface for [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx), built with Next.js 16 and the Helperei brutalist design system (Teal variant).

## Prerequisites

- Node.js 20+
- pnpm
- Running `lightning-whisper-mlx` FastAPI backend

## Setup

```bash
pnpm install
cp .env.example .env.local  # Edit API_URL if needed
pnpm dev
```

## Backend

Start the FastAPI server in the `lightning-whisper-mlx` repo:

```bash
uv sync --extra server
uv run uvicorn lightning_whisper_mlx.server:app --reload
```

## Design System

Shares the brutalist design system with [attachment.tools](https://attachment.tools):
- **Font:** Lexend
- **Borders:** 2px solid #111111
- **Shadows:** 4px 4px 0 #111111 (brutalist)
- **Accent:** Teal `#0d9488` (Helperei uses Salmon-Red `#e0524c`)
- **Components:** shadcn/ui (New York style)
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| **1: Backend** | Tasks 1-6 | FastAPI server with `/api/models`, `/api/transcribe`, `/api/jobs/{id}` |
| **2: Frontend Scaffold** | Tasks 7-11 | Next.js project + Tailwind Teal theme + shared UI components + layout |
| **3: Feature Sections** | Tasks 12-16 | API client, Upload, Model Settings, Transcription Control, Results |
| **4: Integration** | Tasks 17-20 | Main page assembly, env config, E2E smoke test, README |

Total: 20 tasks. Backend and frontend phases can run in parallel after Task 1.
