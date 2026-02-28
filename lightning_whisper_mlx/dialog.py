"""Dialog TTS utilities: tag stripping, text chunking, silence generation."""
from __future__ import annotations

import re

_TAG_RE = re.compile(r"\[.*?\]")
_SPLIT_RE = re.compile(r"(?<=[;:,.!?])\s+")


def strip_tags(text: str) -> str:
    """Remove all [bracket tags] from text and normalize whitespace."""
    cleaned = _TAG_RE.sub("", text)
    return " ".join(cleaned.split())


def chunk_text(text: str, max_bytes: int = 135) -> list[str]:
    """Split text into chunks of at most max_bytes UTF-8 bytes.

    Splits at punctuation boundaries (;:,.!?) to preserve natural phrasing.
    """
    text = text.strip()
    if not text:
        return []

    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    parts = _SPLIT_RE.split(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate.encode("utf-8")) <= max_bytes:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)

    return chunks
