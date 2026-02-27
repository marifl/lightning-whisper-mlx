from typing import Optional

from .audio import HOP_LENGTH, SAMPLE_RATE


def _seek_to_seconds(seek: int) -> float:
    """Convert a mel-frame seek position to seconds."""
    return seek * HOP_LENGTH / SAMPLE_RATE


def assign_speakers(
    segments: list,
    speaker_turns: list[dict],
) -> list[dict]:
    """Assign speaker labels to transcription segments by temporal overlap.

    Each input segment is [start_seek, end_seek, text] (seek in mel frames).
    Each speaker_turn is {"speaker": str, "start": float, "end": float} (seconds).

    Returns list of dicts with keys: start, end, text, speaker (seconds).
    Speaker is None if no speaker turn overlaps the segment.
    """
    result = []
    for seg in segments:
        start_seek, end_seek, text = seg
        seg_start = _seek_to_seconds(start_seek)
        seg_end = _seek_to_seconds(end_seek)

        best_speaker = None
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        result.append({
            "start": seg_start,
            "end": seg_end,
            "text": text,
            "speaker": best_speaker,
        })

    return result
