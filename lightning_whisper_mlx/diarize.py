import os
from typing import Optional

from .audio import HOP_LENGTH, SAMPLE_RATE


def _seek_to_seconds(seek: int) -> float:
    """Convert a mel-frame seek position to seconds."""
    return seek * HOP_LENGTH / SAMPLE_RATE


def diarize_audio(
    audio_path: str,
    *,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> list[dict]:
    """Run speaker diarization on an audio file using pyannote-audio.

    Requires:
    - pyannote-audio installed: uv sync --extra diarize
    - HF_TOKEN environment variable set (huggingface.co/settings/tokens)

    Returns list of dicts: [{"speaker": "SPEAKER_00", "start": 0.2, "end": 1.5}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote-audio is required for diarization. Install it with:\n"
            "  pip install lightning-whisper-mlx[diarize]\n"
            "  # or: uv sync --extra diarize"
        ) from e

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable required for pyannote diarization models.\n"
            "Get a token at https://huggingface.co/settings/tokens"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    diarization = pipeline(audio_path, num_speakers=num_speakers,
                           min_speakers=min_speakers, max_speakers=max_speakers)

    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })

    return speaker_turns


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
