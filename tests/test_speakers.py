"""Tests for speaker CRUD utilities."""
import json
from pathlib import Path

import pytest

from lightning_whisper_mlx.speakers import (
    create_speaker,
    list_speakers,
    get_speaker,
    delete_speaker,
    SPEAKERS_DIR,
)


@pytest.fixture(autouse=True)
def tmp_speakers(tmp_path, monkeypatch):
    """Point SPEAKERS_DIR to a temp directory for every test."""
    monkeypatch.setattr("lightning_whisper_mlx.speakers.SPEAKERS_DIR", tmp_path)
    return tmp_path


def _make_wav_bytes() -> bytes:
    """Minimal valid WAV header (16kHz mono, 0.1s silence)."""
    import wave, io
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


class TestCreateSpeaker:
    def test_creates_directory_and_metadata(self, tmp_speakers):
        speaker = create_speaker("Alice", _make_wav_bytes(), "Hello, I am Alice.")
        assert speaker["name"] == "Alice"
        assert speaker["ref_text"] == "Hello, I am Alice."
        sid = speaker["id"]
        assert (tmp_speakers / sid / "metadata.json").exists()
        assert (tmp_speakers / sid / "ref_audio.wav").exists()

    def test_metadata_has_required_fields(self, tmp_speakers):
        speaker = create_speaker("Bob", _make_wav_bytes(), "Test text.")
        meta = json.loads((tmp_speakers / speaker["id"] / "metadata.json").read_text())
        assert set(meta.keys()) == {"id", "name", "ref_text", "created_at"}


class TestListSpeakers:
    def test_empty_when_no_speakers(self, tmp_speakers):
        assert list_speakers() == []

    def test_returns_all_speakers(self, tmp_speakers):
        create_speaker("Alice", _make_wav_bytes(), "ref A")
        create_speaker("Bob", _make_wav_bytes(), "ref B")
        speakers = list_speakers()
        assert len(speakers) == 2
        names = {s["name"] for s in speakers}
        assert names == {"Alice", "Bob"}


class TestGetSpeaker:
    def test_returns_speaker_by_id(self, tmp_speakers):
        created = create_speaker("Alice", _make_wav_bytes(), "ref text")
        speaker = get_speaker(created["id"])
        assert speaker["name"] == "Alice"

    def test_returns_none_for_missing(self, tmp_speakers):
        assert get_speaker("nonexistent") is None


class TestDeleteSpeaker:
    def test_deletes_speaker_directory(self, tmp_speakers):
        created = create_speaker("Alice", _make_wav_bytes(), "ref text")
        sid = created["id"]
        assert delete_speaker(sid) is True
        assert not (tmp_speakers / sid).exists()

    def test_returns_false_for_missing(self, tmp_speakers):
        assert delete_speaker("nonexistent") is False
