import pytest

from lightning_whisper_mlx.audio import HOP_LENGTH, SAMPLE_RATE


def _seek(seconds: float) -> int:
    """Convert seconds to seek position (mel frames)."""
    return int(seconds * SAMPLE_RATE / HOP_LENGTH)


class TestAssignSpeakers:
    """assign_speakers must match segments to speaker turns by temporal overlap."""

    def test_two_speakers_clean_turns(self):
        """Each segment falls entirely within one speaker turn."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(0.0), _seek(3.0), "Hello from speaker A."],
            [_seek(3.0), _seek(6.0), "Hello from speaker B."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 6.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello from speaker A."
        assert result[0]["start"] == pytest.approx(0.0, abs=0.02)
        assert result[0]["end"] == pytest.approx(3.0, abs=0.02)
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_single_speaker(self):
        """All segments assigned to the sole speaker."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(0.0), _seek(2.0), "First sentence."],
            [_seek(2.0), _seek(4.0), "Second sentence."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert all(s["speaker"] == "SPEAKER_00" for s in result)

    def test_overlap_assigns_dominant_speaker(self):
        """Segment spanning two turns gets the speaker with more overlap."""
        from lightning_whisper_mlx.diarize import assign_speakers

        # Segment 1.0-4.0s: speaker A covers 1.0-2.0 (1s), speaker B covers 2.0-4.0 (2s)
        segments = [
            [_seek(1.0), _seek(4.0), "Overlapping segment."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 5.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["speaker"] == "SPEAKER_01"  # 2s > 1s overlap

    def test_segment_in_silence_gap(self):
        """Segment falling in a gap between speaker turns gets speaker None."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [
            [_seek(5.0), _seek(7.0), "Nobody is speaking here."],
        ]
        speaker_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0},
            {"speaker": "SPEAKER_01", "start": 10.0, "end": 13.0},
        ]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["speaker"] is None

    def test_empty_segments(self):
        """Empty segments list returns empty list."""
        from lightning_whisper_mlx.diarize import assign_speakers

        result = assign_speakers([], [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}])
        assert result == []

    def test_empty_speaker_turns(self):
        """No speaker turns means all segments get speaker None."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [[_seek(0.0), _seek(2.0), "Some text."]]
        result = assign_speakers(segments, [])

        assert result[0]["speaker"] is None

    def test_output_has_seconds_not_seeks(self):
        """Output start/end must be in seconds, not seek positions."""
        from lightning_whisper_mlx.diarize import assign_speakers

        segments = [[_seek(1.5), _seek(3.5), "Test."]]
        speaker_turns = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]
        result = assign_speakers(segments, speaker_turns)

        assert result[0]["start"] == pytest.approx(1.5, abs=0.02)
        assert result[0]["end"] == pytest.approx(3.5, abs=0.02)


class TestDiarizeAudioGuards:
    """diarize_audio must raise clear errors for missing dependencies."""

    def test_missing_pyannote_raises_import_error(self, monkeypatch):
        """Missing pyannote-audio raises ImportError with install instructions."""
        import lightning_whisper_mlx.diarize as diarize_mod

        # Block pyannote.audio import
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "pyannote.audio" or name.startswith("pyannote.audio."):
                raise ImportError("No module named 'pyannote.audio'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        # Clear any cached pipeline
        diarize_mod._pipeline_cache.clear()
        monkeypatch.setenv("HF_TOKEN", "fake-token")

        with pytest.raises(ImportError, match="uv sync --extra diarize"):
            diarize_mod.diarize_audio("dummy.wav")

    def test_missing_hf_token_raises(self, monkeypatch):
        """Missing HF_TOKEN raises EnvironmentError."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Only test if pyannote is actually installed
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            pytest.skip("pyannote-audio not installed")

        from lightning_whisper_mlx.diarize import diarize_audio

        with pytest.raises(EnvironmentError, match="HF_TOKEN"):
            diarize_audio("dummy.wav")


class TestTranscribeDiarizeIntegration:
    """transcribe(diarize=True) must wire diarize_audio + assign_speakers correctly."""

    def test_diarize_transforms_segments_to_dicts(self, monkeypatch):
        """When diarize=True, segments become dicts with speaker labels."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        mock_transcription = {
            "text": "Hello world.",
            "segments": [[0, 300, "Hello world."]],
            "language": "en",
        }

        mock_turns = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
        ]

        # Patch transcribe_audio to avoid loading a real model
        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *args, **kwargs: mock_transcription,
        )
        # Patch diarize_audio to avoid needing pyannote
        monkeypatch.setattr(
            "lightning_whisper_mlx.diarize.diarize_audio",
            lambda *args, **kwargs: mock_turns,
        )
        # Patch __init__ to avoid downloading models
        monkeypatch.setattr(
            LightningWhisperMLX, "__init__",
            lambda self, *args, **kwargs: setattr(self, "name", "tiny") or setattr(self, "batch_size", 12),
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe("dummy.wav", diarize=True)

        assert isinstance(result["segments"][0], dict)
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert "start" in result["segments"][0]
        assert "end" in result["segments"][0]
        assert result["segments"][0]["text"] == "Hello world."

    def test_diarize_false_preserves_original_format(self, monkeypatch):
        """When diarize=False, segments stay as lists (backward compatible)."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        mock_transcription = {
            "text": "Hello world.",
            "segments": [[0, 300, "Hello world."]],
            "language": "en",
        }

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *args, **kwargs: mock_transcription,
        )
        monkeypatch.setattr(
            LightningWhisperMLX, "__init__",
            lambda self, *args, **kwargs: setattr(self, "name", "tiny") or setattr(self, "batch_size", 12),
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe("dummy.wav", diarize=False)

        assert isinstance(result["segments"][0], list)

    def test_speaker_count_params_forwarded(self, monkeypatch):
        """num_speakers/min_speakers/max_speakers reach diarize_audio."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        captured = {}

        def mock_diarize(audio_path, **kwargs):
            captured.update(kwargs)
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0}]

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *args, **kwargs: {"text": "", "segments": [[0, 300, "Hi."]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.diarize.diarize_audio",
            mock_diarize,
        )
        monkeypatch.setattr(
            LightningWhisperMLX, "__init__",
            lambda self, *args, **kwargs: setattr(self, "name", "tiny") or setattr(self, "batch_size", 12),
        )

        whisper = LightningWhisperMLX("tiny")
        whisper.transcribe("dummy.wav", diarize=True, num_speakers=2, min_speakers=1, max_speakers=3)

        assert captured["num_speakers"] == 2
        assert captured["min_speakers"] == 1
        assert captured["max_speakers"] == 3
