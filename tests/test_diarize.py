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
