import pytest

from lightning_whisper_mlx import LightningWhisperMLX
from lightning_whisper_mlx.tts import LightningTTSMLX


@pytest.mark.slow
class TestTTSToSTTRoundtrip:
    """Generate speech with F5-TTS, transcribe with Whisper, verify content.

    Note: The default F5-TTS model (lucasnewman/f5-tts-mlx) produces English speech.
    Both tests use English text since the model is English-only.
    """

    def test_pangram_roundtrip(self, tmp_path):
        """Classic pangram: TTS->STT must preserve core content words."""
        tts = LightningTTSMLX()
        wav_path = str(tmp_path / "pangram.wav")
        tts.generate(
            text="The quick brown fox jumps over the lazy dog.",
            output_path=wav_path,
            seed=42,
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe(wav_path, language="en")
        text = result["text"].lower()

        assert "quick" in text, f"Expected 'quick' in transcription: {text}"
        assert "fox" in text, f"Expected 'fox' in transcription: {text}"
        assert "dog" in text, f"Expected 'dog' in transcription: {text}"

    def test_numbers_roundtrip(self, tmp_path):
        """Sentence with numbers: TTS->STT must preserve key words."""
        tts = LightningTTSMLX()
        wav_path = str(tmp_path / "numbers.wav")
        tts.generate(
            text="There are seven days in a week and twelve months in a year.",
            output_path=wav_path,
            seed=42,
        )

        whisper = LightningWhisperMLX("tiny")
        result = whisper.transcribe(wav_path, language="en")
        text = result["text"].lower()

        assert "seven" in text or "7" in text, f"Expected 'seven' in transcription: {text}"
        assert "week" in text, f"Expected 'week' in transcription: {text}"
        assert "twelve" in text or "12" in text, f"Expected 'twelve' in transcription: {text}"
