import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.audio import (
    HOP_LENGTH,
    N_FFT,
    N_SAMPLES,
    hanning,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    stft,
)


class TestMelSpectrogram:
    def test_440hz_peak_mel_bin(self, sine_440hz):
        """A 440Hz tone must concentrate energy at mel bin 11."""
        mel = log_mel_spectrogram(sine_440hz, n_mels=80)
        avg_energy = np.array(mel).mean(axis=0)
        assert np.argmax(avg_energy) == 11

    def test_silence_produces_uniform_floor(self, silence):
        """All-zeros input must produce uniform -1.5 across all mel bins."""
        mel = log_mel_spectrogram(silence, n_mels=80)
        mel_np = np.array(mel)
        assert np.allclose(mel_np, -1.5, atol=0.01)

    def test_440hz_has_higher_energy_than_silence(self, sine_440hz, silence):
        """A tone must produce higher mel energy than silence."""
        mel_tone = np.array(log_mel_spectrogram(sine_440hz, n_mels=80))
        mel_silence = np.array(log_mel_spectrogram(silence, n_mels=80))
        assert mel_tone.mean() > mel_silence.mean()

    def test_80_vs_128_mels_same_peak_region(self, sine_440hz):
        """80-mel and 128-mel spectrograms must agree on peak frequency region."""
        mel_80 = np.array(log_mel_spectrogram(sine_440hz, n_mels=80))
        mel_128 = np.array(log_mel_spectrogram(sine_440hz, n_mels=128))
        peak_80 = np.argmax(mel_80.mean(axis=0))
        peak_128 = np.argmax(mel_128.mean(axis=0))
        assert abs(peak_128 - peak_80 * 128 / 80) < 5


class TestStft:
    def test_1000hz_peak_frequency_bin(self, sine_1000hz):
        """1000Hz tone STFT must peak at frequency bin 25."""
        audio = mx.array(sine_1000hz)
        window = hanning(N_FFT)
        freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        avg_mag = np.abs(np.array(freqs)).mean(axis=0)
        assert np.argmax(avg_mag) == 25

    def test_silence_stft_near_zero(self, silence):
        """Silence must produce near-zero STFT magnitudes."""
        audio = mx.array(silence)
        window = hanning(N_FFT)
        freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        avg_mag = np.abs(np.array(freqs)).mean()
        assert avg_mag < 1e-6


class TestPadOrTrim:
    def test_pad_short_fills_zeros(self):
        """Padding a short array must append exact zeros."""
        data = mx.array([1.0, 2.0, 3.0])
        result = pad_or_trim(data, length=5)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, [1.0, 2.0, 3.0, 0.0, 0.0])

    def test_trim_long_keeps_prefix(self):
        """Trimming must keep the first N values exactly."""
        data = mx.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = pad_or_trim(data, length=3)
        result_np = np.array(result)
        np.testing.assert_array_equal(result_np, [10.0, 20.0, 30.0])

    def test_default_length_is_n_samples(self):
        """Default pad target must be N_SAMPLES (480000)."""
        short = mx.zeros(100)
        result = pad_or_trim(short)
        assert result.shape[0] == N_SAMPLES


class TestLoadAudio:
    def test_nonexistent_file_raises(self):
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio("/nonexistent/file.wav")

    def test_invalid_file_raises(self, tmp_path):
        bad_file = tmp_path / "not_audio.txt"
        bad_file.write_text("this is not audio")
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio(str(bad_file))
