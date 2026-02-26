import numpy as np
import mlx.core as mx
import pytest

from lightning_whisper_mlx.audio import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    stft,
)


class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_length(self):
        assert CHUNK_LENGTH == 30

    def test_n_samples(self):
        assert N_SAMPLES == CHUNK_LENGTH * SAMPLE_RATE  # 480000

    def test_n_frames(self):
        assert N_FRAMES == N_SAMPLES // HOP_LENGTH  # 3000


class TestPadOrTrim:
    def test_pad_short_array(self):
        short = mx.zeros(100)
        result = pad_or_trim(short, length=200)
        assert result.shape[0] == 200
        # Padded portion should be zeros
        assert mx.all(result[100:] == 0).item()

    def test_trim_long_array(self):
        long = mx.ones(300)
        result = pad_or_trim(long, length=200)
        assert result.shape[0] == 200

    def test_exact_length_unchanged(self):
        exact = mx.ones(200)
        result = pad_or_trim(exact, length=200)
        assert result.shape[0] == 200
        assert mx.all(result == 1).item()

    def test_pad_preserves_data(self):
        data = mx.array([1.0, 2.0, 3.0])
        result = pad_or_trim(data, length=5)
        assert result[0].item() == 1.0
        assert result[1].item() == 2.0
        assert result[2].item() == 3.0

    def test_multidimensional_axis(self):
        data = mx.ones((10, 50))
        result = pad_or_trim(data, length=30, axis=-1)
        assert result.shape == (10, 30)

    def test_pad_default_length(self):
        short = mx.zeros(100)
        result = pad_or_trim(short)
        assert result.shape[0] == N_SAMPLES


class TestStft:
    def test_output_shape(self, sample_audio):
        audio = mx.array(sample_audio)
        from lightning_whisper_mlx.audio import hanning

        window = hanning(N_FFT)
        result = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        # STFT output: (time_frames, freq_bins) where freq_bins = N_FFT//2 + 1
        assert result.shape[1] == N_FFT // 2 + 1
        assert result.ndim == 2

    def test_output_is_complex(self, sample_audio):
        audio = mx.array(sample_audio)
        from lightning_whisper_mlx.audio import hanning

        window = hanning(N_FFT)
        result = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
        # rfft output should be complex
        assert result.dtype in (mx.complex64,)


class TestLogMelSpectrogram:
    def test_output_shape_80_mels(self, sample_audio):
        result = log_mel_spectrogram(sample_audio, n_mels=80)
        assert result.shape[-1] == 80
        assert result.ndim == 2

    def test_output_shape_128_mels(self, sample_audio):
        result = log_mel_spectrogram(sample_audio, n_mels=128)
        assert result.shape[-1] == 128

    def test_accepts_numpy_array(self, sample_audio):
        result = log_mel_spectrogram(sample_audio)
        assert isinstance(result, mx.array)

    def test_accepts_mx_array(self, sample_audio):
        result = log_mel_spectrogram(mx.array(sample_audio))
        assert isinstance(result, mx.array)

    def test_padding_increases_frames(self, sample_audio):
        without_pad = log_mel_spectrogram(sample_audio, padding=0)
        with_pad = log_mel_spectrogram(sample_audio, padding=16000)
        assert with_pad.shape[0] > without_pad.shape[0]

    def test_output_values_finite(self, sample_audio):
        result = log_mel_spectrogram(sample_audio)
        result_np = np.array(result)
        assert np.all(np.isfinite(result_np))


class TestLoadAudio:
    def test_nonexistent_file_raises(self):
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio("/nonexistent/file.wav")

    def test_invalid_file_raises(self, tmp_path):
        bad_file = tmp_path / "not_audio.txt"
        bad_file.write_text("this is not audio")
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio(str(bad_file))
