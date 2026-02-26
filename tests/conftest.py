import pytest
import numpy as np
import mlx.core as mx


@pytest.fixture
def sample_audio():
    """Generate a 1-second sine wave at 440Hz as test audio (16kHz sample rate)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sample_mel(sample_audio):
    """Generate mel spectrogram from sample audio."""
    from lightning_whisper_mlx.audio import log_mel_spectrogram

    return log_mel_spectrogram(sample_audio)
