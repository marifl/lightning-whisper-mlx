import numpy as np
import pytest


@pytest.fixture
def sine_440hz():
    """1-second 440Hz sine wave at 16kHz sample rate."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sine_1000hz():
    """1-second 1000Hz sine wave at 16kHz sample rate."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t)


@pytest.fixture
def silence():
    """1 second of silence at 16kHz."""
    return np.zeros(16000, dtype=np.float32)
