from .lightning import LightningWhisperMLX


def __getattr__(name):
    if name == "LightningTTSMLX":
        from .tts import LightningTTSMLX
        return LightningTTSMLX
    if name in ("diarize_audio", "assign_speakers"):
        from . import diarize
        return getattr(diarize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
