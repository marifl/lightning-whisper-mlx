from .lightning import LightningWhisperMLX


def __getattr__(name):
    if name == "LightningTTSMLX":
        from .tts import LightningTTSMLX
        return LightningTTSMLX
    if name in ("diarize_audio", "assign_speakers"):
        from . import diarize
        return getattr(diarize, name)
    if name in ("correct_transcription", "build_correction_prompt"):
        from . import correct
        return getattr(correct, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
