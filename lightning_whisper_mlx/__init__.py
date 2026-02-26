from .lightning import LightningWhisperMLX


def __getattr__(name):
    if name == "LightningTTSMLX":
        from .tts import LightningTTSMLX
        return LightningTTSMLX
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
