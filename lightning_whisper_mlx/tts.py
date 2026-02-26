from typing import Optional


class LightningTTSMLX:
    """Text-to-speech wrapper around f5-tts-mlx for Apple Silicon.

    Requires the optional `tts` extra: pip install lightning-whisper-mlx[tts]
    """

    def __init__(self, model: str = "lucasnewman/f5-tts-mlx"):
        self.model_name = model
        self._f5_generate = None

    def _ensure_loaded(self):
        if self._f5_generate is not None:
            return
        try:
            from f5_tts_mlx.generate import generate
        except ImportError:
            raise ImportError(
                "f5-tts-mlx is required for TTS. Install it with:\n"
                "  pip install lightning-whisper-mlx[tts]\n"
                "  # or: uv sync --extra tts"
            )
        self._f5_generate = generate

    def generate(
        self,
        text: str,
        output_path: str,
        *,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        steps: int = 8,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            output_path: Path for output WAV file.
            ref_audio: Reference audio path for voice cloning.
            ref_text: Transcript of reference audio.
            steps: Diffusion steps (more = better quality, slower).
            speed: Speech speed multiplier.
            seed: Random seed for reproducibility.

        Returns:
            The output_path where the WAV file was written.
        """
        self._ensure_loaded()
        self._f5_generate(
            generation_text=text,
            model_name=self.model_name,
            ref_audio_path=ref_audio,
            ref_audio_text=ref_text,
            steps=steps,
            speed=speed,
            seed=seed,
            output_path=output_path,
        )
        return output_path
