from .transcribe import transcribe_audio
from huggingface_hub import hf_hub_download 

models = {
    "tiny": {
        "base": "mlx-community/whisper-tiny", 
        "4bit": "mlx-community/whisper-tiny-mlx-4bit", 
        "8bit": "mlx-community/whisper-tiny-mlx-8bit"
    }, 
    "small": {
        "base": "mlx-community/whisper-small-mlx", 
        "4bit": "mlx-community/whisper-small-mlx-4bit", 
        "8bit": "mlx-community/whisper-small-mlx-8bit"
    },
    "distil-small.en": {
        "base": "mustafaaljadery/distil-whisper-mlx", 
    },
    "base": {
        "base" : "mlx-community/whisper-base-mlx", 
        "4bit" : "mlx-community/whisper-base-mlx-4bit",
        "8bit" : "mlx-community/whisper-base-mlx-8bit"
    },
    "medium": {
        "base": "mlx-community/whisper-medium-mlx",
        "4bit": "mlx-community/whisper-medium-mlx-4bit", 
        "8bit": "mlx-community/whisper-medium-mlx-8bit"
    }, 
    "distil-medium.en": {
        "base": "mustafaaljadery/distil-whisper-mlx", 
    }, 
    "large": {
        "base": "mlx-community/whisper-large-mlx", 
        "4bit": "mlx-community/whisper-large-mlx-4bit", 
        "8bit": "mlx-community/whisper-large-mlx-8bit", 
    },
    "large-v2": {
        "base": "mlx-community/whisper-large-v2-mlx",
        "4bit": "mlx-community/whisper-large-v2-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v2-mlx-8bit", 
    },
    "distil-large-v2": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large-v3": {
        "base": "mlx-community/whisper-large-v3-mlx",
        "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v3-mlx-8bit", 
    },
    "distil-large-v3": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
}

class LightningWhisperMLX():
    def __init__(self, model, batch_size=12, quant=None):
        if quant and (quant != "4bit" and quant != "8bit"):
            raise ValueError("Quantization must be `4bit` or `8bit`")

        if model not in models:
            raise ValueError("Please select a valid model")

        if quant and "distil" in model:
            raise ValueError(
                f"Quantization is not supported for distilled model '{model}'. "
                "Distilled models are already optimized for speed. Use quant=None."
            )

        self.name = model
        self.batch_size = batch_size

        if quant:
            repo_id = models[model][quant]
        else:
            repo_id = models[model]['base']

        if "distil" in model:
            filename1 = f"./mlx_models/{self.name}/weights.npz"
            filename2 = f"./mlx_models/{self.name}/config.json"
            local_dir = "./"
        else:
            filename1 = "weights.npz"
            filename2 = "config.json"
            local_dir = f"./mlx_models/{self.name}"

        hf_hub_download(repo_id=repo_id, filename=filename1, local_dir=local_dir)
        hf_hub_download(repo_id=repo_id, filename=filename2, local_dir=local_dir)
    
    def transcribe(self, audio_path, language=None, diarize=False,
                   num_speakers=None, min_speakers=None, max_speakers=None):
        result = transcribe_audio(audio_path, path_or_hf_repo=f'./mlx_models/{self.name}', language=language, batch_size=self.batch_size)
        if diarize:
            from .diarize import diarize_audio, assign_speakers
            speaker_turns = diarize_audio(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            result["segments"] = assign_speakers(result["segments"], speaker_turns)
        return result