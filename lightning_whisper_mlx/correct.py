"""LLM-based transcription correction (optional dependency)."""

from typing import Callable, Optional

_DEFAULT_LOCAL_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def build_correction_prompt(
    text: str,
    language: Optional[str] = None,
    glossary: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Build a (system_prompt, user_prompt) pair for transcription correction.

    The LLM is instructed to return only the corrected text with no explanation.
    """
    lang_phrase = f"{language} " if language else ""
    system_prompt = (
        f"You are a transcription correction assistant. "
        f"Correct the following {lang_phrase}transcription for grammar, spelling, and accuracy. "
        "Return ONLY the corrected text with no explanation, preamble, or commentary. "
        "Preserve the original meaning and do not add or remove content."
    )
    if glossary:
        glossary_str = ", ".join(glossary)
        system_prompt += (
            f"\n\nDomain-specific vocabulary to respect (do not alter these terms): {glossary_str}."
        )
    return system_prompt, text


def _correct_with_local(
    text: str,
    language: Optional[str],
    glossary: Optional[list[str]],
    model: Optional[str] = None,
) -> str:
    """Correct text using a local mlx-lm model."""
    resolved_model: str = model if model is not None else _DEFAULT_LOCAL_MODEL
    try:
        import mlx_lm
    except ImportError as e:
        raise ImportError(
            "mlx-lm is required for local LLM correction. Install it with:\n"
            "  pip install lightning-whisper-mlx[correct]\n"
            "  # or: uv sync --extra correct"
        ) from e

    system_prompt, user_prompt = build_correction_prompt(text, language, glossary)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    loaded_model, tokenizer = mlx_lm.load(resolved_model)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = mlx_lm.generate(
        loaded_model,
        tokenizer,
        prompt=prompt,
        max_tokens=max(64, len(text.split()) * 3),
    )
    return result.strip()


def _correct_with_anthropic(
    text: str,
    language: Optional[str],
    glossary: Optional[list[str]],
    model: Optional[str] = None,
) -> str:
    """Correct text using the Anthropic API."""
    resolved_model: str = model if model is not None else _DEFAULT_ANTHROPIC_MODEL
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic is required for API-based correction. Install it with:\n"
            "  pip install lightning-whisper-mlx[correct-api]\n"
            "  # or: uv sync --extra correct-api"
        ) from e

    system_prompt, user_prompt = build_correction_prompt(text, language, glossary)
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=resolved_model,
        max_tokens=len(text.split()) * 3 + 64,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def correct_transcription(
    segments: list,
    *,
    backend: str = "anthropic",
    model: Optional[str] = None,
    glossary: Optional[list[str]] = None,
    custom_fn: Optional[Callable[[str], str]] = None,
    language: Optional[str] = None,
) -> list:
    """Correct transcription text using an LLM backend.

    Supports both segment formats:
    - List format (no diarization): [start_seek, end_seek, text]
    - Dict format (post-diarization): {"start": float, "end": float, "text": str, "speaker": str|None}

    Args:
        segments: Transcription segments in either list or dict format.
        backend: One of "local" (mlx-lm), "anthropic" (Anthropic API), or "custom".
        model: Model identifier. Defaults to a small model appropriate for the backend.
        glossary: Domain-specific terms the LLM should preserve unchanged.
        custom_fn: Required when backend="custom". Called as custom_fn(text) -> str.
        language: Language of the transcription (e.g. "English", "German").

    Returns:
        Segments in the same format as input, with text corrected.
        - Dict-format segments gain a "raw_text" key with the original text.
        - List-format segments become [start_seek, end_seek, corrected_text, raw_text].

    Raises:
        ValueError: If backend is invalid or custom_fn is missing for "custom" backend.
        ImportError: If required backend dependency is not installed.
    """
    valid_backends = {"local", "anthropic", "custom"}
    if backend not in valid_backends:
        raise ValueError(
            f"Invalid backend {backend!r}. Must be one of: {', '.join(sorted(valid_backends))}"
        )

    if backend == "custom" and custom_fn is None:
        raise ValueError(
            "custom_fn is required when backend='custom'. "
            "Pass a callable that takes a text string and returns corrected text."
        )
    if not segments:
        return segments

    dict_format = isinstance(segments[0], dict)

    result = []
    for seg in segments:
        if dict_format:
            text = seg["text"]
        else:
            text = seg[2]

        if backend == "local":
            corrected = _correct_with_local(text, language, glossary, model)
        elif backend == "anthropic":
            corrected = _correct_with_anthropic(text, language, glossary, model)
        else:
            assert custom_fn is not None  # validated by ValueError guard above
            corrected = custom_fn(text)

        if dict_format:
            new_seg = dict(seg)
            new_seg["raw_text"] = text
            new_seg["text"] = corrected
        else:
            new_seg = [seg[0], seg[1], corrected, text]

        result.append(new_seg)

    return result
