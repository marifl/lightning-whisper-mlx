"""Tests for the correction module (correct.py).

Mocking approach:
- mlx_lm.generate and mlx_lm.load are monkeypatched to avoid real LLM calls.
- anthropic.Anthropic is monkeypatched to avoid real API calls.
- transcribe_audio and LightningWhisperMLX.__init__ are monkeypatched for integration tests.
"""

import sys
import types
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_list_segment(start_seek: int, end_seek: int, text: str) -> list:
    return [start_seek, end_seek, text]


def _make_dict_segment(start: float, end: float, text: str, speaker: str = "SPEAKER_00") -> dict:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def _patch_mlx_lm(monkeypatch, corrected_text: str = "Corrected text."):
    """Inject a fake mlx_lm module that returns corrected_text from generate()."""
    fake_mlx_lm = types.ModuleType("mlx_lm")

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "fake_prompt"

    def fake_load(model_path):
        return object(), FakeTokenizer()

    def fake_generate(model, tokenizer, prompt, **kwargs):
        return corrected_text

    fake_mlx_lm.load = fake_load  # type: ignore[attr-defined]
    fake_mlx_lm.generate = fake_generate  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    return fake_mlx_lm


def _patch_anthropic(monkeypatch, corrected_text: str = "Corrected text."):
    """Inject a fake anthropic module that returns corrected_text from messages.create()."""
    fake_anthropic = types.ModuleType("anthropic")

    class FakeMessage:
        content = [types.SimpleNamespace(text=corrected_text)]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeMessage()

    class FakeClient:
        messages = FakeMessages()

    class FakeAnthropic:
        def __new__(cls, *args, **kwargs):
            return FakeClient()

    fake_anthropic.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return fake_anthropic


# ---------------------------------------------------------------------------
# TestBuildCorrectionPrompt
# ---------------------------------------------------------------------------

class TestBuildCorrectionPrompt:
    """build_correction_prompt constructs an LLM prompt from transcription text.

    Assertions are on CONCRETE expected content — not vague structural checks.
    """

    def test_input_text_appears_verbatim_in_user_prompt(self):
        """The exact input text must appear in the user/content part of the prompt."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        text = "hello wrold, this iz a tset."
        result = build_correction_prompt(text, language="en")
        # Support both single-string and (system, user) tuple return shapes
        combined = result if isinstance(result, str) else " ".join(str(p) for p in result)
        assert text in combined, (
            f"Input text {text!r} must appear verbatim in the prompt"
        )

    def test_language_code_fr_appears_in_prompt(self):
        """The language code 'fr' must appear so the LLM knows the target language."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        result = build_correction_prompt("Bonjour.", language="fr")
        combined = result if isinstance(result, str) else " ".join(str(p) for p in result)
        assert "fr" in combined, "Language code 'fr' must appear in the prompt"

    def test_language_code_es_also_appears(self):
        """Language is not hard-coded — a different code must also appear."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        result = build_correction_prompt("Hola mundo.", language="es")
        combined = result if isinstance(result, str) else " ".join(str(p) for p in result)
        assert "es" in combined, "Language code 'es' must appear in the prompt"

    def test_glossary_terms_appear_verbatim(self):
        """Each glossary term must appear literally so the LLM can preserve spelling."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        glossary = ["Kubernetes", "kubectl", "PyAnnote"]
        result = build_correction_prompt("Some text.", language="en", glossary=glossary)
        combined = result if isinstance(result, str) else " ".join(str(p) for p in result)
        for term in glossary:
            assert term in combined, f"Glossary term {term!r} must appear verbatim in prompt"

    def test_no_glossary_omits_glossary_word(self):
        """When glossary=None the word 'glossary' should not appear — clean prompt."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        result = build_correction_prompt("Some text.", language="en", glossary=None)
        combined = result if isinstance(result, str) else " ".join(str(p) for p in result)
        assert "glossary" not in combined.lower(), (
            "Prompt must not mention 'glossary' when none is provided"
        )

    def test_prompt_instructs_return_only_corrected_text(self):
        """Prompt must tell the LLM to return ONLY corrected text without explanations."""
        from lightning_whisper_mlx.correct import build_correction_prompt

        result = build_correction_prompt("Test.", language="en")
        combined = (result if isinstance(result, str) else " ".join(str(p) for p in result)).lower()
        has_instruction = (
            "only" in combined
            or "no explanation" in combined
            or "no commentary" in combined
            or "just the" in combined
        )
        assert has_instruction, (
            "Prompt must instruct LLM to return only corrected text without explanations"
        )


# ---------------------------------------------------------------------------
# TestCorrectTranscription
# ---------------------------------------------------------------------------

class TestCorrectTranscription:
    """correct_transcription applies LLM correction to transcription segments.

    Each test injects a controlled fake backend and asserts on EXACT output values.
    """

    def test_local_backend_text_is_exact_llm_output(self, monkeypatch):
        """text must be EXACTLY what fake mlx_lm.generate returned, not the original."""
        _patch_mlx_lm(monkeypatch, corrected_text="Hello world.")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription(
            [_make_dict_segment(0.0, 3.0, "hello wrold")],
            backend="local",
        )

        assert result[0]["text"] == "Hello world.", (
            "text must be exactly the string returned by mlx_lm.generate"
        )
        assert result[0]["raw_text"] == "hello wrold", (
            "raw_text must be the original pre-correction input, not the corrected text"
        )

    def test_anthropic_backend_text_is_exact_api_response(self, monkeypatch):
        """text must be exactly the string the fake Anthropic API returned."""
        _patch_anthropic(monkeypatch, corrected_text="Anthropic corrected.")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription(
            [_make_dict_segment(0.0, 3.0, "orignal txt")],
            backend="anthropic",
        )

        assert result[0]["text"] == "Anthropic corrected."
        assert result[0]["raw_text"] == "orignal txt"

    def test_custom_fn_receives_exact_text_and_result_is_used(self):
        """custom_fn must receive the exact segment text; its return value must be result text."""
        received = []

        def my_fn(text: str) -> str:
            received.append(text)
            return "Custom corrected."

        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription(
            [_make_dict_segment(0.0, 3.0, "teh qiuck brwon fox")],
            backend="custom",
            custom_fn=my_fn,
        )

        assert received == ["teh qiuck brwon fox"], (
            "custom_fn must be called with exactly the original segment text"
        )
        assert result[0]["text"] == "Custom corrected.", (
            "result text must be exactly what custom_fn returned"
        )
        assert result[0]["raw_text"] == "teh qiuck brwon fox"

    def test_start_end_speaker_preserved_exactly_after_correction(self, monkeypatch):
        """start=1.234, end=5.678, speaker='SPEAKER_02' must survive unchanged."""
        _patch_mlx_lm(monkeypatch, corrected_text="Fixed.")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription(
            [_make_dict_segment(1.234, 5.678, "Orignal.", speaker="SPEAKER_02")],
            backend="local",
        )

        assert result[0]["start"] == 1.234
        assert result[0]["end"] == 5.678
        assert result[0]["speaker"] == "SPEAKER_02"

    def test_list_segment_seek_values_preserved_and_text_corrected(self, monkeypatch):
        """List-format: start_seek=150 and end_seek=450 must be unchanged; text corrected."""
        _patch_mlx_lm(monkeypatch, corrected_text="Fixed text.")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription([_make_list_segment(150, 450, "Orignal.")], backend="local")

        out = result[0]
        if isinstance(out, list):
            assert out[0] == 150, "start_seek must be preserved"
            assert out[1] == 450, "end_seek must be preserved"
            assert out[2] == "Fixed text.", "text at index 2 must be the corrected string"
        else:
            # dict output: seek values in start_seek/end_seek keys, or seconds in start/end
            assert "start_seek" in out or "start" in out
            assert out.get("text") == "Fixed text."

    def test_raw_text_stored_for_list_segment(self, monkeypatch):
        """Original text must be in raw_text key (dict) or index 3 (list)."""
        _patch_mlx_lm(monkeypatch, corrected_text="Better text.")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription([_make_list_segment(0, 300, "Orignal text.")], backend="local")

        out = result[0]
        if isinstance(out, list):
            assert out[3] == "Orignal text.", "raw_text must be at index 3 for list segments"
        else:
            assert out["raw_text"] == "Orignal text."

    def test_each_segment_corrected_independently(self):
        """All segments corrected; each gets the right corrected text and preserved metadata."""
        outputs = ["First corrected.", "Second corrected.", "Third corrected."]
        call_index = [0]

        def ordered_corrector(text: str) -> str:
            out = outputs[call_index[0]]
            call_index[0] += 1
            return out

        from lightning_whisper_mlx.correct import correct_transcription

        segments = [
            _make_dict_segment(0.0, 2.0, "First errror.", speaker="SPEAKER_00"),
            _make_dict_segment(2.0, 4.0, "Scond errror.", speaker="SPEAKER_01"),
            _make_dict_segment(4.0, 6.0, "Thrid errror.", speaker="SPEAKER_00"),
        ]
        result = correct_transcription(segments, backend="custom", custom_fn=ordered_corrector)

        assert len(result) == 3
        assert result[0]["text"] == "First corrected."
        assert result[0]["raw_text"] == "First errror."
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["text"] == "Second corrected."
        assert result[1]["raw_text"] == "Scond errror."
        assert result[1]["speaker"] == "SPEAKER_01"
        assert result[2]["text"] == "Third corrected."
        assert result[2]["raw_text"] == "Thrid errror."

    def test_llm_response_whitespace_is_stripped(self, monkeypatch):
        """Leading/trailing whitespace in the LLM response must be stripped from text."""
        _patch_mlx_lm(monkeypatch, corrected_text="  Hello world.  \n")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription(
            [_make_dict_segment(0.0, 1.0, "hello wrold")],
            backend="local",
        )

        assert result[0]["text"] == "Hello world.", (
            "Whitespace around LLM response must be stripped"
        )

    def test_empty_segment_text_does_not_crash(self, monkeypatch):
        """A segment with empty text must not raise; raw_text must be empty string."""
        _patch_mlx_lm(monkeypatch, corrected_text="")
        from lightning_whisper_mlx.correct import correct_transcription

        result = correct_transcription([_make_dict_segment(0.0, 1.0, "")], backend="local")

        assert len(result) == 1
        assert result[0]["raw_text"] == ""

    def test_empty_segments_returns_empty(self, monkeypatch):
        """Empty input list returns empty list without errors."""
        _patch_mlx_lm(monkeypatch)
        from lightning_whisper_mlx.correct import correct_transcription

        assert correct_transcription([], backend="local") == []

    def test_invalid_backend_raises_valueerror_naming_bad_value(self):
        """ValueError message must contain the invalid backend name."""
        from lightning_whisper_mlx.correct import correct_transcription

        with pytest.raises(ValueError, match="notabackend"):
            correct_transcription(
                [_make_dict_segment(0.0, 1.0, "text.")],
                backend="notabackend",
            )

    def test_custom_without_fn_raises_valueerror_mentioning_custom_fn(self):
        """backend='custom' with custom_fn=None raises ValueError mentioning 'custom_fn'."""
        from lightning_whisper_mlx.correct import correct_transcription

        with pytest.raises(ValueError, match="custom_fn"):
            correct_transcription(
                [_make_dict_segment(0.0, 1.0, "text.")],
                backend="custom",
                custom_fn=None,
            )


# ---------------------------------------------------------------------------
# TestCorrectTranscriptionGuards
# ---------------------------------------------------------------------------

class TestCorrectTranscriptionGuards:
    """correct_transcription raises ImportError with actionable install instructions."""

    def test_missing_mlx_lm_raises_import_error_with_install_hint(self, monkeypatch):
        """Missing mlx-lm → ImportError whose message mentions how to install it."""
        # Setting to None makes `import mlx_lm` raise ImportError
        monkeypatch.setitem(sys.modules, "mlx_lm", None)  # type: ignore[arg-type]
        from lightning_whisper_mlx.correct import correct_transcription

        with pytest.raises(ImportError) as exc_info:
            correct_transcription(
                [_make_dict_segment(0.0, 1.0, "text.")],
                backend="local",
            )

        msg = str(exc_info.value).lower()
        has_hint = "mlx" in msg or "lightning-whisper-mlx" in msg or "uv sync" in msg or "pip" in msg
        assert has_hint, (
            f"ImportError must contain install instructions; got: {exc_info.value!r}"
        )

    def test_missing_anthropic_raises_import_error_with_install_hint(self, monkeypatch):
        """Missing anthropic → ImportError whose message mentions how to install it."""
        monkeypatch.setitem(sys.modules, "anthropic", None)  # type: ignore[arg-type]
        from lightning_whisper_mlx.correct import correct_transcription

        with pytest.raises(ImportError) as exc_info:
            correct_transcription(
                [_make_dict_segment(0.0, 1.0, "text.")],
                backend="anthropic",
            )

        msg = str(exc_info.value).lower()
        has_hint = "anthropic" in msg or "pip install" in msg or "uv" in msg
        assert has_hint, (
            f"ImportError must contain install instructions; got: {exc_info.value!r}"
        )


# ---------------------------------------------------------------------------
# TestTranscribeCorrectIntegration
# ---------------------------------------------------------------------------

def _mock_whisper_init(self, *args, **kwargs):
    """Mock __init__ that sets required attributes without downloading models."""
    self.name = "tiny"
    self.batch_size = 12


class TestTranscribeCorrectIntegration:
    """LightningWhisperMLX.transcribe(correct=True/False) pipeline behavior."""

    def test_correct_true_uses_corrected_output_not_raw(self, monkeypatch):
        """When correct=True, result segments must come from correct_transcription."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        corrected = [{"start": 0.0, "end": 3.0, "text": "Hello world.", "raw_text": "hello wrold"}]

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "hello wrold", "segments": [[0, 300, "hello wrold"]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            lambda segments, **kw: corrected,
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        result = LightningWhisperMLX("tiny").transcribe("dummy.wav", correct=True)

        assert result["segments"][0]["text"] == "Hello world.", (
            "Output must be the corrected text, not the raw transcription"
        )
        assert result["segments"][0]["raw_text"] == "hello wrold", (
            "raw_text must preserve what transcribe_audio produced"
        )

    def test_correct_false_never_calls_correct_transcription(self, monkeypatch):
        """When correct=False, correct_transcription must not be invoked at all."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        called = []

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "hello wrold", "segments": [[0, 300, "hello wrold"]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            lambda *a, **kw: called.append(True) or [],
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        result = LightningWhisperMLX("tiny").transcribe("dummy.wav", correct=False)

        assert called == [], "correct_transcription must NOT be called when correct=False"
        assert result["segments"] == [[0, 300, "hello wrold"]], (
            "Raw list segments must be returned unchanged"
        )

    def test_diarize_runs_before_correct_receives_dict_segments(self, monkeypatch):
        """diarize=True + correct=True: correct_transcription must receive dict segments
        (speaker labels already assigned), not raw list segments."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        segments_received = []

        def capturing_correct(segments, **kwargs):
            segments_received.extend(segments)
            return [{"start": 0.0, "end": 3.0, "text": "Hi there corrected.",
                     "speaker": "SPEAKER_00", "raw_text": "Hi there."}]

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "Hi there.", "segments": [[0, 300, "Hi there."]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.diarize.diarize_audio",
            lambda *a, **kw: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0}],
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            capturing_correct,
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        result = LightningWhisperMLX("tiny").transcribe("dummy.wav", diarize=True, correct=True)

        assert len(segments_received) == 1
        assert isinstance(segments_received[0], dict), (
            "correct_transcription must receive dict segments — diarize must run first"
        )
        assert segments_received[0]["speaker"] == "SPEAKER_00", (
            "Segment passed to correct must already have speaker label from diarize"
        )
        out = result["segments"][0]
        assert out["speaker"] == "SPEAKER_00"
        assert out["text"] == "Hi there corrected."
        assert out["raw_text"] == "Hi there."

    def test_glossary_forwarded_unchanged(self, monkeypatch):
        """The exact glossary list must arrive at correct_transcription unchanged."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        captured = {}

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "x", "segments": [[0, 300, "x"]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            lambda segments, **kw: (captured.update(kw) or segments),
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        LightningWhisperMLX("tiny").transcribe("dummy.wav", correct=True, glossary=["MLX", "Whisper"])

        assert "glossary" in captured, "glossary kwarg must be forwarded to correct_transcription"
        assert captured["glossary"] == ["MLX", "Whisper"], (
            "glossary must be forwarded unchanged — not mutated or dropped"
        )

    def test_correct_backend_param_forwarded_as_backend_kwarg(self, monkeypatch):
        """correct_backend='anthropic' on transcribe() must arrive as backend='anthropic'."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        captured = {}

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "x", "segments": [[0, 300, "x"]], "language": "en"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            lambda segments, **kw: (captured.update(kw) or segments),
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        LightningWhisperMLX("tiny").transcribe("dummy.wav", correct=True, correct_backend="anthropic")

        assert captured.get("backend") == "anthropic", (
            "correct_backend='anthropic' must arrive as backend='anthropic' in correct_transcription"
        )

    def test_language_from_transcription_forwarded(self, monkeypatch):
        """language from transcription result must reach correct_transcription."""
        from lightning_whisper_mlx.lightning import LightningWhisperMLX

        captured = {}

        monkeypatch.setattr(
            "lightning_whisper_mlx.lightning.transcribe_audio",
            lambda *a, **kw: {"text": "Hallo.", "segments": [[0, 300, "Hallo."]], "language": "de"},
        )
        monkeypatch.setattr(
            "lightning_whisper_mlx.correct.correct_transcription",
            lambda segments, **kw: (captured.update(kw) or segments),
        )
        monkeypatch.setattr(LightningWhisperMLX, "__init__", _mock_whisper_init)

        LightningWhisperMLX("tiny").transcribe("dummy.wav", correct=True)

        assert captured.get("language") == "de", (
            "Language detected by transcription ('de') must be forwarded to correct_transcription"
        )

    def test_list_format_raw_text_preserved_at_index_3(self, monkeypatch):
        """For list-format segments, raw_text must be at index 3 of the output list."""
        _patch_mlx_lm(monkeypatch, corrected_text="Fixed text.")
        from lightning_whisper_mlx.correct import correct_transcription

        segments = [_make_list_segment(0, 300, "Orignal text.")]
        result = correct_transcription(segments, backend="local")

        out = result[0]
        assert isinstance(out, list), "List-format input must produce list-format output"
        assert len(out) == 4, "Output list must be [start_seek, end_seek, corrected, raw_text]"
        assert out[0] == 0, "start_seek must be preserved"
        assert out[1] == 300, "end_seek must be preserved"
        assert out[2] == "Fixed text.", "Corrected text at index 2"
        assert out[3] == "Orignal text.", "Original raw text preserved at index 3"
