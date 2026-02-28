"""Tests for dialog TTS utilities."""
import pytest

from lightning_whisper_mlx.dialog import strip_tags, chunk_text


class TestStripTags:
    def test_strips_single_tag(self):
        assert strip_tags("[warmly] Hello there.") == "Hello there."

    def test_strips_multiple_tags(self):
        assert strip_tags("[curious] Mhm, [pause] bin da.") == "Mhm, bin da."

    def test_no_tags_unchanged(self):
        assert strip_tags("Just plain text.") == "Just plain text."

    def test_strips_and_trims(self):
        assert strip_tags("[tag]  Hello  [tag2]") == "Hello"

    def test_pause_tags_stripped(self):
        assert strip_tags("[long pause] Text [short pause] more") == "Text more"


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.")
        assert chunks == ["Hello world."]

    def test_splits_at_punctuation(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, max_bytes=40)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.encode("utf-8")) <= 40

    def test_respects_max_bytes(self):
        text = "Geh auf Folie zwölf. Dann schauen wir uns das genauer an. Was siehst du dort?"
        chunks = chunk_text(text, max_bytes=50)
        for c in chunks:
            assert len(c.encode("utf-8")) <= 50

    def test_long_word_not_split(self):
        text = "Donaudampfschifffahrtsgesellschaft."
        chunks = chunk_text(text, max_bytes=135)
        assert len(chunks) == 1

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
