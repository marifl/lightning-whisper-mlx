from lightning_whisper_mlx.tokenizer import get_tokenizer


class TestSpecialTokenIds:
    """Whisper special tokens must have exact known IDs from the tokenizer spec."""

    def test_eot(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.eot == 50257

    def test_sot(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.sot == 50258

    def test_timestamp_begin(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.timestamp_begin == 50364

    def test_no_timestamps(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.no_timestamps == 50363


class TestLanguageTokens:
    """Language tokens must map to exact known IDs."""

    def test_english_token(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.language_token == 50259  # <|en|>

    def test_german_token(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.language_token == 50261  # <|de|>

    def test_french_token(self):
        tok = get_tokenizer(multilingual=True, language="fr")
        assert tok.language_token == 50265  # <|fr|>

    def test_language_name_lookup_resolves(self):
        """Passing 'german' must resolve to language code 'de'."""
        tok = get_tokenizer(multilingual=True, language="german")
        assert tok.language == "de"
        assert tok.language_token == 50261


class TestSotSequence:
    """The start-of-transcript sequence must contain exact token IDs."""

    def test_english_transcribe(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.sot_sequence == (50258, 50259, 50359)

    def test_english_translate(self):
        tok = get_tokenizer(multilingual=True, language="en", task="translate")
        assert tok.sot_sequence == (50258, 50259, 50358)

    def test_german_transcribe(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.sot_sequence == (50258, 50261, 50359)


class TestEncodeDecode:
    """Encoding and decoding must produce exact known token sequences."""

    def test_encode_hello(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.encode(" hello") == [7751]

    def test_roundtrip(self):
        tok = get_tokenizer(multilingual=True, language="en")
        original = " hello world"
        decoded = tok.decode(tok.encode(original))
        assert decoded == original

    def test_empty_string(self):
        tok = get_tokenizer(multilingual=True, language="en")
        assert tok.encode("") == []
        assert tok.decode([]) == ""


class TestSplitToWordTokens:
    """Word splitting must produce correct word boundaries."""

    def test_hello_world_splits_into_two_words(self):
        tok = get_tokenizer(multilingual=True, language="en")
        tokens = tok.encode(" hello world")
        words, word_tokens = tok.split_to_word_tokens(tokens + [tok.eot])
        assert len(words) == 2
        assert words[0].strip() == "hello"
        assert words[1].strip() == "world"


class TestInvalidInput:
    def test_invalid_language_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported language"):
            get_tokenizer(multilingual=True, language="klingon")
