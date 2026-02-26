import pytest

from lightning_whisper_mlx.tokenizer import Tokenizer, get_tokenizer


class TestGetTokenizer:
    def test_english_only(self):
        tok = get_tokenizer(multilingual=False)
        assert isinstance(tok, Tokenizer)
        assert tok.language is None
        assert tok.task is None

    def test_multilingual_defaults(self):
        tok = get_tokenizer(multilingual=True)
        assert tok.language == "en"
        assert tok.task == "transcribe"

    def test_multilingual_with_language(self):
        tok = get_tokenizer(multilingual=True, language="de")
        assert tok.language == "de"

    def test_multilingual_translate_task(self):
        tok = get_tokenizer(multilingual=True, language="fr", task="translate")
        assert tok.task == "translate"

    def test_invalid_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            get_tokenizer(multilingual=True, language="klingon")

    def test_language_name_lookup(self):
        tok = get_tokenizer(multilingual=True, language="german")
        assert tok.language == "de"


class TestTokenizerEncodeDecode:
    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer(multilingual=True, language="en")

    def test_encode_returns_list_of_ints(self, tokenizer):
        tokens = tokenizer.encode("hello world")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_decode_returns_string(self, tokenizer):
        tokens = tokenizer.encode("hello world")
        text = tokenizer.decode(tokens)
        assert isinstance(text, str)

    def test_roundtrip(self, tokenizer):
        original = "hello world"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original

    def test_empty_string(self, tokenizer):
        tokens = tokenizer.encode("")
        assert tokens == []
        assert tokenizer.decode([]) == ""


class TestTokenizerSpecialTokens:
    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer(multilingual=True, language="en")

    def test_eot_exists(self, tokenizer):
        assert isinstance(tokenizer.eot, int)
        assert tokenizer.eot > 0

    def test_sot_exists(self, tokenizer):
        assert isinstance(tokenizer.sot, int)
        assert tokenizer.sot > 0

    def test_timestamp_begin(self, tokenizer):
        assert isinstance(tokenizer.timestamp_begin, int)
        assert tokenizer.timestamp_begin > tokenizer.eot

    def test_no_timestamps_token(self, tokenizer):
        assert isinstance(tokenizer.no_timestamps, int)

    def test_sot_sequence_structure(self, tokenizer):
        seq = tokenizer.sot_sequence
        assert isinstance(seq, tuple)
        assert len(seq) >= 1
        assert seq[0] == tokenizer.sot

    def test_sot_sequence_includes_language_and_task(self, tokenizer):
        seq = tokenizer.sot_sequence
        # For multilingual with language="en" and task="transcribe":
        # [sot, language_token, transcribe_token]
        assert len(seq) == 3

    def test_all_language_tokens(self, tokenizer):
        lang_tokens = tokenizer.all_language_tokens
        assert len(lang_tokens) > 0
        assert all(isinstance(t, int) for t in lang_tokens)

    def test_non_speech_tokens(self, tokenizer):
        non_speech = tokenizer.non_speech_tokens
        assert isinstance(non_speech, tuple)
        assert len(non_speech) > 0


class TestTokenizerWordSplit:
    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer(multilingual=True, language="en")

    def test_split_to_word_tokens(self, tokenizer):
        tokens = tokenizer.encode(" hello world")
        words, word_tokens = tokenizer.split_to_word_tokens(
            tokens + [tokenizer.eot]
        )
        assert len(words) > 0
        assert len(words) == len(word_tokens)
        # Each word should have at least one token
        assert all(len(wt) > 0 for wt in word_tokens)
