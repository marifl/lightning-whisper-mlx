import mlx.core as mx
import numpy as np
import pytest

from lightning_whisper_mlx.decoding import (
    GreedyDecoder,
    SuppressBlank,
    SuppressTokens,
    compression_ratio,
)
from lightning_whisper_mlx.tokenizer import get_tokenizer


class TestCompressionRatio:
    def test_repetitive_text_high_ratio(self):
        text = "aaa" * 100
        ratio = compression_ratio(text)
        # Highly repetitive text compresses very well, ratio should be high
        assert ratio > 5.0

    def test_random_text_low_ratio(self):
        import string
        import random

        random.seed(42)
        text = "".join(random.choices(string.ascii_letters + string.digits, k=200))
        ratio = compression_ratio(text)
        # Random text doesn't compress well
        assert ratio < 3.0

    def test_normal_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        ratio = compression_ratio(text)
        assert ratio > 0
        assert isinstance(ratio, float)

    def test_empty_string_does_not_crash(self):
        # zlib.compress of empty bytes produces a small header
        ratio = compression_ratio("")
        assert ratio >= 0


class TestGreedyDecoder:
    @pytest.fixture
    def eot_token(self):
        return 50257  # typical Whisper EOT

    def test_temperature_zero_picks_argmax(self, eot_token):
        decoder = GreedyDecoder(temperature=0, eot=eot_token)
        batch_size = 1
        vocab_size = 100

        # Create logits where token 42 has the highest value
        logits = mx.zeros((batch_size, vocab_size))
        logits = logits.at[:, 42].add(10.0)

        tokens = mx.zeros((batch_size, 1), dtype=mx.int32)
        sum_logprobs = mx.zeros((batch_size,))

        new_tokens, completed, new_sum = decoder.update(tokens, logits, sum_logprobs)
        # Should pick token 42 (argmax)
        assert new_tokens[0, -1].item() == 42
        assert not completed.item()

    def test_temperature_nonzero_samples(self, eot_token):
        decoder = GreedyDecoder(temperature=1.0, eot=eot_token)
        batch_size = 1
        vocab_size = 100

        # Create logits where one token is extremely dominant
        logits = mx.full((batch_size, vocab_size), -100.0)
        logits = logits.at[:, 7].add(200.0)

        tokens = mx.zeros((batch_size, 1), dtype=mx.int32)
        sum_logprobs = mx.zeros((batch_size,))

        mx.random.seed(42)
        new_tokens, completed, new_sum = decoder.update(tokens, logits, sum_logprobs)
        # With temperature=1.0 and one extremely dominant token,
        # sampling should still overwhelmingly pick token 7
        assert new_tokens[0, -1].item() == 7

    def test_eot_propagation(self, eot_token):
        decoder = GreedyDecoder(temperature=0, eot=eot_token)
        batch_size = 1
        vocab_size = eot_token + 10

        # Previous token was EOT
        tokens = mx.array([[eot_token]], dtype=mx.int32)
        logits = mx.zeros((batch_size, vocab_size))
        logits = logits.at[:, 5].add(10.0)  # argmax would be 5
        sum_logprobs = mx.zeros((batch_size,))

        new_tokens, completed, new_sum = decoder.update(tokens, logits, sum_logprobs)
        # After EOT, next token should also be EOT regardless of logits
        assert new_tokens[0, -1].item() == eot_token
        assert completed.item()

    def test_tokens_grow_by_one(self, eot_token):
        decoder = GreedyDecoder(temperature=0, eot=eot_token)
        batch_size = 2
        vocab_size = 100

        tokens = mx.zeros((batch_size, 3), dtype=mx.int32)
        logits = mx.zeros((batch_size, vocab_size))
        sum_logprobs = mx.zeros((batch_size,))

        new_tokens, _, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens.shape == (batch_size, 4)

    def test_finalize_pads_eot(self, eot_token):
        decoder = GreedyDecoder(temperature=0, eot=eot_token)
        # tokens shape: (n_audio, 1, seq_len)
        tokens = mx.zeros((1, 1, 5), dtype=mx.int32)
        sum_logprobs = mx.zeros((1,))

        final_tokens, final_probs = decoder.finalize(tokens, sum_logprobs)
        # Should have one extra token (EOT padding)
        assert final_tokens.shape[-1] == 6
        assert final_tokens[0, 0, -1].item() == eot_token


class TestSuppressBlank:
    @pytest.fixture
    def tokenizer(self):
        return get_tokenizer(multilingual=True, language="en")

    def test_suppresses_at_sample_begin(self, tokenizer):
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tokenizer, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        # Tokens with exactly sample_begin length
        tokens = mx.zeros((1, sample_begin), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        # Space tokens and EOT should be suppressed (set to -inf)
        space_tokens = tokenizer.encode(" ")
        for st in space_tokens:
            assert result_np[st] == float("-inf")
        assert result_np[tokenizer.eot] == float("-inf")

    def test_no_suppression_after_sample_begin(self, tokenizer):
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tokenizer, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        # Tokens longer than sample_begin
        tokens = mx.zeros((1, sample_begin + 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        # Should be unchanged
        assert mx.all(result == logits).item()


class TestSuppressTokens:
    def test_suppresses_specified_tokens(self):
        n_vocab = 100
        suppress = [5, 10, 15]
        filt = SuppressTokens(suppress, n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        for idx in suppress:
            assert result_np[idx] == float("-inf")

        # Non-suppressed tokens should be unchanged
        assert result_np[0] == 0.0
        assert result_np[20] == 0.0

    def test_empty_suppress_list(self):
        n_vocab = 100
        filt = SuppressTokens([], n_vocab)

        logits = mx.ones((1, n_vocab))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        assert mx.all(result == logits).item()
