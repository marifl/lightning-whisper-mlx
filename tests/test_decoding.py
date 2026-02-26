import mlx.core as mx
import numpy as np

from lightning_whisper_mlx.decoding import (
    GreedyDecoder,
    SuppressBlank,
    SuppressTokens,
    compression_ratio,
)
from lightning_whisper_mlx.tokenizer import get_tokenizer


class TestCompressionRatio:
    def test_known_value(self):
        """compression_ratio('hello world') must equal the pre-computed value."""
        assert compression_ratio("hello world") == 11 / 19

    def test_repetitive_high_ratio(self):
        """Highly repetitive text must compress well (ratio > 5)."""
        assert compression_ratio("aaa" * 100) > 5.0

    def test_empty_string_does_not_crash(self):
        """Empty string must not raise."""
        ratio = compression_ratio("")
        assert ratio >= 0


class TestGreedyDecoder:
    EOT = 50257

    def test_argmax_at_temperature_zero(self):
        """Temperature=0 must pick the argmax token."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        logits = mx.zeros((1, 100))
        logits = logits.at[:, 42].add(10.0)
        tokens = mx.zeros((1, 1), dtype=mx.int32)
        sum_logprobs = mx.zeros((1,))

        new_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens[0, -1].item() == 42
        assert not completed.item()

    def test_eot_propagation(self):
        """Once EOT is emitted, all subsequent tokens must be EOT."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        vocab_size = self.EOT + 10
        tokens = mx.array([[self.EOT]], dtype=mx.int32)
        logits = mx.zeros((1, vocab_size))
        logits = logits.at[:, 5].add(10.0)
        sum_logprobs = mx.zeros((1,))

        new_tokens, completed, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens[0, -1].item() == self.EOT
        assert completed.item()

    def test_tokens_grow_by_one(self):
        """Each update step must append exactly one token."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        tokens = mx.zeros((2, 3), dtype=mx.int32)
        logits = mx.zeros((2, 100))
        sum_logprobs = mx.zeros((2,))

        new_tokens, _, _ = decoder.update(tokens, logits, sum_logprobs)
        assert new_tokens.shape == (2, 4)

    def test_sum_logprobs_accumulates(self):
        """sum_logprobs must increase (become less negative) by the selected token's log-prob."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        logits = mx.zeros((1, 100))
        logits = logits.at[:, 0].add(10.0)
        tokens = mx.zeros((1, 1), dtype=mx.int32)
        sum_logprobs_before = mx.zeros((1,))

        _, _, sum_logprobs_after = decoder.update(tokens, logits, sum_logprobs_before)
        assert sum_logprobs_after[0].item() != 0.0
        assert sum_logprobs_after[0].item() < 0.0

    def test_finalize_appends_eot(self):
        """finalize must append EOT token at the end."""
        decoder = GreedyDecoder(temperature=0, eot=self.EOT)
        tokens = mx.zeros((1, 1, 5), dtype=mx.int32)
        sum_logprobs = mx.zeros((1,))

        final_tokens, _ = decoder.finalize(tokens, sum_logprobs)
        assert final_tokens.shape[-1] == 6
        assert final_tokens[0, 0, -1].item() == self.EOT


class TestSuppressBlank:
    def test_suppresses_blank_and_eot_at_sample_begin(self):
        """At sample_begin position, space tokens and EOT must be set to -inf."""
        tok = get_tokenizer(multilingual=True, language="en")
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tok, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, sample_begin), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        space_tokens = tok.encode(" ")
        for st in space_tokens:
            assert result_np[st] == float("-inf")
        assert result_np[tok.eot] == float("-inf")

    def test_no_suppression_after_sample_begin(self):
        """After sample_begin, logits must pass through unchanged."""
        tok = get_tokenizer(multilingual=True, language="en")
        n_vocab = 51865
        sample_begin = 3
        filt = SuppressBlank(tok, sample_begin, n_vocab)

        logits = mx.zeros((1, n_vocab))
        tokens = mx.zeros((1, sample_begin + 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        assert mx.all(result == logits).item()


class TestSuppressTokens:
    def test_suppresses_exact_tokens(self):
        """Specified tokens must be -inf, all others unchanged."""
        suppress = [5, 10, 15]
        filt = SuppressTokens(suppress, 100)

        logits = mx.ones((1, 100))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        result_np = np.array(result[0])

        for idx in suppress:
            assert result_np[idx] == float("-inf")
        assert result_np[0] == 1.0
        assert result_np[20] == 1.0

    def test_empty_suppress_list_is_identity(self):
        """Empty suppress list must not change any logits."""
        filt = SuppressTokens([], 100)
        logits = mx.ones((1, 100))
        tokens = mx.zeros((1, 1), dtype=mx.int32)

        result = filt.apply(logits, tokens)
        assert mx.all(result == logits).item()
