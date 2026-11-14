"""Model-free checks of the removed-token type rules (experiment 03)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from k_sweep import token_types  # noqa: E402


def flagged(toks, kind):
    f = token_types(toks)[kind]
    return [t for t, x in zip(toks, f) if x]


# "We fine-tune (BERT) on the LLM data. Results in 2024 show gains."
TOKS = ["▁We", "▁fine", "-", "t", "une", "▁(", "BER", "T", ")", "▁on", "▁the", "▁LL", "M", "▁data", ".",
        "▁Results", "▁in", "▁", "2", "0", "2", "4", "▁show", "▁gains", "."]


def test_capitalized_skips_sentence_starts_and_finds_names_after_symbols():
    assert flagged(TOKS, "capitalized") == ["BER", "▁LL"]


def test_word_pieces_are_letters_after_the_first_letter_token():
    assert flagged(TOKS, "word_piece") == ["t", "une", "T", "M"]


def test_digits_symbols_function_words():
    assert flagged(TOKS, "digit") == ["2", "0", "2", "4"]
    assert flagged(TOKS, "symbol") == ["-", "▁(", ")", ".", "▁", "."]
    assert flagged(TOKS, "function_word") == ["▁We", "▁on", "▁the", "▁in"]


def test_llm_terms_cover_every_token_of_the_word():
    assert flagged(TOKS, "llm_term") == ["▁fine", "t", "une", "▁LL", "M"]  # fine-tune is on the list
    assert flagged(["▁GPT", "-", "4", "o", "▁models"], "llm_term") == ["▁GPT", "4", "o"]
    assert flagged(["▁fine", "-", "tun", "ing"], "llm_term") == ["▁fine", "tun", "ing"]


def test_newline_byte_starts_a_sentence():
    toks = ["▁see", "<0x0A>", "Results", "▁hold"]
    assert flagged(toks, "capitalized") == []
