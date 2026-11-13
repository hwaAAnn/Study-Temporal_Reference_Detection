"""Model-free checks of the text clean-up used for polished abstracts (experiment 05)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from polish_texts import clean_polish, edit_sim  # noqa: E402

BODY = "We study graphs. Our method improves accuracy by 3%."


def test_clean_polish_drops_lead_in_and_change_notes():
    raw = f"Here's a polished abstract:\n\n{BODY}\n\nChanges made:\n- fixed grammar\n- clarified the claim"
    assert clean_polish(raw) == BODY


def test_clean_polish_inline_lead_and_header():
    assert clean_polish(f"Here is the corrected abstract: {BODY}") == BODY
    assert clean_polish(f"**Polished Abstract:**\n{BODY}") == BODY
    assert clean_polish(f"Abstract: {BODY}") == BODY


def test_clean_polish_keeps_multi_paragraph_text_and_strips_note():
    raw = f"{BODY}\n\nWe also release code.\n\nNote: I kept the technical terms unchanged."
    assert clean_polish(raw) == f"{BODY} We also release code."
    assert clean_polish(f"{BODY} (Note: minor edits only.)") == BODY


def test_edit_sim():
    assert edit_sim(BODY, BODY) == 1.0
    assert edit_sim(BODY, BODY.upper()) == 1.0  # case-insensitive
    assert 0.0 < edit_sim(BODY, "We study trees. Our method improves accuracy by 3%.") < 1.0
