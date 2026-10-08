import pytest

from fintrax import lexicon
from fintrax.dataset import split_sentences, to_sentences

pytest.importorskip("requests")


@pytest.mark.parametrize("sentence,label", [
    ("We are very confident we will deliver double-digit growth this year.", "high"),
    ("It is probably too early to say, and demand could soften depending on the macro.", "low"),
    ("Revenue was $94 billion, up 10% year over year.", "neutral"),
    ("We have nearly 600,000 customers on the platform today.", "neutral"),
])
def test_weak_label(sentence, label):
    assert lexicon.weak_label(sentence) == label


def test_single_soft_hedge_is_dropped_not_low():
    assert lexicon.weak_label("I think margins were solid in the quarter overall.") is None


def test_mask_cues_removes_every_cue():
    masked = lexicon.mask_cues("We think demand may soften, but we are confident.", "[MASK]")
    assert "think" not in masked and "may" not in masked and "confident" not in masked
    assert "demand" in masked


def test_sentences_drop_boilerplate_and_analysts():
    import pandas as pd
    turns = pd.DataFrame([
        {"ticker": "X", "call_date": "2026-01-01", "section": "prepared", "speaker": "A", "role": "CEO",
         "speaker_type": "executive",
         "text": "These forward-looking statements involve risk. Our pipeline has never been stronger today."},
        {"ticker": "X", "call_date": "2026-01-01", "section": "qna", "speaker": "B", "role": "Analyst",
         "speaker_type": "analyst", "text": "Can you talk about your margin outlook for next year please?"},
    ])
    s = to_sentences(turns, min_tokens=6)
    assert s["sentence"].tolist() == ["Our pipeline has never been stronger today."]


def test_split_sentences_strips_bracket_noise():
    assert split_sentences("[Operator instructions] First one. Second one.") == ["First one.", "Second one."]
