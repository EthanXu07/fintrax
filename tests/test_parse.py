from pathlib import Path

import pytest

from fintrax.parse import name_key, parse_html

FIX = Path(__file__).parent / "fixtures"


def turns(name: str) -> list[dict]:
    return parse_html((FIX / name).read_text())


def test_current_layout_splits_at_first_question():
    t = turns("current_layout.html")
    assert [x["section"] for x in t] == ["prepared"] * 3 + ["qna"] * 4
    assert t[3]["speaker"] == "Operator" and "first question" in t[3]["text"]


def test_current_layout_roles_and_continuation_paragraphs():
    t = turns("current_layout.html")
    jane = t[1]
    assert jane["speaker_type"] == "executive" and jane["role"] == "Chief Executive Officer"
    assert "record revenue" in jane["text"]          # second <p> joined to the same turn
    assert t[4]["speaker"] == "Sam Analyst" and t[4]["speaker_type"] == "analyst"
    assert t[5]["speaker_type"] == "executive"       # "John Roe" matched with " - " separator
    assert not any("Promo" in x["text"] for x in t)  # stops at the next <h2>


def test_legacy_layout_uses_headers():
    t = turns("legacy_layout.html")
    assert {x["section"] for x in t} == {"prepared", "qna"}
    pat = [x for x in t if x["speaker"] == "Pat Exec"]
    assert [x["section"] for x in pat] == ["prepared", "qna"]
    assert all(x["speaker_type"] == "executive" for x in pat)
    assert any(x["speaker_type"] == "analyst" for x in t)
    assert not any(x["speaker"].startswith("Duration") for x in t)
    assert not any(x["text"].startswith("[Operator") and x["text"].endswith("]") for x in t)


@pytest.mark.parametrize("a,b", [("Amy E. Hood", "Amy Hood"), ("Timothy D. Cook", "Timothy Cook")])
def test_name_key_ignores_middle_initials(a, b):
    assert name_key(a) == name_key(b)


def test_call_date_comes_from_page_not_url():
    from fintrax.parse import parse_page
    (date, time), _ = parse_page((FIX / "current_layout.html").read_text())
    assert (date, time) == ("2026-07-30", "17:00")


def test_lookup_role_falls_back_to_unique_last_name():
    from fintrax.parse import lookup_role
    participants = {name_key("Jensen Huang"): "CEO", name_key("Colette Kress"): "CFO"}
    assert lookup_role(participants, "Jen-Hsun Huang") == "CEO"
    assert lookup_role(participants, "Joseph Moore") is None
