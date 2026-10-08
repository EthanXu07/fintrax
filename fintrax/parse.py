"""Clean Motley Fool transcript HTML and split it into speaker turns.

Two page layouts exist:

* legacy (through ~2025): ``<h2>Prepared Remarks:</h2>`` / ``<h2>Questions & Answers:</h2>``
  headers, speaker lines ``<p><strong>Name</strong> -- <em>Role</em></p>`` followed by
  text paragraphs, and a trailing "Call participants" list.
* current: a "CALL PARTICIPANTS" bullet list (``Role - Name``) up top and a
  "Full Conference Call Transcript" section of ``<p><strong>Name:</strong> text</p>``
  turns with no Q&A header, so the Q&A boundary is inferred.

Each turn becomes a row: ticker, call_date, section (prepared|qna), speaker,
speaker_type (executive|analyst|operator), role, text.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

QNA_CUE = re.compile(
    r"question-and-answer|question and answer|q&a|first question|open (?:up )?the (?:call|line|floor) (?:up )?(?:for|to) questions|"
    r"take (?:your|our) (?:first )?questions",
    re.I,
)
PARTICIPANT_SEP = re.compile(r"\s+[-–—]+\s+")
NOISE = re.compile(r"^\s*(\[.*?\]|duration:.*)\s*$", re.I)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def name_key(name: str) -> str:
    """Normalize names so 'Amy E. Hood' and 'Amy Hood' match."""
    parts = re.sub(r"[^a-z ]", "", name.lower()).split()
    return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else " ".join(parts)


def speaker_type(name: str, role: str) -> str:
    if name.lower().startswith("operator"):
        return "operator"
    return "analyst" if "analyst" in role.lower() else "executive"


def article_body(soup: BeautifulSoup) -> Tag:
    body = soup.select_one("#article-body-transcript") or soup.select_one(".article-body")
    if body is None:
        raise ValueError("no article body")
    return body


def parse_legacy(body: Tag) -> list[dict]:
    turns, section, current = [], None, None
    for el in body.find_all(["h2", "p"]):
        if el.name == "h2":
            heading = clean(el.get_text()).lower()
            if heading.startswith("prepared remarks"):
                section = "prepared"
            elif heading.startswith("questions"):
                section = "qna"
            elif heading.startswith("call participants"):
                break
            continue
        if section is None:
            continue
        strong = el.find("strong")
        text = clean(el.get_text(" "))
        # Speaker line: the paragraph is just "<strong>Name</strong> [-- <em>Role</em>]".
        if strong and clean(strong.get_text()) and text.startswith(clean(strong.get_text())) and (
            " -- " in text or text == clean(strong.get_text())
        ):
            name = clean(strong.get_text())
            if name.lower().startswith("duration"):
                continue
            em = el.find("em")
            role = clean(em.get_text()) if em else ("Operator" if name.lower() == "operator" else "")
            current = {"section": section, "speaker": name, "role": role, "paras": []}
            turns.append(current)
        elif current is not None and text and not NOISE.match(text):
            current["paras"].append(text)
    return turns


def parse_current(body: Tag) -> list[dict]:
    # Participant entries are "Role - Name" (or em/en dash), as <li>s or <p>s.
    participants: dict[str, str] = {}
    h = body.find("h2", id="call-participants")
    for sib in h.find_next_siblings() if h else []:
        if sib.name == "h2":
            break
        for item in sib.find_all("li") if sib.name == "ul" else [sib] if sib.name == "p" else []:
            parts = PARTICIPANT_SEP.split(clean(item.get_text()))
            if len(parts) >= 2:
                participants[name_key(parts[-1])] = " ".join(parts[:-1])
    start = body.find("h2", id="full-conference-call-transcript")
    if start is None:
        raise ValueError("no transcript section")
    turns, current, participants_fallback = [], None, {}
    for el in start.find_all_next(["p", "h2"]):
        if el.name == "h2" or not body in el.parents:
            break
        text = clean(el.get_text(" "))
        strong = el.find("strong")
        label = clean(strong.get_text()) if strong else ""
        if label.endswith(":") and text.startswith(label):
            name = label[:-1].strip()
            role = participants.get(name_key(name))
            if role is None:
                role = "Operator" if name.lower().startswith("operator") else "Analyst"
                # No participant list: anyone who spoke before the first Q&A cue is management.
                if role == "Analyst" and not participants and not any(
                    QNA_CUE.search(" ".join(t["paras"])) for t in turns
                ):
                    role = "Executive"
                if not participants and role != "Operator":
                    participants_fallback.setdefault(name_key(name), role)
                    role = participants_fallback[name_key(name)]
            current = {"speaker": name, "role": role, "paras": []}
            turns.append(current)
            text = text[len(label):].strip()
        if current is not None and text and not NOISE.match(text):
            current["paras"].append(text)
    # Q&A begins at the first analyst turn; pull back one turn if the operator
    # (or an executive) introduced it with a question cue.
    first_analyst = next(
        (i for i, t in enumerate(turns) if speaker_type(t["speaker"], t["role"]) == "analyst"), None
    )
    if first_analyst is None:
        boundary = next(
            (i for i, t in enumerate(turns) if QNA_CUE.search(" ".join(t["paras"]))), len(turns)
        )
    else:
        boundary = first_analyst
        if boundary > 0 and speaker_type(turns[boundary - 1]["speaker"], turns[boundary - 1]["role"]) == "operator":
            boundary -= 1
    for i, t in enumerate(turns):
        t["section"] = "prepared" if i < boundary else "qna"
    return turns


def parse_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    body = article_body(soup)
    is_legacy = any(clean(h.get_text()).lower().startswith("prepared remarks") for h in body.find_all("h2"))
    turns = parse_legacy(body) if is_legacy else parse_current(body)
    out = []
    for t in turns:
        text = " ".join(t["paras"]).strip()
        if text:
            out.append(
                {
                    "section": t["section"],
                    "speaker": t["speaker"],
                    "speaker_type": speaker_type(t["speaker"], t["role"]),
                    "role": t["role"],
                    "text": text,
                    "layout": "legacy" if is_legacy else "current",
                }
            )
    return out


def run(cfg: dict) -> pd.DataFrame:
    raw: Path = cfg["paths"]["raw"]
    index = json.loads((raw / "index.json").read_text())
    rows, stats = [], []
    for rec in index:
        try:
            turns = parse_html((raw / rec["file"]).read_text())
        except (ValueError, FileNotFoundError) as e:
            print(f"parse failed {rec['file']}: {e}")
            continue
        sections = {t["section"] for t in turns}
        stats.append({"file": rec["file"], "ok": sections == {"prepared", "qna"}, "turns": len(turns)})
        for t in turns:
            rows.append({"ticker": rec["ticker"], "call_date": rec["call_date"],
                         "fiscal_quarter": rec["fiscal_quarter"], **t})
    df = pd.DataFrame(rows)
    out = cfg["paths"]["interim"] / "turns.parquet"
    df.to_parquet(out, index=False)
    st = pd.DataFrame(stats)
    print(f"{len(st)} calls parsed, {st['ok'].mean():.1%} with both sections, {len(df)} turns -> {out}")
    if (~st["ok"]).any():
        print("missing a section:", st.loc[~st["ok"], "file"].tolist())
    return df
