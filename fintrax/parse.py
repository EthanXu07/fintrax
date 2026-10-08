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
import gzip
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag

# Phrases that mark the hand-off to Q&A. Deliberately strict: an opening line like
# "after the prepared remarks we'll open the call to questions" must not match.
QNA_CUE = re.compile(
    r"first question|next question|question-and-answer session|question and answer session|"
    r"q(?:&| and |-and-)a session|poll (?:the audience )?for questions|poll the audience|"
    r"(?:move|turn|go|get) (?:over |on )?(?:to |into )?(?:the )?(?:investor |analyst )?(?:q(?:&| and )a\b|questions)|"
    r"(?:start|begin|open) (?:our |the )?(?:analyst |investor )?(?:q(?:&| and )a\b|questions from)|"
    r"\[operator instructions\]",
    re.I,
)
HANDOFF_WORDS = 80  # longer turns carry real remarks, not just "first question, please"
MONOLOGUE_WORDS = 300  # unlisted speakers this long are management, not analysts
FIRST_QUESTION = re.compile(r"first question|question-and-answer session|question and answer session", re.I)
PARTICIPANT_SEP = re.compile(r"\s+[-–—]+\s+")
NOISE = re.compile(r"^\s*(\[.*?\]|duration:.*)\s*$", re.I)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def name_key(name: str) -> str:
    """Normalize names so 'Amy E. Hood' and 'Amy Hood' match."""
    parts = re.sub(r"[^a-z ]", "", name.lower()).split()
    return f"{parts[0]} {parts[-1]}" if len(parts) >= 2 else " ".join(parts)


def looks_like_name(label: str) -> bool:
    """'Jane Doe' or 'Analyst (Sam Lee)' -- not a bolded sentence like 'In APLA, revenue was down 2%.'"""
    return 0 < len(label.split()) <= 7 and not re.search(r"\d|[.!?]\s", label)


def speaker_type(name: str, role: str) -> str:
    if name.lower().startswith("operator"):
        return "operator"
    return "analyst" if "analyst" in role.lower() else "executive"


DATE_RE = re.compile(r"([A-Z][a-z]{2,8})\.? (\d{1,2}), (\d{4})")
TIME_RE = re.compile(r"(\d{1,2})(?::(\d{2}))? ?([ap])\.?m", re.I)


def call_datetime(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """Actual call date (YYYY-MM-DD) and 24h ET time from the page header.

    Fool's URL date is the publish date, which lags the call by days and,
    for republished transcripts, by years.
    """
    node = soup.select_one("#date")
    if node is not None:
        text = clean((node.find_next_sibling() or node).get_text(" ") if node.name == "h2" else node.get_text(" "))
        if node.name == "span" and (em := soup.select_one("#time")):
            text += " " + clean(em.get_text())
    else:
        # 2017-2018 pages: a header paragraph "Company (TKR) / Q4 2017 Earnings Call / Feb. 8, 2018, 9:00 a.m. ET".
        header = next((p for p in soup.find_all("p", limit=6)
                       if "call" in p.get_text().lower() and DATE_RE.search(p.get_text(" "))), None)
        if header is None:
            return None, None
        text = clean(header.get_text(" "))
    m = DATE_RE.search(text)
    if not m:
        return None, None
    month = m.group(1)[:3]
    date = pd.to_datetime(f"{month} {m.group(2)} {m.group(3)}", format="%b %d %Y").strftime("%Y-%m-%d")
    t = TIME_RE.search(text)
    time = None
    if t:
        hour = int(t.group(1)) % 12 + (12 if t.group(3).lower() == "p" else 0)
        time = f"{hour:02d}:{t.group(2) or '00'}"
    return date, time


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


def lookup_role(participants: dict[str, str], name: str) -> str | None:
    """Exact first+last match, else a unique last-name match ('Jen-Hsun Huang' -> 'Jensen Huang')."""
    key = name_key(name)
    if key in participants:
        return participants[key]
    last = key.split()[-1] if key else ""
    hits = [role for k, role in participants.items() if k.split()[-1:] == [last]]
    return hits[0] if len(hits) == 1 else None


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
    turns, current = [], None
    for el in start.find_all_next(["p", "h2"]):
        if el.name == "h2" or not body in el.parents:
            break
        text = clean(el.get_text(" "))
        strong = el.find("strong")
        label = clean(strong.get_text()) if strong else ""
        if label.endswith(":") and text.startswith(label) and looks_like_name(label[:-1]):
            name = label[:-1].strip()
            role = lookup_role(participants, name) or (
                "Operator" if name.lower().startswith("operator") else "Analyst")
            current = {"speaker": name, "role": role, "paras": []}
            turns.append(current)
            text = text[len(label):].strip()
        if current is not None and text and not NOISE.match(text):
            current["paras"].append(text)
    # Analysts don't deliver 300-word monologues before the first question;
    # unlisted speakers who do are management the participant list left out.
    first_q = next((i for i, t in enumerate(turns) if FIRST_QUESTION.search(" ".join(t["paras"]))), len(turns))
    unlisted_mgmt = {t["speaker"] for t in turns[:first_q]
                     if t["role"] == "Analyst" and len(" ".join(t["paras"]).split()) >= MONOLOGUE_WORDS}
    for t in turns:
        if t["speaker"] in unlisted_mgmt:
            t["role"] = "Executive (unlisted)"
    return split_qna(turns)


def split_qna(turns: list[dict]) -> list[dict]:
    """Label turns prepared/qna when the page has no Q&A header.

    Q&A starts at the first turn that hands off to questions (operator
    instructions, "first question", ...); without such a cue, at the first
    analyst turn. Unlisted speakers before the boundary (usually an
    investor-relations host the participant list left out) are management.
    """
    # Ignore cues before management has said anything substantial: operator
    # welcomes often contain "[Operator Instructions]" before any remarks.
    words = [len(" ".join(t["paras"]).split()) for t in turns]
    first_remarks = next((i for i, t in enumerate(turns)
                          if speaker_type(t["speaker"], t["role"]) != "operator" and words[i] >= 150), 0)
    kinds = [speaker_type(t["speaker"], t["role"]) for t in turns]
    cue = next((i for i, t in enumerate(turns)
                if i > first_remarks and QNA_CUE.search(" ".join(t["paras"]))), None)
    # The cue often closes a long prepared remark ("With that, let's go to Q&A"),
    # so Q&A starts at the short hand-off turn right before the next analyst.
    analyst = next((i for i in range(cue or 0, len(turns)) if kinds[i] == "analyst"), None)
    if cue is None:
        cue = len(turns) if analyst is None else analyst
    elif analyst is not None:
        cue = analyst - 1 if analyst > cue and words[analyst - 1] < HANDOFF_WORDS else analyst
    elif words[cue] >= HANDOFF_WORDS:
        cue += 1
    hosts = {name_key(t["speaker"]) for t, k in zip(turns[:cue], kinds) if k == "analyst"}
    for i, t in enumerate(turns):
        t["section"] = "prepared" if i < cue else "qna"
        if name_key(t["speaker"]) in hosts:
            t["role"] = "Investor Relations (unlisted)"
    return turns


def parse_html(html: str) -> list[dict]:
    return parse_page(html)[1]


def parse_page(html: str) -> tuple[tuple[str | None, str | None], list[dict]]:
    """((call_date, call_time_et), turns)"""
    soup = BeautifulSoup(html, "lxml")
    body = article_body(soup)
    is_legacy = any(clean(h.get_text()).lower().startswith("prepared remarks") for h in body.find_all("h2"))
    turns = parse_legacy(body) if is_legacy else parse_current(body)
    if is_legacy and turns and all(t["section"] == "prepared" for t in turns):
        turns = split_qna(turns)
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
    return call_datetime(soup), out


def parse_record(args: tuple[dict, str, int]) -> tuple[dict, list[dict]]:
    """Parse one stored page -> (call stats, turn rows). Runs in a worker process."""
    rec, raw, min_words = args
    stat = {"url": rec["url"], "ticker": rec.get("ticker", ""), "published": rec["published"],
            "call_date": None, "ok": False, "both_sections": False}
    try:
        html = gzip.decompress((Path(raw) / rec["file"]).read_bytes()).decode()
        (call_date, call_time), turns = parse_page(html)
    except (ValueError, OSError, AttributeError) as e:
        stat["error"] = type(e).__name__
        return stat, []
    words = {sec: sum(len(t["text"].split()) for t in turns
                      if t["section"] == sec and t["speaker_type"] == "executive")
             for sec in ("prepared", "qna")}
    stat.update(call_date=call_date, prepared_words=words["prepared"], qna_words=words["qna"],
                both_sections=min(words.values()) > 0,
                ok=bool(call_date and stat["ticker"]) and min(words.values()) >= min_words)
    if not stat["ok"]:
        return stat, []
    base = {"ticker": stat["ticker"], "company": rec.get("company", ""), "call_date": call_date,
            "call_time_et": call_time, "fiscal_quarter": rec.get("fiscal_quarter", ""), "url": rec["url"]}
    return stat, [{**base, **t} for t in turns]


def run(cfg: dict) -> None:
    from fintrax.scrape import load_index

    raw: Path = cfg["paths"]["raw"]
    out_dir = cfg["paths"]["interim"] / "turns"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    records = [r for r in load_index(raw).values() if r["status"] == "ok"]
    # Oldest publish date first, so a republished copy loses to the original below.
    records.sort(key=lambda r: r["published"])
    min_words = cfg["parse"]["min_section_words"]

    stats, seen, part, buffer = [], set(), 0, []
    with ProcessPoolExecutor(cfg["parse"]["workers"]) as pool:
        for stat, rows in pool.map(parse_record, [(r, str(raw), min_words) for r in records], chunksize=32):
            key = (stat["ticker"], stat["call_date"])
            # Republished transcripts appear under several URLs; keep one per actual call.
            if stat["ok"] and key in seen:
                stat.update(ok=False, error="duplicate")
                rows = []
            seen.add(key)
            stats.append(stat)
            buffer.extend(rows)
            if len(buffer) >= 200_000:
                pd.DataFrame(buffer).to_parquet(out_dir / f"part-{part:04d}.parquet", index=False)
                part, buffer = part + 1, []
    if buffer:
        pd.DataFrame(buffer).to_parquet(out_dir / f"part-{part:04d}.parquet", index=False)

    st = pd.DataFrame(stats)
    st.to_parquet(cfg["paths"]["interim"] / "parse_stats.parquet", index=False)
    print(f"{len(st)} pages parsed, {st['both_sections'].mean():.1%} with both sections; "
          f"kept {int(st['ok'].sum())} calls from {st.loc[st['ok'], 'ticker'].nunique()} tickers -> {out_dir}")
    if "error" not in st:
        st["error"] = None
    reasons = st.loc[~st["ok"], "error"].fillna("short section or no date/ticker")
    print("dropped:", reasons.value_counts().to_dict())


def read_turns(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["interim"] / "turns", columns=columns)
