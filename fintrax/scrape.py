"""Discover and download Motley Fool earnings-call transcripts.

Transcript URLs come from two public sources: each ticker's quote page
(https://www.fool.com/quote/{exchange}/{ticker}/), which links that company's
recent calls, and fool.com's monthly sitemaps (https://www.fool.com/sitemap/YYYY/MM),
which robots.txt advertises. Pages are
fetched politely (fixed delay plus jitter) and cached to disk, so re-running
never refetches a page that is already saved.
"""
import json
import random
import re
import time
from datetime import date
from pathlib import Path

import requests

SITEMAP_URL = "https://www.fool.com/sitemap/{year}/{month:02d}"
QUOTE_URL = "https://www.fool.com/quote/{exchange}/{ticker}/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129 Safari/537.36"
    )
}
# e.g. /earnings/call-transcripts/2026/08/07/apple-aapl-q3-2026-earnings-call-transcript/
# Slugs vary: "-earnings-transcript", truncated "-transcrip", or no ticker at all.
URL_RE = re.compile(
    r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/"
    r"([a-z0-9-]+?)-(q[1-4])-(\d{4})-earnings[a-z-]*/?"
)


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def get(session: requests.Session, url: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code == 404:
                raise FileNotFoundError(url)
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"failed to fetch {url}")


def match_ticker(slug: str, tickers: set[str]) -> str | None:
    """The slug ends with the ticker, e.g. 'apple-aapl' -> 'AAPL'."""
    last = slug.rsplit("-", 1)[-1].upper()
    return last if last in tickers else None


def to_record(match: re.Match, ticker: str) -> dict:
    return {
        "ticker": ticker,
        "published": f"{match.group(1)}-{match.group(2)}-{match.group(3)}",
        "fiscal_quarter": f"{match.group(5).upper()} {match.group(6)}",
        "url": "https://www.fool.com" + match.group(0).rstrip("/") + "/",
    }


def quarter_key(fiscal_quarter: str) -> tuple[int, int]:
    q, year = fiscal_quarter.split()
    return int(year), int(q[1])


def cached_get(session: requests.Session, url: str, path: Path, refresh: bool, delay: float) -> str:
    if refresh or not path.exists():
        path.write_text(get(session, url))
        time.sleep(delay)
    return path.read_text()


def discover(cfg: dict, session: requests.Session, only: list[str] | None = None) -> list[dict]:
    """Return transcript records for configured tickers, newest fiscal quarter first."""
    tickers = set(only or cfg["tickers"])
    sc = cfg["scrape"]
    cache = cfg["paths"]["raw"] / "discovery"
    cache.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}

    # 1) Quote pages: every transcript link there belongs to that ticker.
    for ticker in sorted(tickers):
        path = cache / f"quote_{ticker}.html"
        if not path.exists():
            for exchange in ("nasdaq", "nyse"):
                try:
                    cached_get(session, QUOTE_URL.format(exchange=exchange, ticker=ticker.lower()),
                               path, True, sc["delay_seconds"])
                    break
                except FileNotFoundError:
                    continue
        if path.exists():
            for match in URL_RE.finditer(path.read_text()):
                rec = to_record(match, ticker)
                found[rec["url"]] = rec

    # 2) Monthly sitemaps fill gaps; the slug must end with the ticker here.
    current = date.today().strftime("%Y-%m")
    for y, m in months(sc["start_month"], sc["end_month"]):
        month = f"{y}-{m:02d}"
        # The current month's sitemap still grows, so don't trust a cached copy.
        xml = cached_get(session, SITEMAP_URL.format(year=y, month=m), cache / f"sitemap_{month}.xml",
                         month == current, sc["delay_seconds"])
        for match in URL_RE.finditer(xml):
            if ticker := match_ticker(match.group(4), tickers):
                rec = to_record(match, ticker)
                found.setdefault(rec["url"], rec)

    start = sc["start_month"] + "-01"
    # Fool republishes old calls under new URL dates, so rank by fiscal quarter
    # (comparable within a ticker); parse.py reads the true call date from the page.
    records = sorted((r for r in found.values() if r["published"] >= start),
                     key=lambda r: (r["ticker"], quarter_key(r["fiscal_quarter"])), reverse=True)
    # Keep the newest N calls per ticker; drop duplicate (ticker, quarter) URLs.
    kept, seen, per_ticker = [], set(), {}
    for r in records:
        key = (r["ticker"], r["fiscal_quarter"])
        if key in seen or per_ticker.get(r["ticker"], 0) >= sc["max_calls_per_ticker"]:
            continue
        seen.add(key)
        per_ticker[r["ticker"]] = per_ticker.get(r["ticker"], 0) + 1
        kept.append(r)
    return kept


def run(cfg: dict, only: list[str] | None = None) -> list[dict]:
    raw: Path = cfg["paths"]["raw"]
    session = requests.Session()
    records = discover(cfg, session, only)
    print(f"discovered {len(records)} transcripts")
    for i, r in enumerate(records, 1):
        r["file"] = f"{r['ticker']}_{r['fiscal_quarter'].replace(' ', '-')}.html"
        out = raw / r["file"]
        if out.exists():
            continue
        try:
            out.write_text(get(session, r["url"]))
            print(f"[{i}/{len(records)}] {r['ticker']} {r['fiscal_quarter']}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[{i}/{len(records)}] skip {e}")
        time.sleep(cfg["scrape"]["delay_seconds"] + random.uniform(0, 1.5))
    index_path = raw / "index.json"
    index = {r["url"]: r for r in json.loads(index_path.read_text())} if index_path.exists() else {}
    index.update({r["url"]: r for r in records if (raw / r["file"]).exists()})
    index_path.write_text(json.dumps(sorted(index.values(), key=lambda r: r["file"]), indent=1))
    return list(index.values())
