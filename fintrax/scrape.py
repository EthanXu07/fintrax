"""Discover and download every Motley Fool earnings-call transcript.

Discovery walks fool.com's public monthly sitemaps (https://www.fool.com/sitemap/YYYY/MM,
advertised in robots.txt) and collects every ``/earnings/call-transcripts/`` URL.
Old URLs are truncated ``.aspx`` slugs without a ticker, so the ticker, title and
fiscal quarter come from the page itself (``<meta name="primary_tickers">``).

Fetching is rate-limited across a few worker threads and resumable: each URL gets
one line in ``index.jsonl`` (ok or failed), and only the transcript body is kept,
gzipped (~15 KB instead of ~500 KB per page).
"""
import gzip
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.fool.com/sitemap/{year}/{month:02d}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129 Safari/537.36"
    )
}
TRANSCRIPT_URL = re.compile(r"https://www\.fool\.com/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/[^<\s\"]+")
QUARTER_RE = re.compile(r"\b(Q[1-4])\s+(?:FY\s*)?((?:19|20)\d{2})\b", re.I)


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


class RateLimiter:
    """At most `rate` requests per second across all threads."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next_at = max(self.next_at + self.interval, now)
            delay = self.next_at - now
        time.sleep(delay)

    def penalize(self, seconds: float) -> None:
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic()) + seconds


def get(session: requests.Session, url: str, limiter: RateLimiter, retries: int = 4) -> str:
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException:
            time.sleep(5 * 2 ** attempt)
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code in (404, 410):
            raise FileNotFoundError(url)
        if r.status_code in (403, 429, 503):
            # Being throttled: slow everyone down, not just this thread.
            print(f"throttled ({r.status_code}) on attempt {attempt + 1}: {url}", flush=True)
            limiter.penalize(30 * 2 ** attempt)
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}")


def discover(cfg: dict, session: requests.Session, limiter: RateLimiter) -> dict[str, str]:
    """All transcript URLs in the configured sitemap window -> publish date."""
    sc = cfg["scrape"]
    cache = cfg["paths"]["raw"] / "sitemaps"
    cache.mkdir(parents=True, exist_ok=True)
    end = sc.get("end_month") or date.today().strftime("%Y-%m")
    current = date.today().strftime("%Y-%m")
    urls: dict[str, str] = {}
    for y, m in months(sc["start_month"], end):
        month = f"{y}-{m:02d}"
        path = cache / f"{month}.xml"
        # The current month's sitemap still grows, so don't trust a cached copy.
        if not path.exists() or month == current:
            try:
                path.write_text(get(session, SITEMAP_URL.format(year=y, month=m), limiter))
            except FileNotFoundError:
                continue
        for match in TRANSCRIPT_URL.finditer(path.read_text()):
            urls.setdefault(match.group(0), f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
    return urls


def meta(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    return tag["content"].strip() if tag and tag.get("content") else ""


def extract(html: str) -> tuple[dict, str]:
    """Page metadata plus a minimal HTML document holding only the transcript."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("#article-body-transcript") or soup.select_one(".article-body")
    if body is None:
        raise ValueError("no transcript body")
    title = soup.title.get_text(" ", strip=True).removesuffix("| The Motley Fool").strip() if soup.title else ""
    quarter = QUARTER_RE.search(title)
    info = {
        "ticker": meta(soup, "primary_tickers").split(",")[0].strip().upper(),
        "company": meta(soup, "primary_tickers_companies").split(",")[0].strip(),
        "title": title,
        "fiscal_quarter": f"{quarter.group(1).upper()} {quarter.group(2)}" if quarter else "",
    }
    return info, f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def page_path(raw: Path, url: str) -> Path:
    h = hashlib.sha1(url.encode()).hexdigest()
    return raw / "pages" / h[:2] / f"{h}.html.gz"


def load_index(raw: Path) -> dict[str, dict]:
    """Latest record per URL (a retried URL appears more than once)."""
    path = raw / "index.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        return {(r := json.loads(line))["url"]: r for line in f if line.strip()}


def run(cfg: dict, only: list[str] | None = None) -> None:
    raw: Path = cfg["paths"]["raw"]
    sc = cfg["scrape"]
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=sc["workers"]))
    limiter = RateLimiter(sc["requests_per_second"])

    urls = discover(cfg, session, limiter)
    done = load_index(raw)
    # Retry earlier transient failures; 404s and pages without a transcript are final.
    todo = [u for u in sorted(urls, key=urls.get, reverse=True)
            if u not in done or done[u]["status"] == "error"]
    print(f"{len(urls)} transcript URLs in sitemaps; {len(done)} already indexed; {len(todo)} to fetch",
          flush=True)

    lock = threading.Lock()
    counts = {"ok": 0, "missing": 0, "error": 0, "skipped": 0}
    started = time.monotonic()

    def fetch(url: str) -> None:
        rec = {"url": url, "published": urls[url]}
        try:
            info, doc = extract(get(session, url, limiter))
            if only and info["ticker"] not in only:
                rec["status"] = "skipped"
            else:
                path = page_path(raw, url)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(doc.encode()))
                rec.update(info, status="ok", file=str(path.relative_to(raw)))
        except (FileNotFoundError, ValueError) as e:
            rec.update(status="missing", error=type(e).__name__)
        except RuntimeError as e:
            rec.update(status="error", error=str(e))
        with lock:
            if rec["status"] != "skipped":
                index_file.write(json.dumps(rec) + "\n")
                index_file.flush()
            counts[rec["status"]] += 1
            n = sum(counts.values())
            if n % 100 == 0 or n == len(todo):
                rate = n / (time.monotonic() - started)
                eta = (len(todo) - n) / rate / 3600
                print(f"[{n}/{len(todo)}] {counts} {rate:.2f} pages/s, ~{eta:.1f} h left", flush=True)

    with open(raw / "index.jsonl", "a") as index_file, ThreadPoolExecutor(sc["workers"]) as pool:
        list(pool.map(fetch, todo))
    print(f"done: {counts}")
