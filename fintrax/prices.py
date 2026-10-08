"""Daily prices from Yahoo Finance (via yfinance), cached per ticker.

Tickers Yahoo doesn't know (delisted, renamed, foreign listings) are remembered in
``_missing.txt`` so reruns don't ask again. Delisted names dropping out is a
survivorship bias the README calls out.
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

COLS = ["Open", "Close", "Volume"]


def yahoo_symbol(ticker: str) -> str:
    """Fool writes share classes with a dot (BRK.B); Yahoo uses a dash (BRK-B)."""
    return ticker.replace(".", "-").replace("/", "-")


def load(tickers: list[str], start: str, cache: Path, batch: int = 100) -> dict[str, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    missing_file = cache / "_missing.txt"
    missing = set(missing_file.read_text().split()) if missing_file.exists() else set()
    later_file = cache / "_listed_later.txt"
    listed_later = set(later_file.read_text().split()) if later_file.exists() else set()

    def covered(t: str) -> bool:
        """Cached and starting early enough (a cache built for a later window is refetched)."""
        path = cache / f"{t}.parquet"
        if not path.exists():
            return False
        first = pd.read_parquet(path, columns=["Close"]).index.min()
        return t in listed_later or first <= pd.Timestamp(start) + pd.Timedelta(days=10)

    todo = [t for t in tickers if t not in missing and not covered(t)]
    if todo:
        print(f"downloading prices for {len(todo)} tickers ({len(tickers) - len(todo)} cached or known missing)")
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        symbols = {yahoo_symbol(t): t for t in chunk}
        for attempt in range(3):
            try:
                data = yf.download(list(symbols), start=start, auto_adjust=True, progress=False,
                                   group_by="ticker", threads=True)
                break
            except Exception as e:  # yfinance raises assorted network/rate-limit errors
                print(f"  batch {i // batch} failed ({e}); retrying")
                time.sleep(30 * (attempt + 1))
        else:
            continue
        got = set(data.columns.get_level_values(0)) if not data.empty else set()
        for symbol, ticker in symbols.items():
            px = data[symbol][COLS].dropna() if symbol in got else pd.DataFrame()
            if px.empty:
                missing.add(ticker)
            else:
                px.to_parquet(cache / f"{ticker}.parquet")
                if px.index.min() > pd.Timestamp(start) + pd.Timedelta(days=10):
                    listed_later.add(ticker)  # IPO'd after `start`: the history really starts later
        missing_file.write_text("\n".join(sorted(missing)))
        later_file.write_text("\n".join(sorted(listed_later)))
        print(f"  {min(i + batch, len(todo))}/{len(todo)} done, {len(missing)} without data", flush=True)
        time.sleep(2)
    out = {}
    for t in tickers:
        path = cache / f"{t}.parquet"
        if path.exists():
            out[t] = pd.read_parquet(path)
    return out
