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
    todo = [t for t in tickers if t not in missing and not (cache / f"{t}.parquet").exists()]
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
        missing_file.write_text("\n".join(sorted(missing)))
        print(f"  {min(i + batch, len(todo))}/{len(todo)} done, {len(missing)} without data", flush=True)
        time.sleep(2)
    out = {}
    for t in tickers:
        path = cache / f"{t}.parquet"
        if path.exists():
            out[t] = pd.read_parquet(path)
    return out
