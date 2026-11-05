"""backtest - checks how the signals did after the fact. not part of `all`

entry = first open after the call (same day if premarket)
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from fintrax import charts, inventory, prices

ORDER = ["BUY", "HOLD", "SELL"]
FEATURES = ["confidence", "conf_prepared", "conf_qna", "gap"]


def entry_index(px, call_date, premarket):
    day = pd.Timestamp(call_date)
    i = px.index.searchsorted(day, side="left" if premarket else "right")
    return i if i < len(px) else None


def event_returns(sig, px, bench, horizons, bt):
    rows = []
    for r in sig.itertuples():
        p = px.get(r.ticker)
        if p is None or len(p) < 30:
            continue
        premarket = isinstance(r.call_time_et, str) and "" < r.call_time_et < "09:30"
        i = entry_index(p, r.call_date, premarket)
        if i is None or i < 20:
            continue
        entry_day, entry_px = p.index[i], p["Open"].iloc[i]
        dollar_vol = (p["Close"].iloc[i - 20:i] * p["Volume"].iloc[i - 20:i]).mean()
        # skip penny stocks / illiquid stuff, they wreck the averages
        if entry_px < bt["min_price"] or dollar_vol < bt["min_dollar_volume"]:
            continue
        j = bench.index.searchsorted(entry_day)
        if j >= len(bench) or bench.index[j] != entry_day:
            continue
        row = {"ticker": r.ticker, "company": r.company, "call_date": r.call_date, "entry_date": entry_day,
               "signal": r.signal, "entry_price": entry_px, "dollar_volume": dollar_vol,
               **{f: getattr(r, f) for f in FEATURES}}
        for h in horizons:
            if i + h - 1 < len(p) and j + h - 1 < len(bench):
                stock = p["Close"].iloc[i + h - 1] / entry_px - 1
                spy = bench["Close"].iloc[j + h - 1] / bench["Open"].iloc[j] - 1
                row[f"ret_{h}d"], row[f"excess_{h}d"] = stock, stock - spy
                row[f"exit_date_{h}d"] = p.index[i + h - 1]
        rows.append(row)
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev["month"] = pd.to_datetime(ev["entry_date"]).dt.to_period("M")
    ev["year"] = pd.to_datetime(ev["entry_date"]).dt.year
    return ev


def winsorize(x: pd.Series, q: float) -> pd.Series:
    lo, hi = x.quantile([q, 1 - q])
    return x.clip(lo, hi)


def clustered_spread(d: pd.DataFrame, col: str, long_mask, short_mask) -> dict:
    # avg per month first then t-test across months
    monthly = pd.DataFrame({"long": d[long_mask].groupby("month")[col].mean(),
                            "short": d[short_mask].groupby("month")[col].mean()}).dropna()
    spread = monthly["long"] - monthly["short"]
    if len(spread) < 3:
        return {"spread": None, "t_stat": None, "p_value": None, "months": len(spread)}
    t, p = stats.ttest_1samp(spread, 0)
    return {"spread": round(float(spread.mean()), 5), "t_stat": round(float(t), 3),
            "p_value": round(float(p), 4), "months": int(len(spread))}


def quarter_quintiles(ev: pd.DataFrame, feature: str) -> pd.Series:
    q = pd.to_datetime(ev["call_date"]).dt.to_period("Q")
    return ev.groupby(q)[feature].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1 if len(x) >= 25 else np.nan)


def summarize(ev: pd.DataFrame, horizons: list[int], bt: dict) -> dict:
    out = {"events": int(len(ev)), "tickers": int(ev["ticker"].nunique()),
           "first_entry": str(ev["entry_date"].min().date()), "last_entry": str(ev["entry_date"].max().date())}
    for f in ("confidence", "gap"):
        ev[f"q_{f}"] = quarter_quintiles(ev, f)
    for h in horizons:
        col = f"excess_{h}d"
        if col not in ev or ev[col].notna().sum() < 30:  # recent calls won't have 60d yet
            continue
        d = ev.dropna(subset=[col]).copy()
        d[col] = winsorize(d[col], bt["winsorize"])
        per = {}
        for s in ORDER:
            x = d.loc[d["signal"] == s, col]
            per[s] = {"n": int(len(x)), "mean": round(float(x.mean()), 5), "median": round(float(x.median()), 5),
                      "hit_rate": round(float((x > 0).mean() if s != "SELL" else (x < 0).mean()), 4)}
        res = {"by_signal": per,
               "buy_minus_sell": clustered_spread(d, col, d["signal"] == "BUY", d["signal"] == "SELL")}
        for f in ("confidence", "gap"):
            qmeans = d.groupby(f"q_{f}")[col].mean()
            res[f"quintiles_{f}"] = {
                "mean_by_quintile": {int(k): round(float(v), 5) for k, v in qmeans.items()},
                "q5_minus_q1": clustered_spread(d, col, d[f"q_{f}"] == 5, d[f"q_{f}"] == 1),
            }
        res["rank_ic"] = {}
        for f in FEATURES:
            if d[f].notna().sum() < 10:
                continue
            rho, p = stats.spearmanr(d[f], d[col], nan_policy="omit")
            res["rank_ic"][f] = {"spearman": round(float(rho), 4), "p_value": round(float(p), 4)}
        by_year = {}
        for y, g in d.groupby("year"):
            by_year[int(y)] = {
                "n": int(len(g)),
                "buy_minus_sell": clustered_spread(g, col, g["signal"] == "BUY", g["signal"] == "SELL")["spread"],
                "confidence_q5_minus_q1": clustered_spread(g, col, g["q_confidence"] == 5,
                                                           g["q_confidence"] == 1)["spread"],
            }
        res["by_year"] = by_year
        out[f"{h}d"] = res
    return out


def run(cfg: dict) -> dict:
    bt = cfg["backtest"]
    horizons = bt["horizons"]
    out = cfg["paths"]["results"] / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    sig = pd.read_csv(cfg["paths"]["results"] / "signals.csv")
    sig = sig[~sig["warmup"]]
    start = (pd.Timestamp(sig["call_date"].min()) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    tickers = sorted(set(sig["ticker"]))
    px = prices.load(tickers + [bt["benchmark"]], start, cfg["paths"]["prices"])
    bench = px.pop(bt["benchmark"])
    print("prices for", len(px), "/", len(tickers), "tickers")

    ev = event_returns(sig, px, bench, horizons, bt)
    ev.to_csv(out / "event_returns.csv.gz", index=False, float_format="%.5f")
    trades, open_lots = inventory.run_inventory(ev, px, bench, cfg["inventory"])
    book = inventory.daily_book(trades, px, bench)
    inv = inventory.holdings(open_lots, px, bench)
    trades.to_csv(out / "trades.csv.gz", index=False, float_format="%.4f")
    book.to_csv(out / "book_daily.csv.gz", float_format="%.4f")
    inv.to_csv(out / "inventory.csv", index=False, float_format="%.4f")

    summary = {"signal_diagnostics": summarize(ev, horizons, bt),
               "inventory": inventory.summarize(trades, book, inv)}
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str))
    charts.make_all(cfg, summary, book, out)

    d = summary["signal_diagnostics"]
    print(f"evaluation -> {out}: {d['events']} tradeable calls, {d['tickers']} tickers")
    for h in horizons:
        if f"{h}d" not in d:
            continue
        bs, q = d[f"{h}d"]["buy_minus_sell"], d[f"{h}d"]["quintiles_confidence"]["q5_minus_q1"]
        print(f"  {h:>2}d: BUY-SELL {bs['spread']} (t={bs['t_stat']}) | confidence Q5-Q1 {q['spread']} (t={q['t_stat']})")
    s = summary["inventory"]
    print(f"  inventory: P&L ${s['total_pnl']:,.0f} vs SPY shadow ${s['shadow_spy_pnl']:,.0f}")
    return summary
