"""Event backtest: does the signal predict post-call returns relative to SPY?

Entry is the first market open after the call ends: the same day's open for
pre-market calls (before 9:30 ET), otherwise the next trading day's open. Calls
held during market hours therefore skip part of the day-one reaction, which is
conservative. The h-day return runs from that open to the close h trading days
later, minus SPY over the same window.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ORDER = ["BUY", "HOLD", "SELL"]
FEATURES = ["gap", "conf_qna", "conf_prepared", "sentiment_qna"]
COLORS = {"BUY": "#2a9d8f", "HOLD": "#8d99ae", "SELL": "#e76f51"}


def load_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=False)
    prices = {t: data[t][["Open", "Close"]].dropna() for t in tickers if t in data.columns.get_level_values(0)}
    # yfinance occasionally drops a ticker from a batch; retry those one at a time.
    for t in [t for t in tickers if t not in prices or prices[t].empty]:
        one = yf.download(t, start=start, auto_adjust=True, progress=False, multi_level_index=False)
        if not one.empty:
            prices[t] = one[["Open", "Close"]].dropna()
    missing = [t for t in tickers if t not in prices or prices[t].empty]
    if missing:
        print("no prices for:", missing)
    return prices


def window_return(px: pd.DataFrame, call_date: str, h: int, premarket: bool = False) -> float:
    day = pd.Timestamp(call_date)
    after = px.index[px.index >= day] if premarket else px.index[px.index > day]
    if len(after) < h:
        return np.nan
    entry, exit_ = after[0], after[h - 1]
    return px.at[exit_, "Close"] / px.at[entry, "Open"] - 1


def event_returns(sig: pd.DataFrame, prices: dict, benchmark: str, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for r in sig.itertuples():
        if r.ticker not in prices:
            continue
        row = {"ticker": r.ticker, "call_date": r.call_date, "signal": r.signal,
               **{f: getattr(r, f) for f in FEATURES}}
        premarket = isinstance(r.call_time_et, str) and r.call_time_et < "09:30"
        for h in horizons:
            stock = window_return(prices[r.ticker], r.call_date, h, premarket)
            bench = window_return(prices[benchmark], r.call_date, h, premarket)
            row[f"excess_{h}d"] = stock - bench
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame, horizons: list[int]) -> dict:
    out = {}
    for h in horizons:
        col = f"excess_{h}d"
        d = ev.dropna(subset=[col])
        per = {}
        for s in ORDER:
            x = d.loc[d["signal"] == s, col]
            hit = (x > 0).mean() if s != "SELL" else (x < 0).mean()
            per[s] = {"n": int(len(x)), "mean_excess": round(float(x.mean()), 5) if len(x) else None,
                      "hit_rate": round(float(hit), 3) if len(x) else None}
        buy, sell = d.loc[d["signal"] == "BUY", col], d.loc[d["signal"] == "SELL", col]
        spread = {"buy_minus_sell": None, "t_stat": None, "p_value": None}
        if len(buy) > 1 and len(sell) > 1:
            t, p = stats.ttest_ind(buy, sell, equal_var=False)
            spread = {"buy_minus_sell": round(float(buy.mean() - sell.mean()), 5),
                      "t_stat": round(float(t), 3), "p_value": round(float(p), 4)}
        # Threshold-free check: rank correlation of each raw feature with the return.
        ic = {}
        for f in FEATURES:
            rho, p = stats.spearmanr(d[f], d[col])
            ic[f] = {"spearman": round(float(rho), 4), "p_value": round(float(p), 4)}
        out[f"{h}d"] = {"by_signal": per, **spread, "rank_ic": ic}
    return out


def plot(ev: pd.DataFrame, horizons: list[int], results) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(ORDER)
    for i, s in enumerate(ORDER):
        means = [ev.loc[ev["signal"] == s, f"excess_{h}d"].mean() * 100 for h in horizons]
        ax.bar(np.arange(len(horizons)) + (i - 1) * width, means, width, label=s, color=COLORS[s])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(horizons)), [f"{h}-day" for h in horizons])
    ax.set_ylabel("Mean excess return vs SPY (%)")
    ax.set_title("Post-call excess return by Fintrax signal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results / "excess_by_signal.png", dpi=150)
    plt.close(fig)

    # Long BUY / short SELL: one equal-sized bet per signal, summed in call-date order.
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, color in zip(horizons, ["#264653", "#2a9d8f", "#e9c46a"]):
        d = ev[ev["signal"] != "HOLD"].dropna(subset=[f"excess_{h}d"]).sort_values("call_date")
        pnl = np.where(d["signal"] == "BUY", 1, -1) * d[f"excess_{h}d"]
        ax.plot(pd.to_datetime(d["call_date"]), pnl.cumsum() * 100, color=color, label=f"{h}-day")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Cumulative excess return (%, summed per event)")
    ax.set_title("Long BUY / short SELL vs SPY")
    ax.legend(frameon=False, title="holding period")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(results / "cumulative_long_short.png", dpi=150)
    plt.close(fig)


def run(cfg: dict) -> dict:
    bc, results = cfg["backtest"], cfg["paths"]["results"]
    sig = pd.read_csv(results / "signals.csv")
    sig = sig[~sig["warmup"]]
    start = (pd.Timestamp(sig["call_date"].min()) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    prices = load_prices(sorted(set(sig["ticker"])) + [bc["benchmark"]], start)
    ev = event_returns(sig, prices, bc["benchmark"], bc["horizons"])
    ev.round(5).to_csv(results / "event_returns.csv", index=False)
    summary = summarize(ev, bc["horizons"])
    (results / "backtest.json").write_text(json.dumps(summary, indent=2))
    plot(ev, bc["horizons"], results)
    for k, v in summary.items():
        print(f"{k}: " + ", ".join(f"{s} n={m['n']} mean={m['mean_excess']}" for s, m in v["by_signal"].items())
              + f" | BUY-SELL {v['buy_minus_sell']} (t={v['t_stat']}, p={v['p_value']})"
              + " | IC " + ", ".join(f"{f}={m['spearman']}" for f, m in v["rank_ic"].items()))
    return summary
