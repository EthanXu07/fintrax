import numpy as np
import pandas as pd

from fintrax.backtest import clustered_spread, entry_index, event_returns, quarter_quintiles
from fintrax.inventory import daily_book, holdings, run_inventory

DAYS = pd.bdate_range("2024-01-01", periods=80)
BT = {"min_price": 5.0, "min_dollar_volume": 1e6}


def flat(price: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    close = price * (1 + drift) ** np.arange(len(DAYS))
    return pd.DataFrame({"Open": close, "Close": close, "Volume": 1e6}, index=DAYS)


def test_entry_is_same_day_premarket_else_next_day():
    px = flat()
    assert px.index[entry_index(px, "2024-02-01", premarket=True)] == pd.Timestamp("2024-02-01")
    assert px.index[entry_index(px, "2024-02-01", premarket=False)] == pd.Timestamp("2024-02-02")
    # A Saturday call enters Monday either way.
    assert px.index[entry_index(px, "2024-02-03", premarket=False)] == pd.Timestamp("2024-02-05")


def calls(**overrides) -> pd.DataFrame:
    row = {"ticker": "UP", "company": "Up Inc", "call_date": "2024-02-01", "call_time_et": "16:30", "signal": "BUY",
           "gap": 0.1, "conf_qna": 0.1, "conf_prepared": 0.0, "sentiment_qna": 0.0}
    return pd.DataFrame([row | overrides])


def test_event_returns_are_excess_over_benchmark():
    px = {"UP": flat(drift=0.01)}
    ev = event_returns(calls(), px, flat(), [1, 5], BT)
    assert ev.loc[0, "entry_date"] == pd.Timestamp("2024-02-02")
    assert np.isclose(ev.loc[0, "excess_5d"], 1.01 ** 4 - 1)  # open of day 1 to close of day 5


def test_penny_stocks_and_illiquid_names_are_skipped():
    assert event_returns(calls(), {"UP": flat(price=2.0)}, flat(), [1], BT).empty
    thin = flat()
    thin["Volume"] = 100
    assert event_returns(calls(), {"UP": thin}, flat(), [1], BT).empty


def test_clustered_spread_averages_months_first():
    d = pd.DataFrame({"month": pd.PeriodIndex(["2024-01"] * 3 + ["2024-02"] * 2 + ["2024-03"] * 2, freq="M"),
                      "x": [3.0, 3.0, 3.0, 1.0, 0.0, 2.0, 0.0], "side": ["L", "L", "S", "L", "S", "L", "S"]})
    out = clustered_spread(d, "x", d["side"] == "L", d["side"] == "S")
    assert out["months"] == 3 and np.isclose(out["spread"], (0 + 1 + 2) / 3)


def test_quintiles_are_ranked_within_quarter():
    ev = pd.DataFrame({"call_date": ["2024-01-15"] * 25 + ["2024-04-15"] * 25,
                       "gap": list(range(25)) + list(range(100, 125))})
    q = quarter_quintiles(ev, "gap")
    assert q.iloc[0] == 1 and q.iloc[24] == 5 and q.iloc[25] == 1 and q.iloc[49] == 5


INV = {"lot_dollars": 1000, "max_lots": 5}


def signals(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """(ticker, entry_date, signal) rows; entry at that day's open."""
    return pd.DataFrame([{"ticker": t, "entry_date": pd.Timestamp(d), "signal": s} for t, d, s in rows])


def replay(sig: pd.DataFrame, px: dict, bench: pd.DataFrame):
    sig["entry_price"] = [px[r.ticker].at[r.entry_date, "Open"] for r in sig.itertuples()]
    return run_inventory(sig, px, bench, INV)


def test_buy_buy_sell_exits_the_whole_position():
    px = {"UP": flat(drift=0.01)}
    trades, open_lots = replay(signals(("UP", "2024-02-01", "BUY"), ("UP", "2024-02-15", "BUY"),
                                       ("UP", "2024-03-01", "SELL")), px, flat())
    assert trades["action"].tolist() == ["BUY", "BUY", "SELL", "SELL"]
    assert trades["lots_after"].tolist() == [1, 2, 1, 0]
    assert open_lots == {}
    sells = trades[trades["action"] == "SELL"]
    assert (sells["pnl"] > 0).all() and np.isclose(sells["spy_pnl"], 0).all()  # SPY flat
    # Held until the SELL, not for a fixed number of days.
    assert sells["days_held"].tolist() == [21, 11]


def test_sell_with_nothing_held_never_shorts():
    trades, open_lots = replay(signals(("UP", "2024-02-01", "SELL"), ("UP", "2024-02-02", "HOLD")),
                               {"UP": flat()}, flat())
    assert trades["action"].tolist() == ["SELL (no trade)", "HOLD"]
    assert (trades["shares"] == 0).all() and open_lots == {}


def test_buys_stop_at_the_lot_cap():
    days = [d.strftime("%Y-%m-%d") for d in DAYS[25:31]]
    trades, open_lots = replay(signals(*[("UP", d, "BUY") for d in days]), {"UP": flat()}, flat())
    assert (trades["action"] == "BUY").sum() == 5 and trades["action"].iloc[-1] == "BUY (no trade)"
    assert len(open_lots["UP"]) == 5


def test_delisted_stock_is_closed_at_its_last_close():
    gone = flat().iloc[:40]
    trades, open_lots = replay(signals(("GONE", "2024-02-01", "BUY")), {"GONE": gone}, flat())
    last = trades.iloc[-1]
    assert last["action"] == "SELL" and last["reason"] == "price history ended"
    assert last["date"] == gone.index[-1] and open_lots == {}


def test_shadow_spy_matches_when_stock_tracks_the_index():
    same = flat(drift=0.005)
    trades, open_lots = replay(signals(("TRK", "2024-02-01", "BUY")), {"TRK": same}, same)
    book = daily_book(trades, {"TRK": same}, same)
    assert np.allclose(book["excess_pnl"], 0, atol=1e-9)


def test_book_reconciles_with_trade_log():
    px = {"UP": flat(drift=0.01), "DN": flat(drift=-0.004)}
    bench = flat(drift=0.002)
    trades, open_lots = replay(signals(("UP", "2024-02-01", "BUY"), ("DN", "2024-02-05", "BUY"),
                                       ("UP", "2024-02-20", "SELL"), ("DN", "2024-03-01", "BUY")), px, bench)
    book = daily_book(trades, px, bench)
    inv = holdings(open_lots, px, bench)
    realized = trades.loc[trades["action"] == "SELL", "pnl"].sum()
    assert np.isclose(book["pnl"].iloc[-1], realized + inv["unrealized_pnl"].sum())
    spy_realized = trades.loc[trades["action"] == "SELL", "spy_pnl"].sum()
    assert np.isclose(book["spy_pnl"].iloc[-1], spy_realized + inv["spy_unrealized_pnl"].sum())
