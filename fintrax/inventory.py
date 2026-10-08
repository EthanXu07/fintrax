"""Signal-driven inventory: no fixed holding period.

Each call's signal changes the position in that stock at the first market open
after the call (see ``backtest.entry_index``):

* BUY  -> buy one lot (``lot_dollars`` of stock), up to ``max_lots`` per stock;
* SELL -> sell every lot held in that stock; long-only, so with none held nothing happens;
* HOLD -> keep whatever is held.

Lots stay in inventory until a later SELL (or the stock's price history ending,
e.g. a delisting or acquisition, which closes them at the last close). Each lot
is mirrored by a shadow lot of SPY bought and sold on the same days with the
same dollars, so "excess" means "versus putting that money in the index instead".
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Lot:
    ticker: str
    opened: pd.Timestamp
    shares: float
    cost: float
    spy_shares: float


def run_inventory(events: pd.DataFrame, px: dict, bench: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Replay signals in date order -> (trade log, open lots by ticker).

    `events` needs ticker, entry_date, signal and entry_price (the entry-day open).
    Every signal gets a row; only BUY and SELL rows move shares or cash.
    """
    lot_dollars, max_lots = cfg["lot_dollars"], cfg["max_lots"]
    held: dict[str, list[Lot]] = {}
    log = []

    def record(**row) -> None:
        log.append({"shares": 0.0, "dollars": 0.0, "cost": 0.0, "spy_shares": 0.0, "spy_dollars": 0.0,
                    "pnl": np.nan, "spy_pnl": np.nan, "days_held": np.nan} | row)

    def close_lot(lot: Lot, date, price: float, spy_price: float, reason: str) -> None:
        proceeds, spy_proceeds = lot.shares * price, lot.spy_shares * spy_price
        record(date=date, ticker=lot.ticker, action="SELL", reason=reason, price=price,
               shares=-lot.shares, dollars=proceeds, cost=-lot.cost,
               spy_shares=-lot.spy_shares, spy_dollars=spy_proceeds,
               pnl=proceeds - lot.cost, spy_pnl=spy_proceeds - lot.cost,
               days_held=int(np.busday_count(lot.opened.date(), pd.Timestamp(date).date())))

    for r in events.sort_values(["entry_date", "ticker"]).itertuples():
        day, lots = r.entry_date, held.setdefault(r.ticker, [])
        spy_open = bench["Open"].get(day, np.nan)
        if np.isnan(spy_open):
            continue
        if r.signal == "BUY" and len(lots) < max_lots:
            lot = Lot(r.ticker, day, lot_dollars / r.entry_price, lot_dollars, lot_dollars / spy_open)
            lots.append(lot)
            record(date=day, ticker=r.ticker, action="BUY", reason="signal", price=r.entry_price,
                   shares=lot.shares, dollars=-lot_dollars, cost=lot_dollars,
                   spy_shares=lot.spy_shares, spy_dollars=-lot_dollars)
        elif r.signal == "SELL" and lots:
            for lot in lots:
                close_lot(lot, day, r.entry_price, spy_open, "signal")
            lots.clear()
        else:
            reason = {"HOLD": "hold", "BUY": "at max lots", "SELL": "nothing held"}[r.signal]
            record(date=day, ticker=r.ticker, action=r.signal if r.signal == "HOLD" else f"{r.signal} (no trade)",
                   reason=reason, price=r.entry_price)

    # Lots whose price history stops (delisted, acquired) close at the last close.
    last_day = bench.index[-1]
    for ticker, lots in held.items():
        end = px[ticker].index[-1]
        if lots and end < last_day - pd.Timedelta(days=7):
            for lot in lots:
                close_lot(lot, end, px[ticker]["Close"].iloc[-1], bench["Close"].asof(end), "price history ended")
            lots.clear()

    log = pd.DataFrame(log).sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)
    # Lots held after each row (BUY rows add one, SELL rows close one each).
    step = np.select([log["action"] == "BUY", log["action"] == "SELL"], [1, -1], 0)
    log["lots_after"] = pd.Series(step).groupby(log["ticker"]).cumsum()
    return log, {t: lots for t, lots in held.items() if lots}


def daily_book(trades: pd.DataFrame, px: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """Daily mark-to-market of the inventory and its SPY shadow.

    Trades settle at the open (or the last close for delistings), so marking
    shares at each day's close and adding cumulative cash gives total P&L.
    """
    real = trades[trades["action"].isin(["BUY", "SELL"])]
    days = bench.loc[real["date"].min():].index
    shares = real.pivot_table(index="date", columns="ticker", values="shares", aggfunc="sum")
    shares = shares.reindex(days).fillna(0.0).cumsum().clip(lower=0.0)  # clip float dust
    closes = pd.DataFrame({t: px[t]["Close"] for t in shares.columns}).reindex(days).ffill()
    flows = real.groupby("date")[["dollars", "spy_dollars", "spy_shares", "cost"]].sum().reindex(days, fill_value=0.0)

    book = pd.DataFrame(index=days)
    book["market_value"] = (shares * closes).sum(axis=1)
    book["cash"] = flows["dollars"].cumsum()
    book["spy_value"] = flows["spy_shares"].cumsum() * bench["Close"].reindex(days)
    book["spy_cash"] = flows["spy_dollars"].cumsum()
    book["cost_basis_open"] = flows["cost"].cumsum()
    book["stocks_held"] = (shares > 1e-9).sum(axis=1)
    book["pnl"] = book["market_value"] + book["cash"]
    book["spy_pnl"] = book["spy_value"] + book["spy_cash"]
    book["excess_pnl"] = book["pnl"] - book["spy_pnl"]

    # Daily return on capital at work: P&L change over yesterday's value plus today's buys.
    buys = real[real["action"] == "BUY"].groupby("date")["cost"].sum().reindex(days, fill_value=0.0)
    for name, value, pnl in [("ret", "market_value", "pnl"), ("spy_ret", "spy_value", "spy_pnl")]:
        base = book[value].shift(1, fill_value=0.0) + buys
        book[name] = np.where(base > 0, book[pnl].diff().fillna(book[pnl]) / base.where(base > 0, 1), 0.0)
    book["excess_ret"] = book["ret"] - book["spy_ret"]
    return book


def return_stats(r: pd.Series) -> dict:
    growth = (1 + r).cumprod()
    years = len(r) / 252
    return {"annualized_return": round(float(growth.iloc[-1] ** (1 / years) - 1), 4),
            "annualized_vol": round(float(r.std() * np.sqrt(252)), 4),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3) if r.std() > 0 else None,
            "max_drawdown": round(float((growth / growth.cummax() - 1).min()), 4),
            "total_return": round(float(growth.iloc[-1] - 1), 4)}


def holdings(open_lots: dict, px: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """Current inventory per stock, marked at the latest close."""
    spy_close = bench["Close"].iloc[-1]
    rows = []
    for ticker, lots in open_lots.items():
        close = px[ticker]["Close"].iloc[-1]
        shares, cost = sum(l.shares for l in lots), sum(l.cost for l in lots)
        rows.append({"ticker": ticker, "lots": len(lots), "shares": shares, "cost_basis": cost,
                     "first_bought": min(l.opened for l in lots).date(), "last_close": close,
                     "market_value": shares * close, "unrealized_pnl": shares * close - cost,
                     "spy_unrealized_pnl": sum(l.spy_shares for l in lots) * spy_close - cost})
    cols = ["ticker", "lots", "shares", "cost_basis", "first_bought", "last_close", "market_value",
            "unrealized_pnl", "spy_unrealized_pnl"]
    return pd.DataFrame(rows, columns=cols).sort_values("market_value", ascending=False)


def summarize(trades: pd.DataFrame, book: pd.DataFrame, inv: pd.DataFrame) -> dict:
    closed = trades[trades["action"] == "SELL"]
    exits = closed[closed["reason"] == "signal"].groupby(["ticker", "date"]).ngroups
    year = book.index.year
    year_end_excess = book["excess_pnl"].groupby(year).last()
    excess_in_year = year_end_excess.diff().fillna(year_end_excess)
    by_year = {}
    for y, g in book.groupby(year):
        by_year[int(y)] = {"strategy": round(float((1 + g["ret"]).prod() - 1), 4),
                           "shadow_spy": round(float((1 + g["spy_ret"]).prod() - 1), 4),
                           "excess_pnl": round(float(excess_in_year[y]), 2),
                           "lots_bought": int(((trades["action"] == "BUY") & (trades["date"].dt.year == y)).sum())}
    final = book.iloc[-1]
    return {
        "lots_bought": int((trades["action"] == "BUY").sum()),
        "lots_sold": int(len(closed)),
        "sell_exits": int(exits),
        "skipped_buys_at_cap": int((trades["action"] == "BUY (no trade)").sum()),
        "sells_with_nothing_held": int((trades["action"] == "SELL (no trade)").sum()),
        "closed_lot_win_rate": round(float((closed["pnl"] > 0).mean()), 4) if len(closed) else None,
        "closed_lot_beat_spy_rate": round(float((closed["pnl"] > closed["spy_pnl"]).mean()), 4) if len(closed) else None,
        "median_days_held": float(closed["days_held"].median()) if len(closed) else None,
        "realized_pnl": round(float(closed["pnl"].sum()), 2),
        "unrealized_pnl": round(float(inv["unrealized_pnl"].sum()), 2),
        "total_pnl": round(float(final["pnl"]), 2),
        "shadow_spy_pnl": round(float(final["spy_pnl"]), 2),
        "excess_pnl": round(float(final["excess_pnl"]), 2),
        "max_capital_deployed": round(float(book["cost_basis_open"].max()), 2),
        "stocks_held_now": int(len(inv)),
        "strategy": return_stats(book["ret"]),
        "shadow_spy": return_stats(book["spy_ret"]),
        "excess": return_stats(book["excess_ret"]),
        "by_year": by_year,
    }
