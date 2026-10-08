"""Backtest figures: per-stock inventory charts with buy/sell arrows, P&L vs SPY, diagnostics.

Arrow convention (the usual one on trading charts): a green up-arrow under the
price is a buy (one lot added to inventory), a red down-arrow above the price is
a sell (the whole position exited). Faded hollow markers are signals that had
nothing to act on: a BUY at the lot cap, or a SELL with nothing held. The
background turns green while the model holds the stock, darker with more lots.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

GREEN, RED, GRAY, INK, BLUE = "#1a9850", "#d73027", "#9aa0a6", "#202124", "#4575b4"
SIGNAL_COLORS = {"BUY": GREEN, "HOLD": GRAY, "SELL": RED}
DOLLARS = FuncFormatter(lambda v, _: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")


def _arrow(ax, x, y, up: bool, color: str) -> None:
    """Arrow pointing at the price: up from below (buy) or down from above (sell)."""
    tail, head = (y * 0.86, y * 0.985) if up else (y * 1.16, y * 1.015)
    ax.annotate("", xy=(x, head), xytext=(x, tail), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", color=color, lw=2.0,
                                shrinkA=0, shrinkB=0))


def lots_series(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Lots held at each day's close."""
    last = trades.groupby("date")["lots_after"].last()
    return last.reindex(days.union(last.index)).ffill().fillna(0).reindex(days)


def ticker_pnl(t: pd.DataFrame, px: pd.DataFrame, bench: pd.DataFrame) -> tuple[float, float]:
    """(strategy P&L, shadow SPY P&L) for one stock: cash flows plus what's still held."""
    real = t[t["action"].isin(["BUY", "SELL"])]
    pnl = real["dollars"].sum() + real["shares"].sum() * px["Close"].iloc[-1]
    spy = real["spy_dollars"].sum() + real["spy_shares"].sum() * bench["Close"].iloc[-1]
    return pnl, spy


def style_price_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def inventory_chart(ax, ticker: str, px: pd.DataFrame, trades: pd.DataFrame, max_lots: int,
                    lots_ax=None, label_exits: bool = True) -> dict:
    t = trades[trades["ticker"] == ticker]
    if t.empty:
        return {}
    p = px.loc[t["date"].min() - pd.Timedelta(days=60):, "Close"]
    lots = lots_series(t, p.index)

    # Green background while held, darker with more lots.
    change = lots.ne(lots.shift()).cumsum()
    for _, seg in lots.groupby(change):
        if seg.iloc[0] > 0:
            ax.axvspan(seg.index[0], seg.index[-1] + pd.Timedelta(days=1), color=GREEN, lw=0,
                       alpha=0.06 + 0.16 * seg.iloc[0] / max_lots, zorder=0)
    ax.plot(p.index, p.values, color=INK, lw=0.9, zorder=2)

    buys = t[t["action"] == "BUY"]
    for r in buys.itertuples():
        _arrow(ax, r.date, r.price, up=True, color=GREEN)
    exits = t[t["action"] == "SELL"].groupby("date").agg(price=("price", "first"), pnl=("pnl", "sum"),
                                                          cost=("cost", "sum"), reason=("reason", "first"))
    for date, r in exits.iterrows():
        _arrow(ax, date, r["price"], up=False, color=RED)
        if label_exits:
            ret = r["pnl"] / -r["cost"]
            ax.annotate(f"{ret:+.0%}", xy=(date, r["price"] * 1.22), ha="center", va="bottom", fontsize=7,
                        color=INK, zorder=6)
    for action, marker, color, dy in [("BUY (no trade)", "^", GREEN, 0.93), ("SELL (no trade)", "v", RED, 1.07)]:
        s = t[t["action"] == action]
        ax.scatter(s["date"], s["price"] * dy, marker=marker, s=28, facecolors="none", edgecolors=color,
                   alpha=0.55, zorder=4)
    hold = t[t["action"] == "HOLD"]
    ax.scatter(hold["date"], p.reindex(hold["date"], method="ffill"), s=9, color=GRAY, zorder=3)
    style_price_axis(ax)

    if lots_ax is not None:
        lots_ax.fill_between(lots.index, lots.values, step="post", color=GREEN, alpha=0.5, lw=0)
        lots_ax.step(lots.index, lots.values, where="post", color=GREEN, lw=1)
        lots_ax.set_ylim(0, max_lots + 0.5)
        lots_ax.set_yticks(range(0, max_lots + 1))
        lots_ax.set_ylabel("lots held")
        lots_ax.grid(alpha=0.25)
        lots_ax.spines[["top", "right"]].set_visible(False)
    return {"buys": len(buys), "exits": len(exits), "held_now": int(lots.iloc[-1])}


def legend_handles() -> list:
    return [
        Line2D([], [], marker="^", ls="", color=GREEN, markersize=9, label="BUY: add one lot"),
        Line2D([], [], marker="v", ls="", color=RED, markersize=9, label="SELL: exit the position"),
        Line2D([], [], marker="^", ls="", markerfacecolor="none", markeredgecolor=GREEN, markersize=7,
               label="BUY skipped (at lot cap)"),
        Line2D([], [], marker="v", ls="", markerfacecolor="none", markeredgecolor=RED, markersize=7,
               label="SELL with nothing held"),
        Line2D([], [], marker="o", ls="", color=GRAY, markersize=4, label="HOLD"),
        Patch(color=GREEN, alpha=0.25, label="holding (darker = more lots)"),
    ]


def pick_chart_tickers(cfg: dict, trades: pd.DataFrame, n: int = 9) -> list[str]:
    bought = trades.loc[trades["action"] == "BUY", "ticker"].value_counts()
    wanted = [t for t in cfg["backtest"].get("chart_tickers", []) if bought.get(t, 0) >= 2]
    extra = [t for t in bought.index if bought[t] >= 4 and t not in wanted]
    return (wanted + extra)[:n]


def per_ticker_charts(cfg: dict, ev: pd.DataFrame, trades: pd.DataFrame, px: dict, bench: pd.DataFrame) -> list[str]:
    results, max_lots = cfg["paths"]["results"], cfg["inventory"]["max_lots"]
    out = results / "trades"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    tickers = pick_chart_tickers(cfg, trades)
    names = ev.drop_duplicates("ticker").set_index("ticker")["company"]
    for ticker in tickers:
        fig, (ax, lots_ax) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                          gridspec_kw={"height_ratios": [4, 1]})
        s = inventory_chart(ax, ticker, px[ticker], trades, max_lots, lots_ax=lots_ax)
        pnl, spy = ticker_pnl(trades[trades["ticker"] == ticker], px[ticker], bench)
        ax.set_title(f"{names.get(ticker, ticker)} ({ticker}): {s['buys']} lots bought, {s['exits']} SELL exits, "
                     f"{s['held_now']} held now  |  P&L ${pnl:,.0f} vs ${spy:,.0f} in SPY "
                     f"({pnl - spy:+,.0f})", loc="left", fontsize=11)
        ax.legend(handles=legend_handles(), loc="upper left", fontsize=8, frameon=False, ncol=3)
        lots_ax.xaxis.set_major_locator(mdates.YearLocator())
        lots_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(out / f"{ticker}.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, ticker in zip(axes.flat, tickers):
        s = inventory_chart(ax, ticker, px[ticker], trades, max_lots, label_exits=False)
        pnl, spy = ticker_pnl(trades[trades["ticker"] == ticker], px[ticker], bench)
        ax.set_title(f"{ticker} · {s['buys']} buys / {s['exits']} exits · vs SPY {pnl - spy:+,.0f} $", fontsize=10)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes.flat[len(tickers):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, fontsize=9)
    lot = cfg["inventory"]["lot_dollars"]
    fig.suptitle(f"Fintrax inventory: BUY adds a ${lot:,} lot (max {max_lots}), SELL exits the whole position",
                 fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(results / "trade_signals_grid.png", dpi=120)
    plt.close(fig)
    return tickers


def equity_chart(cfg: dict, book: pd.DataFrame, summary: dict) -> None:
    s = summary["inventory"]
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2, 1.3]})
    ax.plot(book.index, book["pnl"], color=GREEN, lw=1.6, label="Fintrax inventory")
    ax.plot(book.index, book["spy_pnl"], color=BLUE, lw=1.3, label="same dollars in SPY (shadow)")
    ax.plot(book.index, book["excess_pnl"], color=INK, lw=1, ls="--", label="difference")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.set_ylabel("cumulative P&L")
    ax.set_title(f"Inventory P&L: ${s['total_pnl']:,.0f} vs ${s['shadow_spy_pnl']:,.0f} in SPY "
                 f"({s['excess_pnl']:+,.0f})  |  {s['strategy']['annualized_return']:+.1%}/yr vs "
                 f"{s['shadow_spy']['annualized_return']:+.1%}/yr", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax2.plot(book.index, (1 + book["ret"]).cumprod(), color=GREEN, lw=1.4, label="Fintrax inventory")
    ax2.plot(book.index, (1 + book["spy_ret"]).cumprod(), color=BLUE, lw=1.2, label="SPY shadow")
    ax2.set_ylabel("growth of $1\n(time-weighted)")
    ax2.legend(frameon=False, fontsize=9)
    ax3.fill_between(book.index, book["cost_basis_open"], color=GRAY, alpha=0.6, lw=0)
    ax3.yaxis.set_major_formatter(DOLLARS)
    ax3.set_ylabel("capital deployed")
    ax3b = ax3.twinx()
    ax3b.plot(book.index, book["stocks_held"], color=INK, lw=0.8)
    ax3b.set_ylabel("stocks held")
    for a in (ax, ax2, ax3):
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["results"] / "equity_curve.png", dpi=130)
    plt.close(fig)

    by_year = pd.DataFrame(s["by_year"]).T
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(by_year))
    ax.bar(x - 0.2, by_year["strategy"].astype(float) * 100, 0.4, color=GREEN, label="Fintrax inventory")
    ax.bar(x + 0.2, by_year["shadow_spy"].astype(float) * 100, 0.4, color=BLUE, label="SPY shadow")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"{y}\n{int(n):,} buys" for y, n in zip(by_year.index, by_year["lots_bought"])], fontsize=8)
    ax.set_ylabel("return on capital deployed (%)")
    ax.set_title("Year by year: inventory vs the same dollars in SPY", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["results"] / "by_year.png", dpi=130)
    plt.close(fig)


def latest_picks(cfg: dict, n: int = 15) -> None:
    """The most recent calls' strongest BUY and SELL signals: what the model says to do now."""
    results = cfg["paths"]["results"]
    sig = pd.read_csv(results / "signals.csv")
    dates = pd.to_datetime(sig["call_date"])
    recent = sig[~sig["warmup"] & (dates >= dates.max() - pd.Timedelta(days=45))]
    picks = pd.concat([recent[recent["signal"] == "BUY"].nlargest(n, "gap_z"),
                       recent[recent["signal"] == "SELL"].nsmallest(n, "gap_z")]).sort_values("gap_z")
    if picks.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(picks) + 1.6))
    ax.barh([f"{r.ticker}  ({r.call_date})" for r in picks.itertuples()], picks["gap_z"],
            color=[SIGNAL_COLORS[s] for s in picks["signal"]])
    for y, r in enumerate(picks.itertuples()):
        ax.text(r.gap_z + (0.08 if r.gap_z > 0 else -0.08), y, "▲ BUY" if r.signal == "BUY" else "▼ SELL",
                va="center", ha="left" if r.gap_z > 0 else "right", fontsize=8, color=SIGNAL_COLORS[r.signal])
    ax.axvline(0, color=INK, lw=0.8)
    lim = np.abs(picks["gap_z"]).max() * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Q&A-vs-prepared confidence gap, z-score vs all earlier calls")
    ax.set_title(f"Latest picks: strongest signals from calls since {recent['call_date'].min()}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(results / "latest_picks.png", dpi=130)
    plt.close(fig)


def diagnostic_charts(cfg: dict, summary: dict) -> None:
    """Fixed-horizon views of signal quality (not the strategy, which has no fixed hold)."""
    results, d = cfg["paths"]["results"], summary["signal_diagnostics"]
    horizons = [h for h in cfg["backtest"]["horizons"] if f"{h}d" in d]

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / 3
    for i, s in enumerate(["BUY", "HOLD", "SELL"]):
        means = [d[f"{h}d"]["by_signal"][s]["mean"] * 100 for h in horizons]
        ax.bar(np.arange(len(horizons)) + (i - 1) * width, means, width, label=s, color=SIGNAL_COLORS[s])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(horizons)), [f"{h}-day" for h in horizons])
    ax.set_ylabel("Mean excess return vs SPY (%)")
    ax.set_title("Signal diagnostic: excess return after each signal (winsorized)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(results / "excess_by_signal.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, f, title in [(axes[0], "gap", "Q&A minus prepared confidence (gap)"),
                         (axes[1], "conf_qna", "Q&A confidence level")]:
        for k, h in enumerate(horizons):
            q = d[f"{h}d"][f"quintiles_{f}"]["mean_by_quintile"]
            xs = np.arange(1, 6) + (k - (len(horizons) - 1) / 2) * 0.18
            ax.bar(xs, [q.get(i, np.nan) * 100 for i in range(1, 6)], 0.18, label=f"{h}-day",
                   color=cmap(0.1 + 0.75 * k / max(len(horizons) - 1, 1)))
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(1, 6), ["Q1\nleast", "Q2", "Q3", "Q4", "Q5\nmost"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean excess return vs SPY (%)")
    axes[1].legend(frameon=False, title="horizon")
    fig.suptitle("Signal diagnostic: returns by confidence quintile (ranked within each calendar quarter)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(results / "confidence_quintiles.png", dpi=130)
    plt.close(fig)


def make_all(cfg: dict, ev: pd.DataFrame, trades: pd.DataFrame, book: pd.DataFrame, px: dict,
             summary: dict, bench: pd.DataFrame) -> None:
    equity_chart(cfg, book, summary)
    diagnostic_charts(cfg, summary)
    latest_picks(cfg)
    tickers = per_ticker_charts(cfg, ev, trades, px, bench)
    print(f"charts -> {cfg['paths']['results']} (inventory charts for {', '.join(tickers)})")
