"""Signal report: what the confidence model says, call by call. No P&L here.

* latest_signals.csv / .png    the most recent calls' BUY and SELL signals, strongest first
* signal_charts/{TICKER}.png   price for context with a green up-arrow at each BUY call and a
  + signal_grid.png            red down-arrow at each SELL call, and the call-by-call confidence
                               z-score underneath with the BUY / SELL thresholds
* confidence_index.png         average management confidence across all calls, quarter by
                               quarter, and how many BUY / SELL signals each quarter produced
* confidence_distribution.png  where calls fall against the thresholds, and prepared vs Q&A
                               confidence by signal
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from fintrax import prices

GREEN, RED, GRAY, INK, BLUE = "#1a9850", "#d73027", "#9aa0a6", "#202124", "#4575b4"
COLORS = {"BUY": GREEN, "HOLD": GRAY, "SELL": RED}


def latest(sig: pd.DataFrame, days: int) -> pd.DataFrame:
    live = sig[~sig["warmup"]]
    dates = pd.to_datetime(live["call_date"])
    recent = live[dates >= dates.max() - pd.Timedelta(days=days)]
    return recent[recent["signal"] != "HOLD"].sort_values("confidence_z", ascending=False)


def latest_chart(recent: pd.DataFrame, sc: dict, path, n: int = 15) -> None:
    picks = pd.concat([recent[recent["signal"] == "BUY"].head(n),
                       recent[recent["signal"] == "SELL"].tail(n)]).sort_values("confidence_z")
    if picks.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(picks) + 1.7))
    labels = [f"{r.ticker}  {r.call_date}" for r in picks.itertuples()]
    ax.barh(labels, picks["confidence_z"], color=[COLORS[s] for s in picks["signal"]])
    for y, r in enumerate(picks.itertuples()):
        ax.text(r.confidence_z + (0.06 if r.confidence_z > 0 else -0.06), y,
                "▲ BUY" if r.signal == "BUY" else "▼ SELL", va="center", fontsize=8,
                ha="left" if r.confidence_z > 0 else "right", color=COLORS[r.signal])
    for thr in (sc["buy_z"], sc["sell_z"]):
        ax.axvline(thr, color=INK, lw=0.8, ls="--")
    lim = np.abs(picks["confidence_z"]).max() * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("call confidence, z-score vs all earlier calls (dashed: BUY / SELL thresholds)")
    ax.set_title(f"Latest signals: strongest BUY and SELL calls since {recent['call_date'].min()}", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _arrow(ax, x, y, up: bool) -> None:
    tail, head = (y * 0.86, y * 0.985) if up else (y * 1.16, y * 1.015)
    ax.annotate("", xy=(x, head), xytext=(x, tail), zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.6,head_width=0.35", lw=2.0,
                                color=GREEN if up else RED, shrinkA=0, shrinkB=0))


def ticker_panels(ax_px, ax_z, t: pd.DataFrame, px: pd.DataFrame | None, sc: dict) -> None:
    """Price with BUY/SELL arrows on top (if prices are available), confidence z below."""
    dates = pd.to_datetime(t["call_date"])
    if px is not None and not px.empty:
        p = px.loc[dates.min() - pd.Timedelta(days=60):, "Close"]
        ax_px.plot(p.index, p.values, color=INK, lw=0.9)
        for d, s in zip(dates, t["signal"]):
            y = p.asof(d)
            if np.isnan(y):
                continue
            if s == "HOLD":
                ax_px.scatter([d], [y], s=9, color=GRAY, zorder=3)
            else:
                _arrow(ax_px, d, y, up=s == "BUY")
        ax_px.set_yscale("log")
        ax_px.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax_px.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax_px.yaxis.set_minor_formatter(NullFormatter())
    ax_px.grid(alpha=0.25)
    ax_px.spines[["top", "right"]].set_visible(False)

    ax_z.axhspan(sc["buy_z"], 10, color=GREEN, alpha=0.07, lw=0)
    ax_z.axhspan(-10, sc["sell_z"], color=RED, alpha=0.07, lw=0)
    ax_z.plot(dates, t["confidence_z"], color=GRAY, lw=0.8, zorder=1)
    ax_z.scatter(dates, t["confidence_z"], c=[COLORS[s] for s in t["signal"]], s=22, zorder=2)
    ax_z.axhline(0, color=INK, lw=0.6)
    lim = max(2.0, np.nanmax(np.abs(t["confidence_z"])) * 1.15)
    ax_z.set_ylim(-lim, lim)
    ax_z.set_ylabel("confidence z")
    ax_z.grid(alpha=0.25)
    ax_z.spines[["top", "right"]].set_visible(False)


def legend() -> list:
    return [Line2D([], [], marker="^", ls="", color=GREEN, markersize=9, label="BUY call"),
            Line2D([], [], marker="v", ls="", color=RED, markersize=9, label="SELL call"),
            Line2D([], [], marker="o", ls="", color=GRAY, markersize=5, label="HOLD call")]


def pick_tickers(cfg: dict, sig: pd.DataFrame, n: int = 9) -> list[str]:
    live = sig[~sig["warmup"]]
    counts = live.groupby("ticker").size()
    wanted = [t for t in cfg["report"]["chart_tickers"] if counts.get(t, 0) >= 3]
    extra = counts[counts >= 6].sort_values(ascending=False).index
    return (wanted + [t for t in extra if t not in wanted])[:n]


def ticker_charts(cfg: dict, sig: pd.DataFrame, out) -> list[str]:
    sc = cfg["signals"]
    live = sig[~sig["warmup"]]
    tickers = pick_tickers(cfg, sig)
    if not tickers:
        return []
    start = (pd.to_datetime(live["call_date"]).min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        px = prices.load(tickers, start, cfg["paths"]["prices"])
    except Exception as e:  # price context is optional; the signals don't depend on it
        print(f"price download failed ({e}); charting confidence only")
        px = {}
    charts = out / "signal_charts"
    charts.mkdir(exist_ok=True)
    for old in charts.glob("*.png"):
        old.unlink()
    for ticker in tickers:
        t = live[live["ticker"] == ticker]
        fig, (ax_px, ax_z) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 2]})
        ticker_panels(ax_px, ax_z, t, px.get(ticker), sc)
        counts = t["signal"].value_counts()
        name = t["company"].iloc[0] or ticker
        ax_px.set_title(f"{name} ({ticker}): {len(t)} calls  ·  {counts.get('BUY', 0)} BUY  ·  "
                        f"{counts.get('SELL', 0)} SELL  ·  {counts.get('HOLD', 0)} HOLD", loc="left", fontsize=11)
        ax_px.legend(handles=legend(), loc="upper left", frameon=False, fontsize=8, ncol=3)
        ax_z.xaxis.set_major_locator(mdates.YearLocator())
        ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(charts / f"{ticker}.png", dpi=130)
        plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(6, 3, height_ratios=[3, 1.6] * 3, hspace=0.35)
    for k, ticker in enumerate(tickers):
        row, col = divmod(k, 3)
        ax_px = fig.add_subplot(grid[2 * row, col])
        ax_z = fig.add_subplot(grid[2 * row + 1, col], sharex=ax_px)
        ticker_panels(ax_px, ax_z, live[live["ticker"] == ticker], px.get(ticker), sc)
        ax_px.set_title(ticker, fontsize=10, loc="left")
        ax_px.tick_params(labelbottom=False)
        ax_z.set_ylabel("")
        ax_z.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.legend(handles=legend(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Fintrax signals by stock: arrows mark BUY / SELL calls; lower panels show call confidence "
                 f"(z) with the BUY (>{sc['buy_z']}) and SELL (<{sc['sell_z']}) zones",
                 x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.98)
    fig.savefig(out / "signal_grid.png", dpi=110)
    plt.close(fig)
    return tickers


def confidence_index(sig: pd.DataFrame, path) -> None:
    live = sig[~sig["warmup"]].copy()
    live["quarter"] = pd.to_datetime(live["call_date"]).dt.to_period("Q")
    q = live.groupby("quarter").agg(conf=("confidence", "mean"), calls=("signal", "size"),
                                    buy=("signal", lambda s: (s == "BUY").mean()),
                                    sell=("signal", lambda s: (s == "SELL").mean()))
    q = q[q["calls"] >= 20]
    if q.empty:
        return
    # Keep thin quarters as gaps so the line doesn't bridge missing data.
    q = q.reindex(pd.period_range(q.index.min(), q.index.max(), freq="Q"))
    x = q.index.to_timestamp()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1.4]})
    ax.plot(x, q["conf"], color=BLUE, lw=1.8, marker="o", ms=3)
    ax.set_ylabel("mean call confidence\nP(high) − P(low)")
    ax.set_title("Management confidence across all earnings calls, by quarter", loc="left")
    width = 60
    ax2.bar(x, q["buy"] * 100, width, color=GREEN, label="BUY")
    ax2.bar(x, -q["sell"] * 100, width, color=RED, label="SELL")
    ax2.axhline(0, color=INK, lw=0.6)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0f}%"))
    ax2.set_ylabel("share of calls")
    ax2.legend(frameon=False, ncol=2, loc="upper left")
    for a in (ax, ax2):
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def distribution(sig: pd.DataFrame, sc: dict, path) -> None:
    live = sig[~sig["warmup"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    z = live["confidence_z"].clip(-4, 4)
    bins = np.linspace(-4, 4, 81)
    for s in ("SELL", "HOLD", "BUY"):
        ax.hist(z[live["signal"] == s], bins=bins, color=COLORS[s],
                label=f"{s}: {(live['signal'] == s).sum():,} calls ({(live['signal'] == s).mean():.0%})")
    for thr in (sc["buy_z"], sc["sell_z"]):
        ax.axvline(thr, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("call confidence z-score")
    ax.set_ylabel("calls")
    ax.set_title("How calls split into signals", loc="left")
    ax.legend(frameon=False, fontsize=9)
    sample = live.sample(min(len(live), 6000), random_state=0)
    ax2.scatter(sample["conf_prepared"], sample["conf_qna"], s=6, alpha=0.5,
                c=[COLORS[s] for s in sample["signal"]])
    lo = min(sample["conf_prepared"].min(), sample["conf_qna"].min())
    hi = max(sample["conf_prepared"].max(), sample["conf_qna"].max())
    ax2.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls="--")
    ax2.set_xlabel("prepared-remarks confidence")
    ax2.set_ylabel("Q&A confidence")
    ax2.set_title("Prepared vs Q&A confidence, colored by signal (dashed: equal)", loc="left", fontsize=10)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: dict, sig: pd.DataFrame, out=None) -> None:
    out = out or cfg["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    sc = cfg["signals"]
    recent = latest(sig, cfg["report"]["latest_days"])
    recent.reindex(columns=["ticker", "company", "call_date", "signal", "confidence", "confidence_z",
                            "conf_prepared", "conf_qna", "gap", "confidence_change"]).round(4).to_csv(
        out / "latest_signals.csv", index=False)
    latest_chart(recent, sc, out / "latest_signals.png")
    confidence_index(sig, out / "confidence_index.png")
    distribution(sig, sc, out / "confidence_distribution.png")
    tickers = ticker_charts(cfg, sig, out)
    print(f"report -> {out}: {len(recent)} BUY/SELL calls in the last {cfg['report']['latest_days']} days; "
          f"signal charts for {', '.join(tickers) or 'none'}")
