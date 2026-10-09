"""Evaluation figures (results/evaluation/). The signal charts themselves are in report.py.

* signal_returns.png        excess return vs SPY after BUY / HOLD / SELL at fixed horizons
* confidence_quintiles.png  returns by confidence quintile, ranked within each quarter
* inventory.png             signal-driven inventory P&L vs the same dollars in SPY, by year
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

GREEN, RED, GRAY, INK, BLUE = "#1a9850", "#d73027", "#9aa0a6", "#202124", "#4575b4"
SIGNAL_COLORS = {"BUY": GREEN, "HOLD": GRAY, "SELL": RED}
DOLLARS = FuncFormatter(lambda v, _: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")


def signal_returns(cfg: dict, d: dict, path) -> None:
    horizons = [h for h in cfg["backtest"]["horizons"] if f"{h}d" in d]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / 3
    for i, s in enumerate(["BUY", "HOLD", "SELL"]):
        means = [d[f"{h}d"]["by_signal"][s]["mean"] * 100 for h in horizons]
        ax.bar(np.arange(len(horizons)) + (i - 1) * width, means, width, label=s, color=SIGNAL_COLORS[s])
    ax.axhline(0, color=INK, lw=0.8)
    ticks = []
    for h in horizons:
        bs = d[f"{h}d"]["buy_minus_sell"]
        ticks.append(f"{h}-day\nmonthly BUY−SELL {bs['spread'] * 100:+.2f}% (t={bs['t_stat']:.2f})"
                     if bs["spread"] is not None else f"{h}-day")
    ax.set_xticks(range(len(horizons)), ticks, fontsize=8)
    ax.set_ylabel("Mean excess return vs SPY (%)")
    ax.set_title("After each signal: mean excess return per call (winsorized)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def quintiles(cfg: dict, d: dict, path) -> None:
    horizons = [h for h in cfg["backtest"]["horizons"] if f"{h}d" in d]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cmap = plt.get_cmap("viridis")
    for k, h in enumerate(horizons):
        q = d[f"{h}d"]["quintiles_confidence"]["mean_by_quintile"]
        xs = np.arange(1, 6) + (k - (len(horizons) - 1) / 2) * 0.18
        ax.bar(xs, [q.get(i, np.nan) * 100 for i in range(1, 6)], 0.18, label=f"{h}-day",
               color=cmap(0.1 + 0.75 * k / max(len(horizons) - 1, 1)))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(1, 6), ["Q1\nleast confident", "Q2", "Q3", "Q4", "Q5\nmost confident"])
    ax.set_ylabel("Mean excess return vs SPY (%)")
    ax.set_title("Returns by call-confidence quintile (ranked within each quarter)", loc="left", fontsize=11)
    ax.legend(frameon=False, title="horizon")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def inventory_chart(cfg: dict, s: dict, book: pd.DataFrame, path) -> None:
    lot, cap = cfg["inventory"]["lot_dollars"], cfg["inventory"]["max_lots"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax.plot(book.index, book["pnl"], color=GREEN, lw=1.5, label="signal inventory")
    ax.plot(book.index, book["spy_pnl"], color=BLUE, lw=1.2, label="same dollars in SPY")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.yaxis.set_major_formatter(DOLLARS)
    ax.set_title(f"Inventory (BUY adds ${lot:,}, max {cap} lots; SELL exits) vs SPY", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    by_year = pd.DataFrame(s["by_year"]).T
    x = np.arange(len(by_year))
    ax2.bar(x - 0.2, by_year["strategy"].astype(float) * 100, 0.4, color=GREEN, label="inventory")
    ax2.bar(x + 0.2, by_year["shadow_spy"].astype(float) * 100, 0.4, color=BLUE, label="SPY")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xticks(x, by_year.index, fontsize=8, rotation=45)
    ax2.set_ylabel("return on capital deployed (%)")
    ax2.set_title("By year", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)
    for a in (ax, ax2):
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(cfg: dict, summary: dict, book: pd.DataFrame, out) -> None:
    signal_returns(cfg, summary["signal_diagnostics"], out / "signal_returns.png")
    quintiles(cfg, summary["signal_diagnostics"], out / "confidence_quintiles.png")
    inventory_chart(cfg, summary["inventory"], book, out / "inventory.png")
