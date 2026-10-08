"""BUY / SELL straight from the confidence model, without the prepared-vs-Q&A gap rule.

Each call gets one number: its overall confidence, the mean P(high) - P(low)
over every management sentence (prepared remarks and Q&A together). It is
z-scored against every earlier call (no look-ahead):

* BUY  if z >  ``buy_z``  (management sounds more confident than usual)
* SELL if z <  ``sell_z`` (less confident than usual)
* HOLD otherwise

and evaluated like the pilot backtest: mean excess return vs SPY after each
signal at fixed horizons, plus a cumulative long-BUY / short-SELL line.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fintrax import backtest, prices
from fintrax.signals import expanding_z

GREEN, RED, GRAY, INK = "#1a9850", "#d73027", "#9aa0a6", "#202124"
COLORS = {"BUY": GREEN, "HOLD": GRAY, "SELL": RED}


def overall_confidence(calls: pd.DataFrame) -> pd.Series:
    """Sentence-weighted mean confidence across both sections."""
    n_p, n_q = calls["n_prepared"].astype(float), calls["n_qna"].astype(float)
    return (calls["conf_prepared"] * n_p + calls["conf_qna"] * n_q) / (n_p + n_q)


def make_signals(calls: pd.DataFrame, cc: dict, min_history: int) -> pd.DataFrame:
    df = calls.sort_values("call_date").reset_index(drop=True).copy()
    df["confidence"] = overall_confidence(df)
    df["confidence_z"] = expanding_z(df, "confidence", min_history)
    df["signal"] = np.select([df["confidence_z"] > cc["buy_z"], df["confidence_z"] < cc["sell_z"]],
                             ["BUY", "SELL"], "HOLD")
    df["warmup"] = df["confidence_z"].isna()
    return df


def stats_by_horizon(ev: pd.DataFrame, horizons: list[int], winsor: float) -> dict:
    out = {}
    for h in horizons:
        col = f"excess_{h}d"
        if col not in ev or ev[col].notna().sum() < 30:
            continue
        d = ev.dropna(subset=[col]).copy()
        d[col] = backtest.winsorize(d[col], winsor)
        per = {}
        for s in ("BUY", "HOLD", "SELL"):
            x = d.loc[d["signal"] == s, col]
            per[s] = {"n": int(len(x)), "mean": round(float(x.mean()), 5) if len(x) else None,
                      "hit_rate": round(float((x > 0).mean() if s != "SELL" else (x < 0).mean()), 4) if len(x) else None}
        out[f"{h}d"] = {"by_signal": per,
                        "buy_minus_sell": backtest.clustered_spread(d, col, d["signal"] == "BUY", d["signal"] == "SELL")}
    return out


def plot(ev: pd.DataFrame, summary: dict, out: Path, label: str) -> None:
    horizons = [int(k[:-1]) for k in summary]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / 3
    for i, s in enumerate(("BUY", "HOLD", "SELL")):
        means = [(summary[f"{h}d"]["by_signal"][s]["mean"] or np.nan) * 100 for h in horizons]
        bars = ax.bar(np.arange(len(horizons)) + (i - 1) * width, means, width, label=s, color=COLORS[s])
        for b, h in zip(bars, horizons):
            n = summary[f"{h}d"]["by_signal"][s]["n"]
            ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, 0), xytext=(0, -10 if b.get_height() >= 0 else 4),
                        textcoords="offset points", ha="center", fontsize=6, color=GRAY)
    ax.axhline(0, color=INK, lw=0.8)
    ticks = []
    for h in horizons:
        bs = summary[f"{h}d"]["buy_minus_sell"]
        t = f"t={bs['t_stat']:.2f}" if bs["t_stat"] is not None else "t=n/a"
        ticks.append(f"{h}-day\nmonthly BUY−SELL {bs['spread'] * 100:+.2f}%\n({t})"
                     if bs["spread"] is not None else f"{h}-day")
    ax.set_xticks(range(len(horizons)), ticks, fontsize=8)
    ax.set_ylabel("Mean excess return vs SPY (%)")
    ax.set_title(f"Confidence-only signals: mean excess return per trade ({label})", loc="left", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "excess_by_signal.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for h, color in zip(horizons, ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]):
        d = ev[ev["signal"] != "HOLD"].dropna(subset=[f"excess_{h}d"]).sort_values("entry_date")
        pnl = np.where(d["signal"] == "BUY", 1, -1) * d[f"excess_{h}d"]
        ax.plot(pd.to_datetime(d["entry_date"]), pnl.cumsum() * 100, color=color, label=f"{h}-day")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("Cumulative excess return (%, summed per trade)")
    ax.set_title(f"Confidence-only signals: long BUY / short SELL vs SPY ({label})", loc="left", fontsize=11)
    ax.legend(frameon=False, title="holding period")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "cumulative_long_short.png", dpi=130)
    plt.close(fig)


def run(cfg: dict, calls_path: Path | None = None, out: Path | None = None, label: str | None = None) -> dict:
    cc, bt = cfg["confidence_signals"], cfg["backtest"]
    calls = pd.read_parquet(calls_path or cfg["paths"]["processed"] / "calls.parquet")
    out = out or cfg["paths"]["results"] / "confidence_only"
    out.mkdir(parents=True, exist_ok=True)

    sig = make_signals(calls, cc, cfg["signals"]["min_history"])
    if "company" not in sig:
        sig["company"] = ""
    sig["gap"] = sig["conf_qna"] - sig["conf_prepared"]
    if "sentiment_qna" not in sig:
        sig["sentiment_qna"] = np.nan
    cols = ["ticker", "company", "call_date", "call_time_et", "conf_prepared", "conf_qna", "confidence",
            "confidence_z", "warmup", "signal"]
    sig[cols].round(4).to_csv(out / "signals.csv", index=False)

    live = sig[~sig["warmup"]]
    start = (pd.Timestamp(live["call_date"].min()) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    px = prices.load(sorted(set(live["ticker"])) + [bt["benchmark"]], start, cfg["paths"]["prices"])
    bench = px.pop(bt["benchmark"])
    ev = backtest.event_returns(live, px, bench, bt["horizons"], bt)
    summary = stats_by_horizon(ev, bt["horizons"], bt["winsorize"])
    (out / "backtest.json").write_text(json.dumps(
        {"calls": int(len(sig)), "warmup": int(sig["warmup"].sum()), "tradeable_calls": int(len(ev)),
         "signals": live["signal"].value_counts().to_dict(), "rules": cc, "horizons": summary}, indent=2))
    plot(ev, summary, out, label or f"{len(ev):,} calls, {ev['entry_date'].min():%Y}–{ev['entry_date'].max():%Y}")

    print(f"confidence-only: {len(ev)} tradeable calls, signals {live['signal'].value_counts().to_dict()} -> {out}")
    for k, v in summary.items():
        bs = v["buy_minus_sell"]
        print(f"  {k:>4}: " + ", ".join(f"{s} n={m['n']} mean={m['mean']}" for s, m in v["by_signal"].items())
              + f" | BUY-SELL {bs['spread']} (t={bs['t_stat']}, p={bs['p_value']})")
    return summary
