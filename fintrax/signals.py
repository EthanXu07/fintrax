"""Prepared-vs-Q&A confidence discrepancy -> BUY / HOLD / SELL.

Idea: prepared remarks are scripted and almost always upbeat; Q&A is not.
Management that stays (or gets more) confident when questioned off-script is a
positive signal; confidence that collapses in Q&A is a negative one.

Each call is z-scored against all calls dated strictly before it (an expanding
window), so a signal never uses information from the future.
"""
import numpy as np
import pandas as pd


def expanding_z(df: pd.DataFrame, col: str, min_history: int) -> pd.Series:
    """z-score of each row vs rows with an earlier call_date (same-day calls excluded).

    Uses running sums per date, so it's O(n) instead of rescanning history per call.
    """
    values = df[col].astype(float)
    by_date = pd.DataFrame({"d": df["call_date"], "x": values, "x2": values ** 2}).groupby("d").agg(
        n=("x", "size"), s=("x", "sum"), s2=("x2", "sum"))
    # Totals over strictly earlier dates.
    prior = by_date.cumsum().shift(1, fill_value=0).reindex(df["call_date"]).to_numpy()
    n, s, s2 = prior[:, 0], prior[:, 1], prior[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = (s2 - n * mean ** 2) / (n - 1)
        z = (values.to_numpy() - mean) / np.sqrt(var)
    z[(n < min_history) | ~(var > 0)] = np.nan
    return pd.Series(z, index=df.index)


def classify(gap_z: float, conf_z: float, sc: dict) -> str:
    if np.isnan(gap_z) or np.isnan(conf_z):
        return "HOLD"
    if gap_z < sc["sell_gap_z"] or conf_z < sc["sell_conf_z"]:
        return "SELL"
    if gap_z > sc["buy_gap_z"] and conf_z > sc["buy_conf_z"]:
        return "BUY"
    return "HOLD"


def make_signals(calls: pd.DataFrame, sc: dict) -> pd.DataFrame:
    df = calls.sort_values("call_date").reset_index(drop=True).copy()
    df["gap_z"] = expanding_z(df, "gap", sc["min_history"])
    df["conf_qna_z"] = expanding_z(df, "conf_qna", sc["min_history"])
    df["signal"] = [classify(g, c, sc) for g, c in zip(df["gap_z"], df["conf_qna_z"])]
    df["warmup"] = df["gap_z"].isna()
    return df


def run(cfg: dict) -> pd.DataFrame:
    calls = pd.read_parquet(cfg["paths"]["processed"] / "calls.parquet")
    sig = make_signals(calls, cfg["signals"])
    cols = ["ticker", "company", "call_date", "call_time_et", "conf_prepared", "conf_qna", "gap", "gap_z", "conf_qna_z",
            "sentiment_prepared", "sentiment_qna", "delta_qoq", "warmup", "signal"]
    out = cfg["paths"]["results"] / "signals.csv"
    sig[cols].round(4).to_csv(out, index=False)
    live = sig[~sig["warmup"]]
    print(f"{len(sig)} calls ({sig['warmup'].sum()} warm-up) -> {out}")
    print(live["signal"].value_counts().to_string())
    return sig
