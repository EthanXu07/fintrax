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
    """z-score of each row vs rows with an earlier call_date (ties excluded)."""
    dates = df["call_date"].to_numpy()
    values = df[col].to_numpy(dtype=float)
    z = np.full(len(df), np.nan)
    for i, d in enumerate(dates):
        past = values[dates < d]
        if len(past) >= min_history and past.std(ddof=1) > 0:
            z[i] = (values[i] - past.mean()) / past.std(ddof=1)
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
    cols = ["ticker", "call_date", "call_time_et", "conf_prepared", "conf_qna", "gap", "gap_z", "conf_qna_z",
            "sentiment_prepared", "sentiment_qna", "delta_qoq", "warmup", "signal"]
    out = cfg["paths"]["results"] / "signals.csv"
    sig[cols].round(4).to_csv(out, index=False)
    live = sig[~sig["warmup"]]
    print(f"{len(sig)} calls ({sig['warmup'].sum()} warm-up) -> {out}")
    print(live["signal"].value_counts().to_string())
    return sig
