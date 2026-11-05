"""confidence -> BUY/HOLD/SELL

confidence is z-scored against all *earlier* calls (no lookahead).
z > buy_z -> BUY, z < sell_z -> SELL, else HOLD
"""
import numpy as np
import pandas as pd

OUTPUT_COLS = [
    "ticker", "company", "call_date", "call_time_et", "signal", "confidence", "confidence_z",
    "conf_prepared", "conf_qna", "gap", "pct_high", "pct_low", "confidence_change", "sentences", "warmup",
]


def expanding_z(df: pd.DataFrame, col: str, min_history: int) -> pd.Series:
    # z vs everything strictly before this date. running sums so it's not O(n^2)
    values = df[col].astype(float)
    by_date = pd.DataFrame({"d": df["call_date"], "x": values, "x2": values ** 2}).groupby("d").agg(
        n=("x", "size"), s=("x", "sum"), s2=("x2", "sum"))
    prior = by_date.cumsum().shift(1, fill_value=0).reindex(df["call_date"]).to_numpy()
    n, s, s2 = prior[:, 0], prior[:, 1], prior[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = (s2 - n * mean ** 2) / (n - 1)
        z = (values.to_numpy() - mean) / np.sqrt(var)
    z[(n < min_history) | ~(var > 0)] = np.nan
    return pd.Series(z, index=df.index)


def weighted(calls, prepared, qna):
    n_p, n_q = calls["n_prepared"].astype(float), calls["n_qna"].astype(float)
    return (calls[prepared] * n_p + calls[qna] * n_q) / (n_p + n_q)


def make_signals(calls: pd.DataFrame, sc: dict) -> pd.DataFrame:
    df = calls.sort_values(["call_date", "ticker"]).reset_index(drop=True).copy()
    df["confidence"] = weighted(df, "conf_prepared", "conf_qna")
    df["confidence_z"] = expanding_z(df, "confidence", sc["min_history"])
    df["signal"] = np.select([df["confidence_z"] > sc["buy_z"], df["confidence_z"] < sc["sell_z"]],
                             ["BUY", "SELL"], "HOLD")
    df["warmup"] = df["confidence_z"].isna()
    # extra stuff, doesn't affect signal
    df["gap"] = df["conf_qna"] - df["conf_prepared"]
    if {"pct_high_prepared", "pct_high_qna"} <= set(df):
        df["pct_high"] = weighted(df, "pct_high_prepared", "pct_high_qna")
        df["pct_low"] = weighted(df, "pct_low_prepared", "pct_low_qna")
    df["sentences"] = df["n_prepared"] + df["n_qna"]
    df["confidence_change"] = df.groupby("ticker")["confidence"].diff()
    if "company" not in df:
        df["company"] = ""
    return df


def run(cfg: dict) -> pd.DataFrame:
    calls = pd.read_parquet(cfg["paths"]["processed"] / "calls.parquet")
    sig = make_signals(calls, cfg["signals"])
    out = cfg["paths"]["results"] / "signals.csv"
    sig.reindex(columns=OUTPUT_COLS).round(4).to_csv(out, index=False)
    live = sig[~sig["warmup"]]
    print(f"{len(sig)} calls ({int(sig['warmup'].sum())} warm-up) -> {out}")
    print(live["signal"].value_counts().to_string())

    from fintrax import report
    report.run(cfg, sig)
    return sig
