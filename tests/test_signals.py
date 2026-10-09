import numpy as np
import pandas as pd

from fintrax.signals import expanding_z, make_signals, weighted

SC = {"min_history": 3, "buy_z": 0.5, "sell_z": -0.5}


def calls(conf: list[float], **extra) -> pd.DataFrame:
    n = len(conf)
    return pd.DataFrame({"ticker": ["X"] * n, "call_date": [f"2026-01-{i + 1:02d}" for i in range(n)],
                         "conf_prepared": conf, "conf_qna": conf, "n_prepared": [10] * n, "n_qna": [30] * n,
                         **extra})


def test_expanding_z_uses_only_earlier_dates():
    df = pd.DataFrame({"call_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"],
                       "x": [0.0, 1.0, 2.0, 100.0, -100.0]})
    z = expanding_z(df, "x", min_history=3)
    assert z.iloc[:3].isna().all()
    # Same-day calls don't see each other: both are scored against [0, 1, 2] only.
    assert z.iloc[3] == (100 - 1) / 1 and z.iloc[4] == (-100 - 1) / 1


def test_confidence_is_sentence_weighted_across_sections():
    two = pd.DataFrame({"conf_prepared": [1.0], "conf_qna": [0.0], "n_prepared": [10], "n_qna": [30]})
    assert np.isclose(weighted(two, "conf_prepared", "conf_qna").iloc[0], 0.25)


def test_signals_follow_confidence_z_thresholds():
    sig = make_signals(calls([0.0, 0.1, 0.2, 0.9, 0.1, -0.9]), SC)
    assert sig["warmup"].tolist() == [True] * 3 + [False] * 3
    assert sig["signal"].tolist() == ["HOLD", "HOLD", "HOLD", "BUY", "HOLD", "SELL"]


def test_gap_is_context_only():
    # Same overall confidence, very different prepared-vs-Q&A gaps -> same signal.
    base = calls([0.0, 0.1, 0.2, 0.9])
    flipped = base.copy()
    flipped.loc[3, ["conf_prepared", "n_prepared", "conf_qna", "n_qna"]] = [0.9, 20, 0.9, 20]
    a, b = make_signals(base, SC), make_signals(flipped, SC)
    assert a["signal"].iloc[3] == b["signal"].iloc[3] == "BUY"
    skewed = base.copy()
    skewed.loc[3, ["conf_prepared", "conf_qna"]] = [1.5, 0.7]  # weighted mean still 0.9
    s = make_signals(skewed, SC)
    assert np.isclose(s["confidence"].iloc[3], 0.9) and s["signal"].iloc[3] == "BUY"
    assert np.isclose(s["gap"].iloc[3], -0.8)


def test_confidence_change_is_per_company():
    df = pd.concat([calls([0.1, 0.3]), calls([0.5, 0.2]).assign(ticker="Y")], ignore_index=True)
    sig = make_signals(df, SC).sort_values(["ticker", "call_date"])
    assert np.allclose(sig["confidence_change"].dropna(), [0.2, -0.3])
