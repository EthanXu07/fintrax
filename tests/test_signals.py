import numpy as np
import pandas as pd

from fintrax.signals import classify, expanding_z, make_signals

SC = {"min_history": 3, "buy_gap_z": 0.5, "buy_conf_z": 0.0, "sell_gap_z": -0.5, "sell_conf_z": -1.0}


def test_classify_rules():
    assert classify(1.0, 0.5, SC) == "BUY"
    assert classify(-1.0, 0.5, SC) == "SELL"        # confidence collapses in Q&A
    assert classify(0.0, -2.0, SC) == "SELL"        # Q&A confidence very low overall
    assert classify(1.0, -0.5, SC) == "HOLD"        # better than script, but still weak
    assert classify(0.0, 0.0, SC) == "HOLD"
    assert classify(np.nan, 1.0, SC) == "HOLD"


def test_expanding_z_uses_only_earlier_dates():
    df = pd.DataFrame({"call_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"],
                       "x": [0.0, 1.0, 2.0, 100.0, -100.0]})
    z = expanding_z(df, "x", min_history=3)
    assert z.iloc[:3].isna().all()
    # Same-day calls don't see each other: both are scored against [0, 1, 2] only.
    assert z.iloc[3] == (100 - 1) / 1 and z.iloc[4] == (-100 - 1) / 1


def test_make_signals_end_to_end():
    calls = pd.DataFrame({
        "call_date": [f"2026-01-0{i}" for i in range(1, 7)],
        "conf_qna": [0.1, 0.2, 0.3, 0.9, 0.2, -0.8],
        "gap": [0.0, 0.1, -0.1, 0.8, 0.0, -0.9],
    })
    sig = make_signals(calls, SC)
    assert sig["warmup"].tolist() == [True] * 3 + [False] * 3
    assert sig["signal"].tolist() == ["HOLD", "HOLD", "HOLD", "BUY", "HOLD", "SELL"]


def test_confidence_only_signals_use_overall_confidence_without_lookahead():
    from fintrax.confidence_signals import make_signals as confidence_signals, overall_confidence
    calls = pd.DataFrame({
        "call_date": [f"2026-01-0{i}" for i in range(1, 7)],
        "conf_prepared": [0.0, 0.1, 0.2, 0.9, 0.1, -0.9],
        "conf_qna": [0.0, 0.1, 0.2, 0.9, 0.1, -0.9],
        "n_prepared": [10] * 6, "n_qna": [30] * 6,
    })
    # Sentence-weighted: 10 prepared + 30 Q&A sentences.
    two = pd.DataFrame({"conf_prepared": [1.0], "conf_qna": [0.0], "n_prepared": [10], "n_qna": [30]})
    assert np.isclose(overall_confidence(two).iloc[0], 0.25)
    sig = confidence_signals(calls, {"buy_z": 0.5, "sell_z": -0.5}, min_history=3)
    assert sig["warmup"].tolist() == [True] * 3 + [False] * 3
    assert sig["signal"].tolist() == ["HOLD", "HOLD", "HOLD", "BUY", "HOLD", "SELL"]
