"""Score every management sentence and aggregate to per-call confidence features.

Per sentence: P(low), P(neutral), P(high) from the fine-tuned model, plus
FinBERT's original sentiment (P(positive) - P(negative)) as a separate feature.
Per call and section: confidence index = mean(P(high) - P(low)).
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict_proba(model_dir: str, texts: list[str], max_len: int, batch_size: int = 128) -> tuple[np.ndarray, dict]:
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device()).eval()
    # Sort by length so batches pad less.
    order = np.argsort([len(t) for t in texts])
    probs = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(model.device)
        probs[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return probs, {v.lower(): k for k, v in model.config.id2label.items()}


def call_features(sents: pd.DataFrame) -> pd.DataFrame:
    """One row per call: confidence by section, the Q&A-vs-prepared gap, and QoQ change."""
    agg = sents.groupby(["call_id", "ticker", "call_date", "section"]).agg(
        conf=("confidence", "mean"),
        pct_low=("pred", lambda p: (p == "low").mean()),
        pct_high=("pred", lambda p: (p == "high").mean()),
        sentiment=("sentiment", "mean"),
        n=("confidence", "size"),
    ).unstack("section")
    agg.columns = [f"{stat}_{section}" for stat, section in agg.columns]
    calls = agg.reset_index().dropna(subset=["conf_prepared", "conf_qna"])
    calls["gap"] = calls["conf_qna"] - calls["conf_prepared"]
    calls["sentiment_gap"] = calls["sentiment_qna"] - calls["sentiment_prepared"]
    calls = calls.sort_values(["ticker", "call_date"])
    calls["delta_qoq"] = calls.groupby("ticker")["conf_qna"].diff()
    return calls.reset_index(drop=True)


def run(cfg: dict) -> pd.DataFrame:
    paths, max_len = cfg["paths"], cfg["train"]["max_len"]
    sents = pd.read_parquet(paths["processed"] / "sentences.parquet")
    texts = sents["sentence"].tolist()
    print(f"scoring {len(texts)} sentences on {device()}")

    probs, label2id = predict_proba(str(paths["model"]), texts, max_len)
    labels = np.array(sorted(label2id, key=label2id.get))
    sents["p_low"], sents["p_neutral"], sents["p_high"] = (probs[:, label2id[k]] for k in ("low", "neutral", "high"))
    sents["confidence"] = sents["p_high"] - sents["p_low"]
    sents["pred"] = labels[probs.argmax(1)]

    sent_probs, s_ids = predict_proba(cfg["train"]["base_model"], texts, max_len)
    sents["sentiment"] = sent_probs[:, s_ids["positive"]] - sent_probs[:, s_ids["negative"]]

    sents.to_parquet(paths["processed"] / "sentences_scored.parquet", index=False)
    calls = call_features(sents)
    calls.to_parquet(paths["processed"] / "calls.parquet", index=False)
    print(f"{len(calls)} calls scored; mean conf prepared {calls['conf_prepared'].mean():.3f}, "
          f"Q&A {calls['conf_qna'].mean():.3f}")
    return calls
