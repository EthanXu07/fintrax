"""Score every management sentence and aggregate to per-call confidence features.

Per sentence: P(low), P(neutral), P(high) from the fine-tuned model, plus
FinBERT's original sentiment (P(positive) - P(negative)) as a separate feature.
Per call and section: confidence index = mean(P(high) - P(low)).
"""
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Scorer:
    """A sequence classifier kept on the GPU, scoring sentences in length-sorted batches."""

    def __init__(self, model_dir: str, max_len: int, half: bool = True):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.float16 if half and device().type != "cpu" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=dtype).to(device()).eval()
        self.max_len = max_len
        self.label2id = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.id2name = {k: v for v, k in self.label2id.items()}

    @torch.inference_mode()
    def __call__(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        # Sort by length so batches pad less.
        order = np.argsort([len(t) for t in texts])
        probs = np.zeros((len(texts), self.model.config.num_labels), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.model.device)
            probs[idx] = torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
        return probs


def predict_proba(model_dir: str, texts: list[str], max_len: int, batch_size: int = 256) -> tuple[np.ndarray, dict]:
    scorer = Scorer(model_dir, max_len)
    return scorer(texts, batch_size), scorer.label2id


def call_features(sents: pd.DataFrame) -> pd.DataFrame:
    """One row per call: confidence by section and the Q&A-vs-prepared gap."""
    sents = sents.assign(call_time_et=sents["call_time_et"].fillna(""), company=sents["company"].fillna(""))
    agg = sents.groupby(["call_id", "ticker", "company", "call_date", "call_time_et", "section"]).agg(
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
    return calls


def run(cfg: dict) -> pd.DataFrame:
    """Score every sentence part (resumable: finished parts are skipped) and aggregate per call."""
    paths, max_len = cfg["paths"], cfg["train"]["max_len"]
    src_dir, out_dir = paths["processed"] / "sentences", paths["processed"] / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(src_dir.glob("part-*.parquet"))
    todo = [p for p in parts if not (out_dir / p.name).exists()
            or (out_dir / p.name).stat().st_mtime < max(p.stat().st_mtime, (paths["model"] / "config.json").stat().st_mtime)]
    print(f"{len(parts)} sentence parts, {len(todo)} to score on {device()}")
    if todo:
        conf_model = Scorer(str(paths["model"]), max_len)
        # FinBERT's own sentiment doubles scoring time; it's an optional extra feature.
        sent_model = Scorer(cfg["train"]["base_model"], max_len) if cfg["score"]["sentiment"] else None
        labels = np.array(sorted(conf_model.label2id, key=conf_model.label2id.get))
    for i, p in enumerate(todo, 1):
        t0 = time.monotonic()
        sents = pd.read_parquet(p)
        texts = sents["sentence"].tolist()
        probs = conf_model(texts)
        ids = conf_model.label2id
        sents["p_low"], sents["p_neutral"], sents["p_high"] = (probs[:, ids[k]] for k in ("low", "neutral", "high"))
        sents["confidence"] = sents["p_high"] - sents["p_low"]
        sents["pred"] = labels[probs.argmax(1)]
        if sent_model is not None:
            sprobs = sent_model(texts)
            sents["sentiment"] = sprobs[:, sent_model.label2id["positive"]] - sprobs[:, sent_model.label2id["negative"]]
        else:
            sents["sentiment"] = np.nan
        sents.drop(columns=["speaker", "role", "turn_id"]).to_parquet(out_dir / p.name, index=False)
        rate = len(texts) / (time.monotonic() - t0)
        print(f"[{i}/{len(todo)}] {p.name}: {len(texts)} sentences, {rate:.0f}/s", flush=True)

    calls = pd.concat([call_features(pd.read_parquet(p)) for p in sorted(out_dir.glob("part-*.parquet"))],
                      ignore_index=True)
    calls = calls.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    calls["delta_qoq"] = calls.groupby("ticker")["conf_qna"].diff()
    calls.to_parquet(paths["processed"] / "calls.parquet", index=False)
    print(f"{len(calls)} calls scored; mean conf prepared {calls['conf_prepared'].mean():.3f}, "
          f"Q&A {calls['conf_qna'].mean():.3f}; gap < 0 on {(calls['gap'] < 0).mean():.1%}")
    return calls
