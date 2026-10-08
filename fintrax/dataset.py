"""Turns -> sentences -> weak labels -> train/val/test sample (split by call).

Works part by part so tens of thousands of calls never sit in memory at once:
every management sentence is labeled and written to processed/sentences/, and a
class-balanced sample of the labeled ones (capped by ``label.max_sentences``)
becomes the train/val/test sets.
"""
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fintrax import lexicon

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[$0-9])")
# Safe-harbor and logistics language is scripted legal boilerplate, not management tone.
BOILERPLATE = re.compile(
    r"forward-looking|safe harbor|risk factors|form 10-[kq]|form 8-k|non-gaap|"
    r"(?:undertake|assume)s? no (?:obligation|duty)|webcast|replay|investor relations website|"
    r"press release|operator instructions|\[.*?\]|turn (?:the call|it) (?:over|back)|next question|"
    r"thank you|thanks,? (?:everyone|operator)|good (?:morning|afternoon|evening)",
    re.I,
)
INLINE_NOISE = re.compile(r"\[[^\]]*\]")
SENTENCE_COLS = ["call_id", "ticker", "company", "call_date", "call_time_et", "section", "speaker", "role", "turn_id", "sentence"]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(INLINE_NOISE.sub(" ", text)) if s.strip()]


def to_sentences(turns: pd.DataFrame, min_tokens: int) -> pd.DataFrame:
    """Management sentences only: prepared remarks plus executives' Q&A answers."""
    mgmt = turns[turns["speaker_type"] == "executive"].copy()
    mgmt["call_id"] = mgmt["ticker"] + "_" + mgmt["call_date"]
    mgmt["turn_id"] = np.arange(len(mgmt))
    mgmt["sentence"] = mgmt["text"].map(split_sentences)
    sents = mgmt.explode("sentence").dropna(subset=["sentence"])
    sents = sents[sents["sentence"].str.split().str.len() >= min_tokens]
    sents = sents[~sents["sentence"].str.contains(BOILERPLATE)]
    return sents[SENTENCE_COLS].reset_index(drop=True)


def split_of(call_id: str) -> str:
    """Deterministic 70/15/15 split by call, stable as new calls are added."""
    bucket = int(hashlib.md5(call_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def label_part(args: tuple[str, str, int]) -> dict:
    src, dst, min_tokens = args
    sents = to_sentences(pd.read_parquet(src), min_tokens)
    lex = lexicon.load()
    sents["weak_label"] = sents["sentence"].map(lambda s: lexicon.weak_label(s, lex))
    sents.to_parquet(dst, index=False)
    return sents["weak_label"].fillna("mixed").value_counts().to_dict() | {"calls": sents["call_id"].nunique()}


def run(cfg: dict) -> None:
    lc, paths = cfg["label"], cfg["paths"]
    out_dir = paths["processed"] / "sentences"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    parts = sorted((paths["interim"] / "turns").glob("part-*.parquet"))
    jobs = [(str(p), str(out_dir / p.name), lc["min_tokens"]) for p in parts]
    with ProcessPoolExecutor(cfg["parse"]["workers"]) as pool:
        counts = pd.DataFrame(list(pool.map(label_part, jobs))).fillna(0).sum()
    n_low, n_high, n_neutral = counts.get("low", 0), counts.get("high", 0), counts.get("neutral", 0)
    print(f"{int(counts.drop('calls').sum())} management sentences from {int(counts['calls'])} calls; "
          f"weak labels: low {int(n_low)}, neutral {int(n_neutral)}, high {int(n_high)}, "
          f"mixed/dropped {int(counts.get('mixed', 0))}")

    # Class-balanced sample: low and high keep their natural ratio; neutral (which
    # dominates) is cut to the mean of the two; everything scaled to the cap.
    target = {"low": n_low, "high": n_high, "neutral": min(n_neutral, lc["neutral_ratio"] * (n_low + n_high) / 2)}
    scale = min(1.0, lc["max_sentences"] / sum(target.values()))
    frac = {k: v * scale / max(counts.get(k, 1), 1) for k, v in target.items()}
    rng = np.random.default_rng(lc["seed"])
    sample = []
    for p in sorted(out_dir.glob("part-*.parquet")):
        df = pd.read_parquet(p, columns=["call_id", "ticker", "call_date", "section", "sentence", "weak_label"])
        df = df.dropna(subset=["weak_label"])
        keep = rng.random(len(df)) < df["weak_label"].map(frac).to_numpy()
        sample.append(df[keep])
    labeled = pd.concat(sample, ignore_index=True)
    labeled["label"] = labeled["weak_label"].map(lexicon.LABELS.index)
    labeled["split"] = labeled["call_id"].map(split_of)
    for name, df in labeled.groupby("split"):
        df.drop(columns="split").reset_index(drop=True).to_parquet(paths["processed"] / f"{name}.parquet", index=False)
        print(f"  {name}: {len(df)} sentences, {df['call_id'].nunique()} calls, "
              f"{df['weak_label'].value_counts().to_dict()}")


def read_sentences(cfg: dict, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["processed"] / "sentences", columns=columns)
