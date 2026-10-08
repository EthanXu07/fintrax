"""Turns -> sentences -> weak labels -> train/val/test (split by call)."""
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(INLINE_NOISE.sub(" ", text)) if s.strip()]


def to_sentences(turns: pd.DataFrame, min_tokens: int) -> pd.DataFrame:
    """Management sentences only: prepared remarks plus executives' Q&A answers."""
    mgmt = turns[turns["speaker_type"] == "executive"].copy()
    mgmt["call_id"] = mgmt["ticker"] + "_" + mgmt["call_date"]
    mgmt["turn_id"] = range(len(mgmt))
    mgmt["sentence"] = mgmt["text"].map(split_sentences)
    sents = mgmt.explode("sentence").dropna(subset=["sentence"])
    sents = sents[sents["sentence"].str.split().str.len() >= min_tokens]
    sents = sents[~sents["sentence"].str.contains(BOILERPLATE)]
    cols = ["call_id", "ticker", "call_date", "call_time_et", "section", "speaker", "role", "turn_id", "sentence"]
    return sents[cols].reset_index(drop=True)


def run(cfg: dict) -> pd.DataFrame:
    lc, paths = cfg["label"], cfg["paths"]
    turns = pd.read_parquet(paths["interim"] / "turns.parquet")
    sents = to_sentences(turns, lc["min_tokens"])
    lex = lexicon.load()
    sents["weak_label"] = sents["sentence"].map(lambda s: lexicon.weak_label(s, lex))
    sents.to_parquet(paths["processed"] / "sentences.parquet", index=False)

    labeled = sents.dropna(subset=["weak_label"])
    # Neutral (no cues) dominates; downsample it to the mean of the other two classes.
    counts = labeled["weak_label"].value_counts()
    n_neutral = int(lc["neutral_ratio"] * (counts.get("low", 0) + counts.get("high", 0)) / 2)
    neutral = labeled[labeled["weak_label"] == "neutral"]
    labeled = pd.concat([
        labeled[labeled["weak_label"] != "neutral"],
        neutral.sample(min(n_neutral, len(neutral)), random_state=lc["seed"]),
    ])
    labeled["label"] = labeled["weak_label"].map(lexicon.LABELS.index)

    # 70/15/15 by call so no call's sentences appear in two splits.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=lc["seed"])
    train_idx, rest_idx = next(outer.split(labeled, groups=labeled["call_id"]))
    train, rest = labeled.iloc[train_idx], labeled.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=lc["seed"])
    val_idx, test_idx = next(inner.split(rest, groups=rest["call_id"]))
    splits = {"train": train, "val": rest.iloc[val_idx], "test": rest.iloc[test_idx]}
    for name, df in splits.items():
        df.reset_index(drop=True).to_parquet(paths["processed"] / f"{name}.parquet", index=False)

    print(f"{len(sents)} management sentences; weak labels: "
          f"{sents['weak_label'].value_counts(dropna=False).to_dict()}")
    for name, df in splits.items():
        print(f"  {name}: {len(df)} sentences, {df['call_id'].nunique()} calls, "
              f"{df['weak_label'].value_counts().to_dict()}")
    return sents
