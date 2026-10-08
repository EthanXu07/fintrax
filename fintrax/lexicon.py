"""Loughran-McDonald word lists plus earnings-call phrases -> confidence weak labels.

The LM dictionary (free for academic use, not redistributed here) is fetched
on first use from the copy bundled in the ``pysentiment2`` wheel on PyPI:
its ``Uncertainty`` column flags uncertainty words and ``Modal`` grades
modal words 1=strong, 2=moderate, 3=weak.
"""
import io
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from fintrax.config import ROOT

LM_PATH = ROOT / "data" / "lexicon" / "LM.csv"
PYPI_JSON = "https://pypi.org/pypi/pysentiment2/0.1.1/json"

# Spoken-language cues the LM lists (built from 10-Ks) miss.
CERTAINTY_PHRASES = [
    "confident", "confidence in", "conviction", "committed to", "on track", "certainly",
    "absolutely", "no doubt", "very clear", "clear line of sight", "firmly", "fully expect",
    "we are well positioned", "we're well positioned", "remain on track", "without question",
    "very pleased", "extremely pleased", "really pleased", "we know", "we delivered", "we achieved",
    "we're confident", "we remain confident", "thrilled", "excited about", "great momentum",
    "proud of", "we have line of sight", "exactly what we", "we're seeing strong", "we are seeing strong",
]
HEDGE_PHRASES = [
    "hopefully", "we hope", "too early",
    "remains to be seen", "hard to say", "difficult to predict", "hard to predict", "not sure",
    "cautious", "we'll see", "we will see", "kind of", "sort of", "it depends", "a little bit",
    "visibility is limited", "limited visibility", "lack of visibility", "uncertain", "uncertainty",
    "not going to speculate", "don't want to speculate", "i don't know", "we don't know", "tough to",
]
# Conversational filler hedges only mildly; one alone doesn't make a sentence "low".
SOFT_HEDGES = ["we believe", "we think", "i think", "i believe", "i guess", "we feel"]
STRONG_WEIGHT = {"will": 0.5}   # "will" is everywhere in calls; count it half
MODERATE_WEIGHT = 0.5           # likely / probably / should / would hedge mildly
SOFT_WEIGHT = 0.5
# Approximators in front of numbers ("nearly 600,000") aren't hedges in spoken calls.
NOT_HEDGES = frozenset({"nearly", "almost", "approximately", "roughly", "approximate"})
LABELS = ["low", "neutral", "high"]
WORD_RE = re.compile(r"[a-z']+")


def fetch_lm(path: Path = LM_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = requests.get(PYPI_JSON, timeout=30).json()
    wheel = next(u["url"] for u in meta["urls"] if u["filename"].endswith(".whl"))
    with zipfile.ZipFile(io.BytesIO(requests.get(wheel, timeout=60).content)) as z:
        path.write_bytes(z.read("pysentiment2/static/LM.csv"))
    return path


@dataclass(frozen=True)
class Lexicon:
    strong: frozenset
    moderate: frozenset
    weak: frozenset
    uncertainty: frozenset

    def cue_words(self) -> frozenset:
        return self.strong | self.moderate | self.weak | self.uncertainty


@lru_cache
def load() -> Lexicon:
    df = pd.read_csv(fetch_lm(), usecols=["Word", "Uncertainty", "Modal"])
    df["Word"] = df["Word"].str.lower()
    words = lambda mask: frozenset(df.loc[mask, "Word"])  # noqa: E731
    weak = words(df["Modal"] == 3)
    return Lexicon(
        strong=words(df["Modal"] == 1),
        moderate=words(df["Modal"] == 2),
        weak=weak - NOT_HEDGES,
        # Weak modals are also flagged Uncertainty; keep them in one bucket.
        uncertainty=words(df["Uncertainty"] > 0) - weak - NOT_HEDGES,
    )


def score(sentence: str, lex: Lexicon | None = None) -> tuple[float, float]:
    """Return (certainty, hedging) cue weights for one sentence."""
    lex = lex or load()
    s = sentence.lower()
    tokens = WORD_RE.findall(s)
    certainty = sum(STRONG_WEIGHT.get(t, 1.0) for t in tokens if t in lex.strong)
    certainty += sum(s.count(p) for p in CERTAINTY_PHRASES)
    hedging = sum(1.0 for t in tokens if t in lex.weak or t in lex.uncertainty)
    hedging += sum(MODERATE_WEIGHT for t in tokens if t in lex.moderate)
    hedging += sum(s.count(p) for p in HEDGE_PHRASES)
    hedging += sum(SOFT_WEIGHT * s.count(p) for p in SOFT_HEDGES)
    return certainty, hedging


def weak_label(sentence: str, lex: Lexicon | None = None) -> str | None:
    """high / low when cues clearly point one way, neutral when there are none,
    None (dropped from training) when the cues are weak or mixed."""
    certainty, hedging = score(sentence, lex)
    net = certainty - hedging
    if certainty == 0 and hedging == 0:
        return "neutral"
    if net >= 1:
        return "high"
    if net <= -1:
        return "low"
    return None


def mask_cues(sentence: str, mask_token: str, lex: Lexicon | None = None) -> str:
    """Replace every lexicon cue with the mask token (for the leakage check)."""
    lex = lex or load()
    out = sentence
    for p in sorted(CERTAINTY_PHRASES + HEDGE_PHRASES + SOFT_HEDGES, key=len, reverse=True):
        out = re.sub(re.escape(p), mask_token, out, flags=re.I)
    cues = lex.cue_words()
    return re.sub(r"[A-Za-z']+", lambda m: mask_token if m.group(0).lower() in cues else m.group(0), out)
