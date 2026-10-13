# Fintrax: how to use the code, and why it is written the way it is

The [README](../README.md) says what Fintrax produces and how well it works. This document is the
companion for someone who has to run it, change it, or check it. Part 1 is a usage guide. Part 2
goes through every module block by block: what the block does, and why it is written that way.

Everything here was checked against the code as of October 2026. Where the reason for a design
choice is stated in a code comment or the README, it is repeated here; where it is not, the
explanation is what the code itself implies.

## Contents

1. [Using the code](#1-using-the-code)
2. [What lives where on disk](#2-what-lives-where-on-disk)
3. [Entry points: `__main__.py`, `cli.py`, `config.py`](#3-entry-points)
4. [`scrape.py`: download the transcripts](#4-scrapepy)
5. [`parse.py`: HTML to speaker turns](#5-parsepy)
6. [`lexicon.py`: weak labels](#6-lexiconpy)
7. [`dataset.py`: sentences and the training sample](#7-datasetpy)
8. [`train.py`: fine-tune FinBERT](#8-trainpy)
9. [`score.py`: score every sentence, aggregate per call](#9-scorepy)
10. [`signals.py`: confidence to BUY / HOLD / SELL](#10-signalspy)
11. [`report.py`: signal charts](#11-reportpy)
12. [`prices.py`: price cache](#12-pricespy)
13. [`backtest.py`: did the signals work?](#13-backtestpy)
14. [`inventory.py`: the long-only simulation](#14-inventorypy)
15. [`charts.py`: evaluation figures](#15-chartspy)
16. [Tests](#16-tests)
17. [Things that will surprise you](#17-things-that-will-surprise-you)

---

## 1. Using the code

### Setup

```sh
make setup        # creates .venv and installs requirements.txt
make test         # 32 tests, about 3 seconds, no network and no model needed
```

`make setup` runs `python3 -m venv .venv` and installs into it. Every other `make` target calls
`.venv/bin/python -m fintrax <stage>`, so you never need to activate the environment. To use a
different interpreter, pass `PY=...` (`make signals PY=python`).

### The stages

The pipeline is six stages that each read the previous stage's files and write their own. Run them
one at a time, or several in one command (`make parse label train`), or all six with `make all`.

| Command | Module | Reads | Writes | Notes |
|---|---|---|---|---|
| `make scrape` | `scrape.py` | fool.com sitemaps and pages | `data/raw/sitemaps/*.xml`, `data/raw/pages/**.html.gz`, `data/raw/index.jsonl` | About 10 h for the full archive. Safe to stop and restart. |
| `make parse` | `parse.py` | `data/raw/` | `data/interim/turns/part-*.parquet`, `data/interim/parse_stats.parquet` | Rebuilds from scratch each time. |
| `make label` | `dataset.py` | `data/interim/turns/` | `data/processed/sentences/part-*.parquet`, `train.parquet`, `val.parquet`, `test.parquet` | Downloads the Loughran-McDonald list on first use. |
| `make train` | `train.py` | the three split files | `models/fintrax-confidence/`, `results/metrics.json` | About 1 h on Apple Silicon. |
| `make score` | `score.py` | `data/processed/sentences/`, the model | `data/processed/scored/part-*.parquet`, `data/processed/calls.parquet` | About 4 h for ~14M sentences. Safe to stop and restart. |
| `make signals` | `signals.py` + `report.py` | `calls.parquet` | `results/signals.csv`, `results/latest_signals.csv`, PNG charts | Seconds. |
| `make backtest` | `backtest.py` + `inventory.py` + `charts.py` | `results/signals.csv`, Yahoo prices | `results/evaluation/` | Optional. Not part of `make all`. |

`make all` stops at `signals` on purpose: the signals are the product, and the backtest is an
evaluation appendix you run when you want it.

### What to rerun after a change

- **Changed `signals.buy_z`, `sell_z` or `min_history`:** only `make signals` (and `make backtest`
  if you want the evaluation updated). Nothing upstream depends on the thresholds.
- **Changed anything under `report:`:** `make signals`. The report is called from the signals stage.
- **Changed `inventory:` or `backtest:`:** only `make backtest`.
- **Changed the lexicon (`lexicon.py`) or `label:` settings:** `make label train score signals`.
  The weak labels change, so the model must be retrained and everything rescored.
- **Changed `train:` settings:** `make train score signals`. `score` notices the model is newer
  than its outputs and rescores everything by itself.
- **Changed the parser:** `make parse label train score signals`.
- **New transcripts published:** `make scrape parse label score signals`. Retraining is optional;
  `score` only scores sentence files that are new or changed.

### Scraping a subset

```sh
.venv/bin/python -m fintrax scrape --tickers AAPL MSFT NVDA
```

This keeps only those tickers. It still has to download every page in the configured month range,
because old transcript URLs do not contain the ticker; the ticker is read from the page. To make a
small run, narrow `scrape.start_month` / `end_month` in `config.yaml` as well.

### Reading the output

`results/signals.csv` has one row per call. The columns the README does not list are `company`,
`call_time_et` (24-hour Eastern time, blank if the page did not give one), `sentences` (how many
management sentences were scored) and `warmup` (true while there is not enough history for a
z-score; such calls are always HOLD).

### Configuration

Everything tunable is in [`config.yaml`](../config.yaml), and each key has a comment there. The
`paths:` block is relative to the repo root; `load_config` turns each entry into an absolute path
and creates the folder.

---

## 2. What lives where on disk

```
data/                         gitignored: transcripts are Motley Fool's copyright
  raw/
    sitemaps/2017-01.xml …    one cached sitemap per month
    pages/ab/abcdef….html.gz  one gzipped transcript body per URL, named by the URL's SHA-1
    index.jsonl               one line per URL attempted: status, ticker, title, file path
  interim/
    turns/part-0000.parquet   one row per speaker turn
    parse_stats.parquet       one row per page: kept or dropped, and why
  processed/
    sentences/part-*.parquet  every management sentence with its weak label
    train|val|test.parquet    the class-balanced labelled sample, split by call
    scored/part-*.parquet     every sentence with model probabilities
    calls.parquet             one row per call: confidence by section
  prices/                     one parquet per ticker, plus _missing.txt and _listed_later.txt
  lexicon/LM.csv              the Loughran-McDonald dictionary, fetched on first use
models/fintrax-confidence/    the fine-tuned model and tokenizer (gitignored)
results/                      signals.csv, charts, metrics.json, evaluation/
results/pilot/                the committed pilot outputs the README shows
```

---

## 3. Entry points

### `fintrax/__main__.py`

```python
from fintrax.cli import main

if __name__ == "__main__":  # guard: worker processes re-import this module on macOS
    main()
```

This is what makes `python -m fintrax` work. The `if __name__` guard is not decoration. `parse`
and `label` use `ProcessPoolExecutor`, and on macOS new worker processes are started by
re-importing the main module. Without the guard each worker would start the whole pipeline again.

### `fintrax/cli.py`

- `STAGES` is the ordered list that `all` expands to. `EXTRA = ["backtest"]` is kept separate so
  that `all` never runs it.
- `MODULES = {"label": "dataset"}` exists because the stage is called `label` but the file is
  `dataset.py`. Every other stage name is also its module name.
- Each stage is imported with `importlib.import_module` inside the loop, not at the top of the
  file. That keeps `python -m fintrax scrape` from importing PyTorch and transformers, which takes
  seconds and is not needed for scraping.
- `load_config` is imported inside `main` for the same reason: `--help` stays fast.
- Every stage module exposes `run(cfg)`. `scrape` is the one exception, taking `only=` for the
  `--tickers` filter.

### `fintrax/config.py`

`ROOT` is the repo root, found from the file's own location, so the commands work from any
directory. `load_config` reads the YAML, replaces each relative path with an absolute `Path`, and
creates the directory. Later code can then write files without checking that folders exist.

---

## 4. `scrape.py`

Goal: find every earnings-call transcript on fool.com and store just the transcript part of each
page, politely and resumably.

### Constants

- `SITEMAP_URL` is fool.com's monthly sitemap, which `robots.txt` advertises. Discovery goes
  through sitemaps instead of crawling listing pages because a sitemap gives every URL for a month
  in one request.
- `TRANSCRIPT_URL` is a regex that picks transcript URLs out of the sitemap XML and captures the
  year, month and day from the path. The XML is searched as text instead of being parsed, which is
  faster and does not care about the sitemap's namespace declarations.
- `QUARTER_RE` finds "Q3 2024" or "Q3 FY 2024" in a page title.

### `months(start, end)`

A generator of `(year, month)` pairs. Tuple comparison (`(y, m) <= (ey, em)`) does the date
arithmetic without a date library.

### `RateLimiter`

One limiter is shared by all worker threads, so `requests_per_second` is a total and not a
per-thread rate.

- `wait()` takes the lock, reserves the next free time slot (`next_at + interval`, or now if the
  limiter has been idle), releases the lock, and only then sleeps. Sleeping outside the lock lets
  other threads reserve their own slots meanwhile, so requests are spaced evenly.
- `penalize(seconds)` pushes the next slot into the future for everyone. It is used when the site
  signals throttling: one thread backing off while two others keep hammering would not help.
- `time.monotonic()` is used instead of wall-clock time so a clock adjustment cannot break pacing.

### `get(session, url, limiter, retries=4)`

Fetches one URL and sorts failures into three kinds, because they need different handling:

- **Network error:** sleep `5 × 2^attempt` seconds and retry.
- **404 or 410:** raise `FileNotFoundError`. The page is gone, and retrying is pointless.
- **403, 429 or 503:** the site is throttling. Penalize the shared limiter by `30 × 2^attempt`
  seconds, then back off and retry.

After four failures it raises `RuntimeError`, which the caller records as a retryable error.

### `discover(cfg, session, limiter)`

Walks the configured months and returns `{url: publish_date}`.

- Each month's sitemap is cached in `data/raw/sitemaps/`. A rerun reads the cache.
- The current month is always refetched, because its sitemap is still growing.
- A month with no sitemap (404) is skipped.
- `urls.setdefault` keeps the first date seen for a URL if it appears in more than one month.

### `extract(html)`

Returns the page's metadata and a minimal HTML document that contains only the transcript.

- The body is `#article-body-transcript`, or `.article-body` on older pages. If neither exists,
  the page is not a transcript and `ValueError` is raised.
- The ticker and company come from `<meta name="primary_tickers">`. The module docstring explains
  why: old URLs are truncated `.aspx` slugs with no ticker in them.
- Only the body is kept and rewrapped in a tiny HTML shell. The docstring gives the reason: about
  15 KB gzipped per page instead of about 500 KB, which matters at ~54,000 pages.

### `page_path` and `load_index`

- `page_path` names each file by the SHA-1 of its URL and puts it in a folder named by the first
  two hex characters. Hashing avoids illegal filename characters; the two-character folders keep
  any one directory from holding tens of thousands of files.
- `load_index` reads `index.jsonl` into a dict keyed by URL. A URL that was retried appears on
  several lines, and because later lines overwrite earlier ones in the dict, the latest status
  wins. That is what lets the index be append-only.

### `run(cfg, only=None)`

1. Build one `requests.Session` whose connection pool is as large as the worker count, so threads
   reuse connections.
2. Discover URLs, load the index, and build the to-do list: URLs never tried, plus URLs whose last
   status was `error`. Statuses `ok` and `missing` are final.
3. Sort the to-do list newest first, so an interrupted run has the most recent calls.
4. `fetch(url)` downloads, extracts, and writes one gzipped file, then records one of four
   statuses: `ok`, `missing` (404, or no transcript body), `error` (retries exhausted), or
   `skipped` (ticker not in `--tickers`).
5. Index lines are written under a lock and flushed immediately. If the process is killed, every
   completed page is already recorded, and the next run resumes where this one stopped.
6. `skipped` pages are deliberately not written to the index, so a later run without `--tickers`
   will fetch them.
7. Progress is printed every 100 pages with a rate and a time estimate.

Threads are used here (not processes) because the work is waiting on the network.

---

## 5. `parse.py`

Goal: turn each stored page into speaker turns labelled with section (prepared remarks or Q&A) and
speaker type (executive, analyst or operator). This is the most intricate module, because Motley
Fool has changed its page layout and the newer layout has no Q&A heading.

### The two body layouts

- **Legacy** (through about 2025): `<h2>Prepared Remarks:</h2>` and `<h2>Questions & Answers:</h2>`
  headings; each speaker is introduced by a paragraph that contains only a bold name and an italic
  role; the speaker's text follows in later paragraphs.
- **Current:** a "Call participants" list at the top in the form `Role - Name`, then a "Full
  Conference Call Transcript" section where each paragraph starts with a bold `Name:`. There is no
  Q&A heading, so the boundary has to be inferred.

The README counts three layouts because the 2017-18 pages also put the call date in a different
place; that is handled in `call_datetime`, not in a third body parser.

### Constants

- `QNA_CUE` matches phrases that hand the call over to questions ("first question",
  "question-and-answer session", "[Operator Instructions]" and so on). The comment says it is
  deliberately strict: an opening line such as "after the prepared remarks we'll open the call to
  questions" must not match, or the whole call would be labelled Q&A.
- `HANDOFF_WORDS = 80`: a turn shorter than this near the boundary is a hand-off ("we'll take our
  first question"), not real remarks.
- `MONOLOGUE_WORDS = 300`: nobody asks a 300-word question before Q&A starts, so an unlisted
  speaker who talks that long is management.
- `NOISE` matches whole paragraphs that are only a bracketed note or a "Duration:" line.

### Small helpers

- `clean` applies Unicode NFKC normalisation and collapses whitespace, so non-breaking spaces and
  typographic variants do not break later matching.
- `name_key` reduces a name to first and last word in lower case. "Amy E. Hood" and "Amy Hood"
  then match, which matters because the participant list and the transcript often spell a name
  differently.
- `looks_like_name` rejects bold text that is really a sentence (it contains a digit, or a
  sentence-ending punctuation mark followed by a space, or more than seven words). Current-layout
  pages sometimes bold a whole sentence, and without this check that sentence would become a
  "speaker".
- `speaker_type` decides executive, analyst or operator from the name and role. Anyone who is not
  the operator and whose role does not contain "analyst" is an executive. That default matters:
  it means unknown roles count as management, and the code compensates for it in `split_qna`.

### `call_datetime(soup)`

Returns the real call date and the call time. The docstring gives the reason this exists: the date
in the URL is the publish date, which is days after the call and, for republished transcripts,
years after. A backtest that entered trades on the publish date would be wrong.

- Newer pages have an element with id `date` (and sometimes `time`).
- 2017-18 pages do not, so the fallback looks in the first six paragraphs for one that mentions
  "call" and contains a date.
- `DATE_RE` accepts both "Feb. 8, 2018" and "February 8, 2018"; only the first three letters of
  the month are parsed, so both forms work with one format string.
- `TIME_RE` converts "4:30 p.m." to `16:30`. The backtest later uses this to decide whether a call
  was before the market opened.

### `parse_legacy(body)`

Walks the `<h2>` and `<p>` elements in order. Headings switch the current section, and "Call
participants" ends the transcript. A paragraph is a speaker line when its bold text is at the
start and the paragraph is either just the name or `Name -- Role`. Every other paragraph is
appended to the current speaker's turn. Paragraphs before the first heading are ignored, which
drops the page preamble.

### `lookup_role` and `parse_current(body)`

- The participant list is read into `{name_key: role}`. Entries are split on a dash with spaces
  around it; the last piece is the name and the rest is the role.
- `lookup_role` tries an exact first-plus-last match, then falls back to a last-name match if it
  is unique. The docstring's example is "Jen-Hsun Huang" in the list and "Jensen Huang" in the
  transcript.
- A speaker who is not in the participant list is provisionally an "Analyst". That is the safe
  guess for the Q&A part of the call, where most unlisted speakers are analysts.
- That guess is wrong for management the list left out, so two corrections follow. First, any
  provisional analyst who speaks 300 or more words before the first question is relabelled
  "Executive (unlisted)". Second, `split_qna` relabels the rest (see below).

### `split_qna(turns)`

Finds the turn where Q&A begins when the page gives no heading. The logic, in order:

1. Find `first_remarks`: the first non-operator turn of at least 150 words. Cues before it are
   ignored, because operator welcomes often contain "[Operator Instructions]" before anyone has
   said anything.
2. Find `cue`: the first later turn that matches `QNA_CUE`.
3. Find `analyst`: the first analyst turn at or after the cue.
4. Decide the boundary:
   - no cue at all: Q&A starts at the first analyst turn (or never, if there is none);
   - cue and a later analyst: if the turn just before that analyst is short (under 80 words) it is
     the hand-off and Q&A starts there; otherwise Q&A starts at the analyst;
   - cue but no analyst afterwards: if the cue turn is long, it was the end of a prepared remark
     ("with that, let's go to Q&A"), so Q&A starts at the next turn.
5. Anyone typed as an analyst who speaks *before* the boundary is relabelled "Investor Relations
   (unlisted)". Analysts do not speak during prepared remarks, so such a speaker is almost always
   the investor-relations host the participant list omitted.

The reason for all this care is that every later number depends on it. The model scores management
sentences only, and reports prepared and Q&A confidence separately. A wrong boundary or a wrong
speaker type puts analyst questions into the management score.

### `parse_page(html)` and `parse_record(args)`

- `parse_page` picks the layout by checking for a "Prepared Remarks" heading. A legacy page that
  has that heading but no Q&A heading is passed through `split_qna` as well.
- `parse_record` is the function each worker process runs. It counts executive words in each
  section and marks the call `ok` only if it has a date, a ticker, and at least
  `parse.min_section_words` (500) executive words in **both** sections. The config comment gives
  the reason for requiring both: some companies post prepared remarks separately and open the call
  straight into Q&A.
- It catches `ValueError`, `OSError` and `AttributeError` and records the error type, so one bad
  page cannot stop a 54,000-page run.

### `run(cfg)`

- Old part files are deleted first, so the output never mixes two parser versions.
- Records are sorted oldest publish date first. Combined with the `seen` set keyed on
  `(ticker, call_date)`, this means that when a transcript was republished under a new URL, the
  original is kept and the copy is marked `duplicate`.
- `pool.map(..., chunksize=32)` returns results in input order, which the duplicate rule relies on.
- Rows are buffered and written as a new parquet part whenever the buffer passes 200,000 rows.
  The buffer is only checked after a whole call has been added, so **a call is never split across
  two part files**. `score.py` depends on that.
- `parse_stats.parquet` records every page with the reason it was dropped, and the summary printed
  at the end shows the counts per reason.

---

## 6. `lexicon.py`

Goal: assign each sentence a weak label (low, neutral or high confidence) from word lists, since no
hand-labelled data exists for this task.

### Getting the dictionary

`fetch_lm` downloads the Loughran-McDonald CSV the first time it is needed and caches it at
`data/lexicon/LM.csv`. It takes the copy bundled in the `pysentiment2` wheel on PyPI: it asks PyPI
for the wheel's URL, downloads it, and reads the CSV out of the zip in memory. The docstring gives
the reason the file is not committed: the dictionary is free for academic use but not
redistributable.

### The word lists

- `CERTAINTY_PHRASES`, `HEDGE_PHRASES` and `SOFT_HEDGES` are hand-written. The comment explains
  why they are needed: the LM lists were built from 10-K filings and miss spoken-language cues
  like "on track" or "we'll see".
- `STRONG_WEIGHT = {"will": 0.5}`: "will" is a strong modal in LM, but it appears in nearly every
  sentence of an earnings call, so it counts half.
- `MODERATE_WEIGHT` and `SOFT_WEIGHT` are 0.5: words like "should" and filler like "I think" hedge
  only mildly, and one of them alone should not make a sentence "low".
- `NOT_HEDGES` removes "nearly", "almost", "approximately" and similar from the hedge lists. In a
  call these words sit in front of numbers ("nearly 600,000 units") and are not hedging.

### `Lexicon` and `load()`

`Lexicon` is a frozen dataclass of four word sets. `load()` is wrapped in `lru_cache`, so the CSV
is read once per process. Weak modals are also flagged as uncertainty words in LM, so `load`
subtracts the weak set from the uncertainty set to avoid counting such a word twice.

### `score(sentence)` and `weak_label(sentence)`

`score` returns two numbers, certainty and hedging, by adding up the weights of the cues found.
Single words are matched on tokens; phrases are matched as substrings of the lower-cased sentence.

`weak_label` turns them into a label:

- no cues at all: `neutral`;
- certainty minus hedging is at least 1: `high`;
- at most -1: `low`;
- anything in between: `None`, and the sentence is left out of training.

Dropping the mixed sentences is the point of the last rule. A sentence with one soft hedge, or
with one certainty cue and one hedge, has no clear label, and training on it would teach the model
noise.

### `mask_cues(sentence, mask_token)`

Replaces every cue with a token (or with nothing). It handles phrases first, longest first, so
that "clear line of sight" is removed whole before any shorter phrase inside it could match. This
function exists only for the leakage check in `train.py`.

---

## 7. `dataset.py`

Goal: split turns into sentences, label every management sentence, and draw a class-balanced
sample split into train, validation and test sets.

### Regular expressions

- `SENT_SPLIT` splits after `.`, `!` or `?` when followed by whitespace and then a capital letter,
  quote, bracket, dollar sign or digit. The lookahead stops it splitting on abbreviations followed
  by a lower-case word.
- `BOILERPLATE` matches safe-harbor language and call logistics ("forward-looking", "non-GAAP",
  "webcast", "turn the call over", "thank you"). The comment gives the reason: this is scripted
  legal and procedural text, not management tone, and it would dilute the score.
- `INLINE_NOISE` removes bracketed notes such as "[inaudible]" before splitting.

### `to_sentences(turns, min_tokens)`

Keeps executive turns only, builds `call_id` as `TICKER_DATE`, explodes each turn into sentences,
and drops sentences shorter than `label.min_tokens` (6 words) or matching the boilerplate pattern.
Very short sentences ("Yes.", "Thanks, John.") carry no confidence signal.

### `split_of(call_id)`

Assigns each call to train, val or test by hashing its id (MD5, modulo 100) into a 70/15/15 split.
Two properties matter:

- The split is **by call**, not by sentence. Sentences from one call share speakers and phrasing,
  so splitting by sentence would leak between train and test.
- It is **deterministic and stable**. A call always lands in the same split, even after new calls
  are scraped, so test results stay comparable across runs.

### `label_part(args)` and `run(cfg)`

- `label_part` processes one turn file into one sentence file. It runs in a process pool, one
  part per task, so the full archive is never in memory at once (the module docstring states this
  as the design goal).
- `run` then draws the labelled sample. `low` and `high` keep their natural ratio. `neutral`,
  which dominates, is cut down to `neutral_ratio` times the average of the other two. Everything
  is then scaled so the total does not exceed `label.max_sentences`.
- Sampling is done with a per-class keep probability (`rng.random(len(df)) < frac`), one part at
  a time. This means the sample size is approximate, but it needs no second pass and never loads
  all sentences.
- The seed comes from `label.seed`, so the sample is reproducible.

---

## 8. `train.py`

Goal: fine-tune FinBERT into a three-class confidence classifier, then measure it.

### `SentenceDataset`

Tokenises all sentences once in the constructor, with truncation but **without padding**. Padding
is added per batch by the Trainer's collator, so each batch is only as long as its longest
sentence.

### `WeightedTrainer`

Overrides `compute_loss` to use cross-entropy weighted by inverse class frequency
(`total / (3 × count)`). The sample is already roughly balanced, so this is a small correction for
whatever imbalance remains between low and high.

### `run(cfg)`: model setup

- The base model is `ProsusAI/finbert`, a BERT already adapted to financial text.
- `model.classifier.reset_parameters()` reinitialises the classification head. The comment gives
  the reason: FinBERT's head predicts sentiment (positive, negative, neutral), which is a
  different task. Starting the confidence head from those weights would bias it.
- `id2label` and `label2id` are stored in the model config, so the saved model carries its own
  label names and `score.py` never has to assume an order.

### `run(cfg)`: training arguments

Each non-default argument has a reason:

- `train_sampling_strategy="group_by_length"`: batches hold sentences of similar length, so less
  padding is processed.
- `bf16=True`: about 1.5 times faster on Apple Silicon, per the config comment.
- `dataloader_pin_memory=False`: pinning is unsupported on MPS and would only print warnings.
- `save_strategy="no"`: with one epoch there is nothing to resume, and checkpoints of a
  110M-parameter model are large.
- `warmup_steps=0.1`: in transformers v5 a value below 1 means a fraction of total steps.
- `report_to="none"`: no experiment-tracking service is contacted.

There is no evaluation during training. The validation split is scored once afterwards and
reported; it is not used for early stopping or model selection.

### `evaluate(scorer, df)`

Scores a split through the same `Scorer` class that `score.py` uses, not through
`Trainer.predict`. The docstring gives the reason: with length-grouped sampling, the Trainer's
predictions can come back in a different order from the labels. Using the production inference
path also means the reported accuracy is the accuracy of the code that actually produces signals.

### `context_check(scorer, test)`

This is the most important measurement in the file. Test accuracy against weak labels only shows
that the model reproduces the lexicon. `context_check` asks whether it learned anything else:

1. Take the test sentences labelled low or high.
2. Delete every lexicon cue from them.
3. Score the stripped sentences and compute the AUC of `P(high) − P(low)` for separating the
   originally-high from the originally-low.

An AUC of 0.5 would mean the model knows nothing beyond the word lists. Neutral sentences are
excluded because they have no cues to delete and would pass trivially. The README's pilot figure
for this is 0.67.

---

## 9. `score.py`

Goal: run the model over every management sentence and aggregate to one row per call.

### `device()` and `Scorer`

- `device()` prefers Apple's MPS, then CUDA, then CPU.
- `Scorer` loads a model in half precision (fp16) when a GPU is available. Inference does not
  need full precision, and half precision roughly halves memory and time.
- `label2id` is read from the model's own config and lower-cased. The same class can therefore
  score with the fine-tuned model (low, neutral, high) and with the original FinBERT (positive,
  negative, neutral) without any hard-coded index.
- `__call__` sorts the texts by length before batching, so each batch pads very little, and then
  writes results back by original index (`probs[idx] = ...`) so the output order matches the input
  order. This is the property `train.evaluate` relies on.
- The logits are converted to float32 before the softmax, because a softmax in fp16 loses
  precision.
- `@torch.inference_mode()` turns off gradient tracking.

### `call_features(sents)`

Groups sentences by call and section and computes, per section: mean confidence
(`P(high) − P(low)`), the share of sentences predicted low and high, mean sentiment, and the
sentence count. `unstack("section")` turns the two sections into columns (`conf_prepared`,
`conf_qna`, and so on). Calls missing either section are dropped. `gap` is Q&A confidence minus
prepared confidence.

### `run(cfg)`

- **Resumable.** A sentence part is rescored only if its output is missing, or older than either
  the sentence file or the model's `config.json`. So an interrupted run continues, and retraining
  the model automatically invalidates old scores.
- The models are only loaded if there is something to score.
- `score.sentiment` in the config controls whether FinBERT's original sentiment is also scored.
  It is off by default because it doubles the scoring time; the column is then all NaN.
- `speaker`, `role` and `turn_id` are dropped from the scored files to keep them small.
- Per-call features are computed part by part and concatenated. This is only correct because
  `parse.py` never splits a call across parts.
- `delta_qoq` (change in Q&A confidence from the company's previous call) is written to
  `calls.parquet` but nothing downstream reads it. `signals.py` computes its own
  `confidence_change` on overall confidence.

---

## 10. `signals.py`

Goal: turn each call's confidence into a z-score against earlier calls, and that into a signal.

### `weighted(calls, prepared, qna)`

The sentence-weighted mean of a per-section statistic. A call's overall confidence is therefore
the mean over all its management sentences, not the average of two section averages. A call with
20 prepared sentences and 200 Q&A sentences is dominated by its Q&A, as it should be.

### `expanding_z(df, col, min_history)`

Computes, for each call, its z-score against every call with a strictly earlier date.

- It groups by date and keeps running sums of the count, the values and the squared values.
  Shifting those cumulative sums by one date gives the totals for all earlier dates. Mean and
  variance follow from the three sums. The docstring's point is that this is O(n); recomputing
  the history for each call would be O(n²), which matters at 50,000 calls.
- The shift is by **date**, not by row. Two calls on the same day do not see each other. The test
  `test_expanding_z_uses_only_earlier_dates` checks exactly this. Without it a call could be
  compared against a call that happened later the same afternoon.
- The z-score is NaN when fewer than `min_history` earlier calls exist or the variance is not
  positive.

No future data is ever used: each call is compared only with what was known before it.

### `make_signals(calls, sc)`

- `np.select` assigns BUY above `buy_z`, SELL below `sell_z`, and HOLD otherwise. A NaN z-score
  fails both comparisons, so warm-up calls come out as HOLD, and `warmup` is set true for them.
- `gap`, `pct_high`, `pct_low`, `sentences` and `confidence_change` are computed for context. They
  do not affect the signal; `test_gap_is_context_only` checks that.
- `confidence_change` is the difference from the same ticker's previous call. It is the
  within-company view the README recommends, since the z-score itself compares across companies.

### `run(cfg)`

Writes `results/signals.csv` with a fixed column order (`OUTPUT_COLS`) rounded to four decimals,
prints the signal counts, and calls `report.run`. The report is imported inside the function so
that importing `signals` in a test does not pull in matplotlib.

---

## 11. `report.py`

Goal: charts that show what the model says, call by call. There is no profit-and-loss here; that
belongs to the backtest.

- `matplotlib.use("Agg")` selects a backend that draws to files and needs no display, so the
  stage runs on a headless machine.
- `latest(sig, days)` returns the BUY and SELL calls from the last `report.latest_days` days,
  strongest first. "Last" is measured from the newest call in the data, not from today's date, so
  an old dataset still produces a report.
- `latest_chart` draws them as a horizontal bar chart with the thresholds as dashed lines.
- `pick_tickers` chooses up to nine tickers to chart: the configured `report.chart_tickers` that
  have at least three live calls, then the tickers with the most calls. The fallback is why a
  small pilot run still gets charts.
- `ticker_panels` draws one stock: price on a log axis with an arrow at each BUY and SELL call,
  and the confidence z-score underneath with the BUY and SELL zones shaded. The log axis makes
  equal percentage moves look equal over a multi-year span.
- `ticker_charts` writes one PNG per ticker and one 3×3 grid. The price download is wrapped in a
  `try`: prices are context only, and a Yahoo outage should not stop the signals from being
  reported.
- `confidence_index` plots mean confidence by quarter and the share of BUY and SELL calls.
  Quarters with fewer than 20 calls are dropped and left as gaps, so the line does not bridge
  missing data.
- `distribution` shows the histogram of z-scores split by signal, and a scatter of prepared
  against Q&A confidence.

---

## 12. `prices.py`

Goal: daily open, close and volume per ticker from Yahoo Finance, cached on disk.

- `yahoo_symbol` converts share-class notation (`BRK.B` to `BRK-B`), because Motley Fool and Yahoo
  write it differently.
- Each ticker is cached as its own parquet file. `_missing.txt` remembers tickers Yahoo has no
  data for, so reruns do not ask again for thousands of delisted names.
- `covered(t)` checks that a cached file starts early enough for the requested window. If an
  earlier backtest cached a shorter history, the ticker is fetched again.
- `_listed_later.txt` handles the case that would otherwise defeat that check: a company that
  listed after the requested start date. Its history really does start late, so it is marked and
  not refetched every run.
- Downloads go in batches of 100 with three attempts each. The `for … else: continue` skips a
  batch that failed all three times without marking its tickers as missing, so they are tried
  again on the next run.
- `auto_adjust=True` gives prices adjusted for splits and dividends, which is what a return
  calculation needs.

---

## 13. `backtest.py`

Goal: check how the signals played out. It writes to `results/evaluation/`.

### `entry_index(px, call_date, premarket)`

Finds the first market open after the call. `searchsorted` with `side="left"` returns the call
day itself (for a call before 9:30 ET); `side="right"` returns the next trading day. Entering at
the open after the call is the earliest price a reader of the transcript could actually trade at.
A weekend call falls through to Monday either way.

### `event_returns(sig, px, bench, horizons, bt)`

For each signal, computes the stock's return from the entry open to the close `h` trading days
later, and the same for SPY, and their difference. Calls are skipped when:

- the ticker has no prices or fewer than 30 days of them;
- there are fewer than 20 trading days before entry (needed for the volume average);
- the entry price is below `min_price` or the 20-day average dollar volume is below
  `min_dollar_volume`. The comment gives the reason: penny stocks and illiquid names dominate
  averages with moves nobody could have traded;
- SPY has no bar on the entry day.

A horizon is filled in only if both the stock and SPY have enough later days, so recent calls
simply lack the long horizons.

### `winsorize`, `clustered_spread`, `quarter_quintiles`

- `winsorize` clips the top and bottom 1% of returns, so one extreme stock does not decide the
  average.
- `clustered_spread` is how significance is measured. It averages the long group and the short
  group within each month, takes the difference per month, and runs a one-sample t-test across
  months. The module docstring gives the reason: calls in the same earnings season are not
  independent, so treating each call as a separate observation would overstate the t-statistic.
- `quarter_quintiles` ranks a feature into five groups *within each calendar quarter*. Ranking
  within the quarter removes any drift in average tone over the years, so quintile 5 always means
  "among the most confident calls of that season". Quarters with fewer than 25 calls are skipped.

### `summarize` and `run`

`summarize` builds, for each horizon: mean, median and hit rate per signal; the clustered BUY
minus SELL spread; quintile means and the top-minus-bottom spread; the Spearman rank correlation
of each feature with the return; and the same spreads per year. A horizon with fewer than 30
observations is left out.

`run` loads the signals (excluding warm-up calls), downloads prices starting 60 days before the
first call, and writes the event table, the trade log, the daily book, the open inventory, the
summary JSON and three charts.

---

## 14. `inventory.py`

Goal: a simple, honest simulation of acting on the signals, with no fixed holding period.

### The rules

- BUY buys one lot of `lot_dollars` at the entry open, up to `max_lots` per stock.
- SELL sells every lot held in that stock. The simulation is long-only: a SELL with nothing held
  does nothing.
- HOLD does nothing.

### The SPY shadow

Every lot records `spy_shares`: the SPY you could have bought with the same dollars on the same
day. When the lot is sold, the shadow SPY is sold too. "Excess" therefore means "against putting
the same money in the index over the same days", which is a fair comparison for a strategy whose
capital at work varies over time.

### `run_inventory(events, px, bench, cfg)`

Replays the signals in date order and returns the trade log and the lots still open.

- Every signal gets a row, including ones that caused no trade, with a reason ("at max lots",
  "nothing held", "hold"). The log then explains itself.
- After the replay, any stock whose price history ends more than a week before the benchmark's
  last day is treated as delisted or acquired, and its lots are closed at the last close.
  Otherwise those lots would sit in the book forever at a stale price.
- `lots_after` is a running count of lots per ticker, added for readability.

### `daily_book(trades, px, bench)`

Marks the position to market every day.

- Share counts are the cumulative sum of the trade log's share changes. They are clipped at zero
  to remove floating-point dust left after a full exit.
- `pnl` is market value plus cumulative cash flow. Because buys are negative cash and sells
  positive, this one sum is total profit, realised and unrealised together.
  `test_book_reconciles_with_trade_log` checks that it agrees with the trade log.
- The daily return is the change in P&L divided by yesterday's market value plus today's
  purchases. That denominator is the capital actually at work that day, which is the right base
  when the invested amount keeps changing.

### `return_stats`, `holdings`, `summarize`

`return_stats` gives annualised return, volatility, Sharpe ratio and maximum drawdown from a daily
return series. `holdings` lists what is still held, marked at the latest close, next to what the
SPY shadow of those lots is worth. `summarize` assembles the numbers the README quotes: lots
bought and sold, win rate, realised and unrealised P&L, the SPY shadow's P&L, and a per-year
breakdown.

---

## 15. `charts.py`

Three figures for `results/evaluation/`, built from the summary dictionary and the daily book:

- `signal_returns.png`: mean excess return after BUY, HOLD and SELL at each horizon, with the
  clustered spread and t-statistic written under each group.
- `confidence_quintiles.png`: mean excess return by confidence quintile for each horizon.
- `inventory.png`: cumulative P&L of the inventory against its SPY shadow, and returns by year.

The charts are kept apart from `report.py` because they depend on prices and the backtest, and the
signal report must work without either.

---

## 16. Tests

`make test` runs 32 tests in about 3 seconds. They use synthetic fixtures, so no transcripts, model
or network are needed.

| File | What it protects |
|---|---|
| `tests/test_parse.py` | Both layouts parse; Q&A starts at the first question; roles and continuation paragraphs are right; the call date comes from the page and not the URL; the 2018 header-paragraph date works; name matching ignores middle initials; the last-name fallback works. |
| `tests/test_lexicon.py` | Example sentences get the expected label; a single soft hedge is dropped and not labelled low; `mask_cues` removes every cue; boilerplate and analyst turns are excluded; bracketed noise is stripped. |
| `tests/test_signals.py` | The z-score uses only earlier dates, and same-day calls do not see each other; confidence is sentence-weighted; signals follow the thresholds; the gap does not affect the signal; `confidence_change` is per company. |
| `tests/test_backtest.py` | Entry is the same day for pre-market calls and the next day otherwise; returns are excess over the benchmark; penny and illiquid stocks are skipped; the spread averages months first; quintiles are ranked within the quarter; the inventory rules (exit everything on SELL, never short, stop at the lot cap, close delisted stocks); the SPY shadow matches when the stock tracks the index; the daily book reconciles with the trade log. |

The two HTML files in `tests/fixtures/` are hand-written stand-ins for the two layouts, which is
why `.gitignore` allows them while excluding every other `.html` file.

---

## 17. Things that will surprise you

- **`--tickers` does not make scraping fast.** Every page in the month range is still downloaded,
  because the ticker is only known after the page is read. Narrow the month range too.
- **`parse` and `label` delete their previous output** before writing. `scrape` and `score` do
  not; they resume.
- **Phrase cues are substring matches.** "uncertain" is counted inside "uncertainty", and
  "confident" inside "confidently", so some words score more than once. This affects the weak
  labels the model is trained on.
- **The validation split is reported, not used.** Nothing is tuned on it.
- **The z-score compares each call with all companies, not with its own history.** The README's
  "Known issues" covers the consequence: companies that always sound upbeat lean BUY.
- **`delta_qoq` in `calls.parquet` is not used by anything.**
- **Warm-up calls are HOLD.** They are in `signals.csv` with `warmup` true, and the backtest
  excludes them.
- **The pilot numbers in the README come from 202 calls.** They will change when the full archive
  is processed.
