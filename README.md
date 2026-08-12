# Autocomplete Engine

A Google-style sentence autocomplete engine. Point it at a corpus of English
text files (nested folders, one sentence per line) and, as a user types a
partial string, it returns the 5 best completions — each with the matched
sentence, the source file it came from, the line offset within that file, and
a score.

A match doesn't have to be at the start of a sentence: the query can align
anywhere in the line, and it still counts as a match with at most one
correction — a single substituted, inserted, or deleted character.

The system runs in two phases. An **offline** phase reads the corpus once and
builds a search index. An **online** phase is an interactive CLI: the user
types text, presses Enter, and sees the five best suggestions; typing `#` ends
the current sentence and resets the session.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python -m autocomplete.cli.main
```

The first run builds the index from `Archive.zip` and caches it to
`index_store/index.pkl`; later runs load that file and start immediately.

Requires **Python 3.10+** (`X | Y` union syntax, `@dataclass(slots=True)`).

## Running the program

```bash
# default: build from Archive.zip in the repo root, then start the session
python -m autocomplete.cli.main

# use any corpus -- a .zip archive or an already-extracted directory
python -m autocomplete.cli.main --corpus tests/fixtures/mini_corpus
python -m autocomplete.cli.main --corpus C:\data\my_corpus

# rebuild instead of reusing the cached index
python -m autocomplete.cli.main --rebuild

# cache the index somewhere else
python -m autocomplete.cli.main --index build/index.pkl
```

Example session:

```
Loading the files and preparing the system...
  17 lines read, 16 unique sentences kept, 1 duplicates dropped, 6 source files
  built in 0.0s -> index_store/index.pkl
The system is ready. Enter your text:
this is
Here are 3 suggestions
1. This is the recommended way to accept binary data. (docs/python_notes.txt 4)
2. this is a nested guide file. (docs/guides/deep_guide.txt 1)
3. this is the last line without newline (docs/guides/deep_guide.txt 3)
this is
```

Typing continues from where you stopped: entering `this is`, then ` a nested`,
searches for `this is a nested`. Entering `#` clears the accumulated text.
Ctrl-C or Ctrl-D exits.

## Running the tests

```bash
python -m pytest tests/ -q                 # everything (119 tests, ~8s)
python -m pytest tests/ -q -m "not slow"   # unit tests only (106 tests, <1s)
python -m pytest tests/ -v                 # verbose
```

Two kinds of test, deliberately kept apart:

- **Unit tests** run against `tests/fixtures/mini_corpus`, a 17-line tree built
  to contain the cases that break readers — a file with no trailing newline,
  blank lines mid-file, tabs, non-ASCII, files three levels deep, and a
  duplicated sentence. Small enough that every expected value is known by hand.
- **Integration tests** (`test_real_corpus_integration.py`, marked `slow`) run
  the real pipeline over `Archive.zip` and assert only what must hold for any
  corpus: the index never drops a true match, results cite a real file and
  line, a typo still matches at a lower score, punctuation survives the round
  trip. Queries are *derived from the corpus itself* rather than hard-coded, so
  they don't silently stop testing anything when the corpus changes. They skip
  automatically if `Archive.zip` is absent.

## How the corpus is stored

The offline phase (`application/builder.py`) streams every line once and keeps
four parallel lists plus a table of interned paths:

```
BuiltCorpus
├── normalized[]    lowercased, punctuation-stripped text (what we search)
├── raw_text[]      the original line, verbatim (what we display)
├── path_ids[]      small int index into paths[]
├── line_numbers[]  1-based line number within the source file
└── paths[]         each distinct file path stored exactly once
```

Three decisions, all forced by measurements on the real corpus:

- **Deduplication.** About 35% of normalized lines are exact duplicates of
  another line. The first occurrence wins and keeps its real path and line
  number; the rest are dropped. A user could not distinguish them anyway.
- **Path interning (Flyweight).** There are ~1,500 distinct paths and millions
  of sentences. A `path_id` int per sentence plus one `paths` list costs a
  fraction of storing the path string on every record.
- **Streaming.** `CorpusReader.read()` is a generator and the builder consumes
  it lazily, so the raw corpus is never held in memory alongside the index
  being built. Measured: 92 KB resident after the first record.

The built corpus and index are pickled together (protocol 5) via
`infrastructure/storage/pickle_store.py`, written to a temporary file and
renamed on success so an interrupted save cannot leave a corrupt index.

## Known limitation: index memory

The full `Archive.zip` (1,504 files, ~2.58M non-blank lines) does not yet fit
in memory. Measured on a 400k-line prefix:

| Component | Memory |
|---|---|
| n-gram postings (`dict[str, set[int]]`) | **506.7 MB** |
| all four corpus lists + strings | ~50 MB |
| **bytes per posting** | **56.9** |

The postings are 86% of the footprint. Growth is linear at roughly 2.7 GB per
million unique sentences, extrapolating to about 4.2 GB for the full 1.55M —
enough to be OOM-killed on a normal machine.

The fix is to store each posting list as a sorted `array("i")` instead of a
Python `set[int]`. Prototyped on the same data: **506.7 MB → 37.7 MB**, or 4.2
bytes per posting instead of 56.9 — a 13× reduction, putting the full corpus at
roughly 350 MB. Set intersection becomes a sorted merge; the `SentenceIndex`
interface is unchanged, so nothing outside `ngram_index.py` moves. This is also
closer to the representation the Phase B C++ backend will use.

Until then, smaller corpora and the test fixture run fine.

## Architecture

Layered hexagonal (Ports & Adapters):

- **`domain/`** — pure logic, zero I/O: models, the normalization pipeline, the
  one-edit matcher, the scorer, the ranker. Depends only on the standard
  library and on `domain/ports.py`, the frozen `Protocol` interfaces
  (`Normalizer`, `Matcher`, `Scorer`, `SentenceIndex`, `IndexStore`,
  `CorpusReader`) the rest of the system is built against.
- **`infrastructure/`** — adapters implementing those protocols against the
  real world: corpus reading (zip or directory), the n-gram candidate index,
  pickle-backed storage.
- **`application/`** — `AutoCompleteEngine`, whose single method
  `get_best_k_completions` is what the project is graded on, plus
  `CorpusBuilder`, the offline phase.
- **`cli/`** — `session.py` holds the loop logic and knows nothing about
  `input()`/`print()`, so it is testable by feeding it strings; `main.py` is the
  thin layer that connects it to a terminal.

Query path: **normalize → filter (index) → verify (matcher) → score → rank**.
The index returns a *superset* of possible matches cheaply; the matcher does
the real check. That split is what keeps queries fast, and it is the seam the
Phase B C++ backend replaces.

## Project structure

```
autocomplete/
├── config.py                     # corpus/index paths and defaults
├── domain/                       # models, normalizer, matcher, scorer, ranker, ports
├── infrastructure/
│   ├── corpus_reader.py          # streams a .zip or a directory tree
│   ├── index/ngram_index.py      # 3-gram inverted index, pigeonhole filter
│   └── storage/pickle_store.py   # save/load behind the IndexStore port
├── application/
│   ├── engine.py                 # the facade
│   └── builder.py                # the offline build
└── cli/                          # session logic + entry point
tests/
├── fixtures/mini_corpus/         # 17-line corpus tree for unit tests
└── test_real_corpus_integration.py   # slow tests against Archive.zip
benchmarks/                       # candidate-matching performance scripts
```

Empty by design, for later phases: `index/cpp_index.py` and `index/_native/`
(Phase B, C++), `storage/protobuf_store.py` and `schema.proto` (Phase B),
`llm/` (Phase C, Gemini), `application/factory.py` (Phase B, backend
selection).

## Corpus

A tree of `.txt` files at arbitrary depth, one sentence per line.
`CorpusReader` accepts either a `.zip` archive or an extracted directory and
streams from both, so no manual unzip step is needed. Zip detection is by
content, not file extension.

`Archive.zip` (33 MB) is committed at the repo root for convenience. Note this
permanently adds its size to the repository history; it could be untracked and
supplied locally instead.

## Contributors

Ofek, Shadi, Orel