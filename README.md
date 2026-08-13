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

The first run builds the index from `Archive.zip` (~80s on the full corpus)
and caches it to `index_store/index.json`; later runs load that file and start
in a few seconds.

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
python -m autocomplete.cli.main --index build/index.json
```

Example session:

```
Loading the files and preparing the system...
  17 lines read, 16 unique sentences kept, 1 duplicates dropped, 6 source files
  built in 0.0s -> index_store/index.json
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
python -m pytest tests/ -q                 # everything (201 tests, ~11s)
python -m pytest tests/ -q -m "not slow"   # unit tests only (188 tests, ~2s)
python -m pytest tests/test_edge_cases.py -q -s   # edge cases with a log line each
```

Three kinds of test, deliberately kept apart:

- **Unit tests** run against `tests/fixtures/mini_corpus`, a 17-line tree built
  to contain the cases that break readers — a file with no trailing newline,
  blank lines mid-file, tabs, non-ASCII, files three levels deep, and a
  duplicated sentence. Small enough that every expected value is known by hand.
- **Edge cases** (`test_edge_cases.py`, 79 tests) cover adversarial input across
  every component: emoji and surrogate pairs, NUL and control characters, a
  100k-character token, invalid UTF-8 bytes, empty and garbage index files,
  negative line numbers, `k=0`, 500 tied exact matches. Each prints a labelled
  `[EDGE]` line under `-s`, so the run doubles as a readable report.
- **Integration tests** (`test_real_corpus_integration.py`, marked `slow`) run
  the real pipeline over `Archive.zip` and assert only what must hold for any
  corpus: the index never drops a true match, results cite a real file and
  line, a typo still matches at a lower score. Queries are *derived from the
  corpus itself* rather than hard-coded, so they don't silently stop testing
  anything when the corpus changes. They skip automatically if `Archive.zip`
  is absent.

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

- **Deduplication.** About 33% of normalized lines are exact duplicates of
  another line (841,127 of 2,583,991). The first occurrence wins and keeps its
  real path and line number; the rest are dropped. A user could not
  distinguish them anyway.
- **Path interning (Flyweight).** There are ~1,500 distinct paths and 1.55M
  sentences. A `path_id` int per sentence plus one `paths` list costs a
  fraction of storing the path string on every record.
- **Streaming.** `CorpusReader.read()` is a generator and the builder consumes
  it lazily, so the raw corpus is never held in memory alongside the index
  being built. Measured: 92 KB resident after the first record.

The built corpus and index are serialised together as JSON via
`infrastructure/storage/json_store.py`, written to a temporary file and renamed
on success so an interrupted save cannot leave a corrupt index.

### Why JSON, and what comes next

JSON needs more work here than a Python-native format: `NGramIndex._postings`
is `dict[str, set[int]]`, and JSON has no set type, so every posting set is
converted to a sorted list on save and back to a set on load. Sorting also
makes the file byte-identical across runs for the same corpus.

Measured on a 150k-line slice, JSON is about 4.4× slower to write and 1.4×
larger than `pickle`, though roughly 2× faster to read back. It is used anyway
because `json.load` cannot execute code embedded in the file — unlike
`pickle.load` — and because the output is human-readable, which is useful when
debugging a small corpus.

Note that **neither** JSON nor pickle solves the Phase B problem. Once the C++
backend has to read the same index, the data needs a language-neutral schema
both sides compile against; that is what `storage/schema.proto` and
`storage/protobuf_store.py` are reserved for. Both satisfy the same
`IndexStore` port, so swapping is a one-line change in `cli/main.py`.

## Performance

Query latency is a graded metric, so `application/engine.py` runs a two-pass
search rather than the obvious single pass.

An exact substring match scores `2 * len(query)` and nothing can beat it: every
edit costs points, and an insertion also lowers the base. So if k sentences
contain the query verbatim, they *are* the answer, and the edit-tolerant search
is provably wasted work. Pass 1 finds those cheaply — the index's
`exact_candidates` intersects on every n-gram instead of splitting the query in
half and unioning, and the check itself is Python's `in` operator. Pass 2, the
full one-edit alignment, runs only when fewer than k exact matches exist.

Two further measures: results are resolved to objects only *after* a match is
confirmed, and a 256-entry LRU cache makes the repeated queries a typing
session generates (`t`, `th`, `the`, …) free on the second pass.

Measured on a 600k-line corpus slice, before and after:

| Query | Single pass | Two-pass | Speedup |
|---|---|---|---|
| `the` | 5,688 ms | 29 ms | 196× |
| `to be` | 2,760 ms | 2.7 ms | 1035× |
| `this is` | 1,341 ms | 4.3 ms | 315× |
| `python object` | 163 ms | 0.6 ms | 295× |
| `a` | 4,810 ms | 95 ms | 51× |

Worst-case latency went from ~5.7 s to under 100 ms, with byte-identical
results on every query tested.

### Known limitation: index memory

The index is memory-heavy. Measured on a 400k-line prefix, the n-gram postings
(`dict[str, set[int]]`) account for 506.7 MB of roughly 560 MB total — about
57 bytes per posting, because a Python `set` of ints is a sparse hash table of
boxed integers. Growth is linear at roughly 2.7 GB per million unique
sentences, so the full 1.55M-sentence corpus needs on the order of 4 GB. It
builds and runs on a machine with enough RAM, but there is no headroom.

Storing each posting list as a sorted `array("i")` instead was prototyped on
the same data: **506.7 MB → 37.7 MB**, or 4.2 bytes per posting — a 13×
reduction, putting the full corpus in the region of 350 MB. Set intersection
becomes a sorted merge; the `SentenceIndex` interface is unchanged, so nothing
outside `ngram_index.py` moves. This is also closer to the representation the
Phase B C++ backend will use.

## Architecture

Layered hexagonal (Ports & Adapters), with dependencies pointing inward:

```
cli/  ──→  application/  ──→  domain/  ←──  infrastructure/
```

- **`domain/`** — pure logic, zero I/O: models, the normalization pipeline, the
  one-edit matcher, the scorer, the ranker. Every module here imports only the
  standard library and other `domain/` modules — verifiable with
  `grep "^from autocomplete\." autocomplete/domain/*.py`, which returns only
  `autocomplete.domain.*`. `domain/ports.py` holds the frozen `Protocol`
  interfaces (`Normalizer`, `Matcher`, `Scorer`, `SentenceIndex`, `IndexStore`,
  `CorpusReader`) the rest of the system is built against.
- **`infrastructure/`** — adapters implementing those protocols against the
  real world: corpus reading (zip or directory), the n-gram candidate index,
  JSON-backed storage.
- **`application/`** — `AutoCompleteEngine`, whose single method
  `get_best_k_completions` is what the project is graded on, plus
  `CorpusBuilder`, the offline phase.
- **`cli/`** — `session.py` holds the loop logic and knows nothing about
  `input()`/`print()`, so it is testable by feeding it strings; `main.py` is the
  thin layer that connects it to a terminal and wires every collaborator
  together.

Query path: **normalize → filter (index) → verify (matcher) → score → rank**.
The index returns a *superset* of possible matches cheaply; the matcher does
the real check. That split is what keeps queries fast, and it is the seam the
Phase B C++ backend replaces.

Constants live in the layer that owns them: `DEFAULT_SUGGESTION_COUNT` is a
spec rule and lives in `domain/ranker.py`, while `config.py` holds only what
varies by environment (corpus and index paths). A domain module importing from
`config.py` would point the dependency outward and break the layering.

## Project structure

```
autocomplete/
├── config.py                     # corpus/index paths (environment settings only)
├── domain/                       # models, normalizer, matcher, scorer, ranker, ports
├── infrastructure/
│   ├── corpus_reader.py          # streams a .zip or a directory tree
│   ├── index/ngram_index.py      # 3-gram inverted index, pigeonhole filter
│   └── storage/json_store.py     # save/load behind the IndexStore port
├── application/
│   ├── engine.py                 # the facade + two-pass search
│   └── builder.py                # the offline build
└── cli/                          # session logic + entry point
tests/
├── fixtures/mini_corpus/         # 17-line corpus tree for unit tests
├── test_edge_cases.py            # 79 adversarial cases, logged
└── test_real_corpus_integration.py   # slow tests against Archive.zip
benchmarks/                       # candidate-matching performance scripts
```

Empty by design, for later phases: `index/cpp_index.py` and `index/_native/`
(Phase B, C++), `storage/protobuf_store.py` and `storage/schema.proto`
(Phase B), `llm/` (Phase C, Gemini), `application/factory.py` (Phase B,
backend selection).

## Corpus

A tree of `.txt` files at arbitrary depth, one sentence per line.
`CorpusReader` accepts either a `.zip` archive or an extracted directory and
streams from both, so no manual unzip step is needed. Zip detection is by
content, not file extension.

Note that the two input modes read files in different orders (`os.walk`
descends after listing a directory's own files; a zip is read in archive
order). The set of lines is identical either way, but because deduplication
keeps the first occurrence, a line appearing in two files can be attributed to
a different source depending on which mode built the index. Both citations are
true; the choice is just not stable across modes. See
`test_69b_dedup_citation_depends_on_read_order`.

`Archive.zip` (33 MB) is committed at the repo root for convenience. Note this
permanently adds its size to the repository history; it could be untracked and
supplied locally instead.

## Contributors

Ofek Revach, Shadi Younis, Orel Zabriko
