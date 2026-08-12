# Autocomplete Engine

A Google-style sentence autocomplete engine. Point it at a corpus of English
text files (nested folders, one sentence per line) and, as a user types a
partial string, it returns the 5 best completions — each with the matched
sentence, the source file it came from, the line offset within that file,
and a score. A match doesn't have to be at the start of a sentence: the
query can align anywhere in the line (start, middle, or end), and it still
counts as a match with at most one correction — a single substituted,
inserted, or deleted character.

The system runs in two phases. An **offline** phase reads the corpus once
and builds a search index. An **online** phase is an interactive CLI: the
user types text and gets live suggestions after every keystroke, and `#`
ends the current sentence and resets the session back to its initial state.

## Status

The core domain logic — normalization, one-edit matching, scoring, ranking,
and the `AutoCompleteEngine` facade that wires them together — is complete
and tested end-to-end against fakes, along with a working n-gram candidate
index (`infrastructure/index/ngram_index.py`). The offline build step
(`application/builder.py`), the pickle-based index store
(`infrastructure/storage/pickle_store.py`), the corpus file reader
(`infrastructure/corpus_reader.py`), and the interactive CLI
(`cli/main.py`, `cli/session.py`) are still empty placeholders — there is
currently no way to run the tool end-to-end from the command line. A few
other pieces (`config.py`, `application/factory.py`,
`infrastructure/storage/protobuf_store.py`,
`infrastructure/index/cpp_index.py`, `infrastructure/llm/`) are likewise
unimplemented; per `.gitignore`, the C++ index and Gemini/LLM integration
are planned for later stages, not Phase A.

## Architecture

The codebase follows a layered hexagonal (Ports & Adapters) style:

- **`domain/`** — pure logic with zero I/O: data models, the normalization
  pipeline, the one-edit matcher, the scorer, and the ranker. Everything
  here depends only on the standard library and on `domain/ports.py`, the
  frozen set of `Protocol` interfaces (`Normalizer`, `Matcher`, `Scorer`,
  `SentenceIndex`, `IndexStore`, `CorpusReader`) that the rest of the
  system is built against.
- **`infrastructure/`** — adapters that implement those protocols against
  the real world: reading corpus files off disk, the n-gram candidate
  index, and (eventually) persisted index storage, a C++-backed index, and
  an LLM provider.
- **`application/`** — wires domain and infrastructure together behind
  `AutoCompleteEngine`, the single method (`get_best_k_completions`) the
  project is graded on, plus the (not-yet-implemented) offline builder.
- **`cli/`** — the interactive, user-facing REPL.

The point of the split is that `domain/` can be fully unit-tested against
fakes without touching a file system, and any adapter — the index, the
storage backend, even a future C++ or LLM-backed implementation — can be
swapped in without changing the engine or the tests around it.

## Setup

This project targets **Python 3.10+** (the domain models use
`@dataclass(slots=True)`, and `domain/ports.py` relies on `X | Y` union
syntax in type annotations).

```bash
git clone <repo-url>
cd google-ofek_shadi_orel
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` currently only pins `pytest`: every implemented module
under `autocomplete/` imports nothing beyond the standard library.

## Running the tests

```bash
pytest
```

As a snapshot at the time of writing, this passes **66 tests, 0 failures**.
That count will grow as the remaining modules are implemented.

## Running the interactive demo

**Not runnable yet.** The offline builder (`application/builder.py`), the
corpus reader (`infrastructure/corpus_reader.py`), the index store
(`infrastructure/storage/pickle_store.py`), and the CLI itself
(`cli/main.py`, `cli/session.py`) are all still empty. Once those land,
this section will document the real build-then-run commands, arguments,
and an example session. Until then, the engine is only exercised through
`autocomplete.application.engine.AutoCompleteEngine` directly, as the tests
in `tests/` do.

## Project structure

```
autocomplete/
├── domain/           # pure logic: models, normalizer, matcher, scorer, ranker, ports
├── infrastructure/   # adapters: corpus reading, indexes, storage, LLM provider
├── application/       # wires domain + infrastructure behind AutoCompleteEngine
└── cli/               # interactive REPL (not yet implemented)
tests/                  # pytest suite, run against domain/application via fakes
benchmarks/             # standalone scripts for measuring candidate-matching performance
```

## Contributors

Ofek, Shadi, Orel
