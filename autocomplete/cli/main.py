"""Entry point: builds the index if needed, then runs the interactive loop.

    python -m autocomplete.cli.main                  # build (or reuse) and run
    python -m autocomplete.cli.main --rebuild        # force a fresh build
    python -m autocomplete.cli.main --corpus path/   # use another corpus

This is the only file that talks to the terminal. Everything it wires together
arrives through constructors, so swapping the index for the Phase B C++ one, or
the store for Protobuf, is a change here and nowhere else.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from autocomplete import config
from autocomplete.application.builder import BuiltCorpus, CorpusBuilder
from autocomplete.application.engine import AutoCompleteEngine
from autocomplete.cli.session import RESET_TOKEN, Session
from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.domain.normalizer import DEFAULT_PIPELINE
from autocomplete.domain.ranker import TopKRanker
from autocomplete.domain.scorer import RuleBasedScorer
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from autocomplete.infrastructure.storage.pickle_store import PickleStore


def build_index(corpus_path: Path, index_path: Path) -> tuple[BuiltCorpus, NGramIndex]:
    """Run the offline phase and persist the result."""
    print("Loading the files and preparing the system...")
    started = time.perf_counter()

    index = NGramIndex()
    builder = CorpusBuilder(normalizer=DEFAULT_PIPELINE, index=index)
    corpus, stats = builder.build(CorpusReader().read(corpus_path))

    print(f"  {stats}")
    PickleStore().save((corpus, index), index_path)
    print(f"  built in {time.perf_counter() - started:.1f}s -> {index_path}")
    return corpus, index


def load_index(index_path: Path) -> tuple[BuiltCorpus, NGramIndex]:
    """Reuse a previously built index."""
    print("Loading the files and preparing the system...")
    started = time.perf_counter()
    corpus, index = PickleStore().load(index_path)
    print(f"  loaded {len(corpus):,} sentences in {time.perf_counter() - started:.1f}s")
    return corpus, index


def run(session: Session) -> None:
    """Read lines from the terminal until EOF or Ctrl-C."""
    print("The system is ready. Enter your text:")
    while True:
        try:
            text = input()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        session.submit(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sentence autocomplete.")
    parser.add_argument("--corpus", type=Path, default=config.DEFAULT_CORPUS,
                        help="zip archive or directory of .txt files")
    parser.add_argument("--index", type=Path, default=config.DEFAULT_INDEX_PATH,
                        help="where the built index is cached")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild even if a cached index exists")
    args = parser.parse_args(argv)

    if args.rebuild or not args.index.exists():
        if not args.corpus.exists():
            print(f"corpus not found: {args.corpus}", file=sys.stderr)
            return 1
        corpus, index = build_index(args.corpus, args.index)
    else:
        corpus, index = load_index(args.index)

    engine = AutoCompleteEngine(
        normalizer=DEFAULT_PIPELINE,
        index=index,
        matcher=OneEditMatcher(RuleBasedScorer()),
        scorer=RuleBasedScorer(),
        corpus=corpus,
        ranker=TopKRanker(),
    )
    run(Session(engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())