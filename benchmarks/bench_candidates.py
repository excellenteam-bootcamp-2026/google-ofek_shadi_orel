"""Measures how well the index filters, and how long each stage takes.

Two numbers matter here, and they pull against each other:

* **survival rate** — what fraction of the corpus the index still hands to
  the matcher. Lower is better; this is the whole point of having an index.
* **verify time** — how long the matcher then spends on those survivors.

Wednesday's C++ work is judged against whatever this prints today, so run it
before changing anything and keep the output.

Run it from the repository root::

    py benchmarks/bench_candidates.py Archive.zip
    py benchmarks/bench_candidates.py path/to/corpus/dir
    py benchmarks/bench_candidates.py            # small built-in corpus
"""

from __future__ import annotations

import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autocomplete.domain.matcher import OneEditMatcher  # noqa: E402
from autocomplete.domain.ports import MatchKind, MatchResult  # noqa: E402
from autocomplete.infrastructure.index.ngram_index import NGramIndex  # noqa: E402

_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")


def simple_normalize(text: str) -> str:
    """Stand-in for the real normalizer, which is not this track's file.

    Swap this for ``domain/normalizer.py`` once it lands. Until then the
    absolute timings here are only comparable against other runs of this same
    script — which is all Wednesday's before/after table needs.
    """
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub("", text.lower())).strip()


class BenchScorer:
    """Copy of the appendix scoring rules, so the bench can rank alignments."""

    _SUBSTITUTION = (5, 4, 3, 2)
    _INDEL = (10, 8, 6, 4)

    def score(self, result: MatchResult) -> int:
        base = 2 * result.matched_chars
        if result.kind is MatchKind.EXACT:
            return base
        if result.kind is MatchKind.SUBSTITUTION:
            table, tail = self._SUBSTITUTION, 1
        else:
            table, tail = self._INDEL, 2
        position = result.error_position
        return base - (table[position - 1] if position <= 4 else tail)


BUILT_IN_CORPUS = [
    "The event loop is the core of every asyncio application.",
    "The B-tree index is the core of every relational database.",
    "At the core of every TCP connection is a three-way handshake.",
    "A hash table provides average constant time lookup.",
    "The resolver caches every response for the duration of its TTL.",
]

DEFAULT_QUERIES = [
    "the core",
    "the core of every",
    "the core off every",
    "the cor of every",
    "hash tabl",
    "resolver caches",
    "zzz nothing here zzz",
]


def load_lines(source: Path | None) -> list[str]:
    if source is None:
        return BUILT_IN_CORPUS
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            return [
                line
                for name in archive.namelist()
                if not name.endswith("/")
                for line in archive.read(name)
                .decode("utf-8", errors="replace")
                .splitlines()
            ]
    return [
        line
        for path in source.rglob("*.txt")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]


def build(raw_lines: list[str]) -> tuple[NGramIndex, list[str]]:
    normalized = [simple_normalize(line) for line in raw_lines]
    normalized = [line for line in normalized if line]

    started = time.perf_counter()
    index = NGramIndex()
    for sentence_id, sentence in enumerate(normalized):
        index.add(sentence_id, sentence)
    elapsed = time.perf_counter() - started

    print(f"corpus     : {len(normalized):,} sentences")
    print(f"build time : {elapsed:.2f}s")
    print()
    return index, normalized


def measure(index: NGramIndex, sentences: list[str], queries: list[str]) -> None:
    matcher = OneEditMatcher(BenchScorer())
    total = len(sentences)

    header = f"{'query':<24}{'cands':>9}{'survive':>9}{'hits':>7}{'filter':>10}{'verify':>10}"
    print(header)
    print("-" * len(header))

    for raw_query in queries:
        query = simple_normalize(raw_query)

        started = time.perf_counter()
        candidates = list(index.candidates(query))
        filter_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        hits = sum(
            1
            for sentence_id in candidates
            if matcher.match(query, sentences[sentence_id]) is not None
        )
        verify_ms = (time.perf_counter() - started) * 1000

        survival = len(candidates) / total * 100 if total else 0.0
        print(
            f"{raw_query[:23]:<24}{len(candidates):>9,}{survival:>8.1f}%"
            f"{hits:>7,}{filter_ms:>9.2f}ms{verify_ms:>9.2f}ms"
        )


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if source is not None and not source.exists():
        raise SystemExit(f"no such corpus: {source}")

    index, sentences = build(load_lines(source))
    measure(index, sentences, DEFAULT_QUERIES)


if __name__ == "__main__":
    main()