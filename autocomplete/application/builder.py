"""Offline phase: turn a corpus of raw text into a queryable, saveable index.

This is the whole offline half of the assignment in one class. It reads every
line once, normalizes it, throws away what can never match, and hands the
survivors to the index.

Three memory decisions are baked in, all of them forced by the real corpus
(1,504 files, 2.58M non-blank lines, 96M normalized characters):

* **Deduplication.** 35% of normalized lines are exact duplicates of another
  line. Indexing all of them would cost a third more memory for results a user
  can never distinguish. The first occurrence wins and keeps its real path and
  line number.
* **Path interning (Flyweight).** There are ~1,500 distinct paths and millions
  of sentences. Storing the path string on every record would dominate memory;
  storing a small int index into a list of paths does not.
* **Streaming.** The reader is a generator and is consumed lazily, so the raw
  corpus is never held in memory alongside the index being built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from autocomplete.domain.ports import Normalizer, RawLine, SentenceIndex


@dataclass
class BuiltCorpus:
    """Sentence store produced by the builder and consumed by the engine.

    Satisfies the ``Corpus`` protocol the engine depends on: ``get`` maps a
    sentence id back to its normalized text and the original line it came
    from. Paths are interned, so a record costs one str plus two ints rather
    than two strs.
    """

    normalized: list[str] = field(default_factory=list)
    raw_text: list[str] = field(default_factory=list)
    path_ids: list[int] = field(default_factory=list)
    line_numbers: list[int] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def get(self, sentence_id: int) -> tuple[str, RawLine]:
        """Return ``(normalized_sentence, original_line)`` for one sentence."""
        return (
            self.normalized[sentence_id],
            RawLine(
                self.raw_text[sentence_id],
                self.paths[self.path_ids[sentence_id]],
                self.line_numbers[sentence_id],
            ),
        )

    def __len__(self) -> int:
        return len(self.normalized)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        """Yield ``(sentence_id, normalized)`` — what an index needs to ingest."""
        return enumerate(self.normalized)


@dataclass(frozen=True)
class BuildStats:
    """What the build actually did. Printed by the CLI, asserted by tests."""

    lines_read: int
    sentences_kept: int
    duplicates_dropped: int
    distinct_paths: int

    def __str__(self) -> str:
        return (
            f"{self.lines_read:,} lines read, "
            f"{self.sentences_kept:,} unique sentences kept, "
            f"{self.duplicates_dropped:,} duplicates dropped, "
            f"{self.distinct_paths:,} source files"
        )


class CorpusBuilder:
    """Builds a :class:`BuiltCorpus` (and fills an index) from raw lines.

    Collaborators arrive through the constructor as ``ports.py`` protocols,
    never as concrete classes, so the same builder works with the Python
    n-gram index today and a C++ index in Phase B.
    """

    def __init__(self, normalizer: Normalizer, index: SentenceIndex) -> None:
        self._normalizer = normalizer
        self._index = index

    def build(self, lines: Iterable[RawLine | tuple[str, str, int]]) -> tuple[BuiltCorpus, BuildStats]:
        """Consume raw lines and return the corpus plus what the build did.

        ``lines`` is consumed lazily; it is expected to be the generator from
        ``CorpusReader.read``, not a materialized list.
        """
        corpus = BuiltCorpus()
        normalize = self._normalizer.normalize
        index_add = self._index.add

        seen: dict[str, int] = {}          # normalized text -> sentence_id
        path_ids: dict[str, int] = {}      # path string -> index into corpus.paths

        lines_read = 0
        duplicates = 0

        for raw_text, path, line_no in lines:
            lines_read += 1

            normalized = normalize(raw_text)
            if not normalized:
                continue                    # cannot ever match a query

            if normalized in seen:
                duplicates += 1
                continue                    # first occurrence keeps the citation

            path_id = path_ids.get(path)
            if path_id is None:
                path_id = len(corpus.paths)
                path_ids[path] = path_id
                corpus.paths.append(path)

            sentence_id = len(corpus.normalized)
            seen[normalized] = sentence_id

            corpus.normalized.append(normalized)
            corpus.raw_text.append(raw_text)
            corpus.path_ids.append(path_id)
            corpus.line_numbers.append(line_no)

            index_add(sentence_id, normalized)

        stats = BuildStats(
            lines_read=lines_read,
            sentences_kept=len(corpus.normalized),
            duplicates_dropped=duplicates,
            distinct_paths=len(corpus.paths),
        )
        return corpus, stats