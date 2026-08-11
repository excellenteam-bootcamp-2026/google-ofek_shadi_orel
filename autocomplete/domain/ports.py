"""Frozen interface contract for the autocomplete engine.

Depends only on the standard library. Every string crossing any interface
here is assumed already normalized identically by whichever Normalizer
implementation is wired in — see Normalizer below. That assumption is the
single highest-risk seam in the whole system; do not let it drift.

This file is FROZEN as of today. Changes go through the group, not a
single branch.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable


class RawLine(NamedTuple):
    """One line as found on disk, before any normalization."""
    text: str
    path: str
    line_no: int


class MatchKind(Enum):
    EXACT = auto()
    SUBSTITUTION = auto()
    INSERTION = auto()
    DELETION = auto()


@dataclass(frozen=True)
class MatchResult:
    """Describes HOW a query aligns to a sentence with at most one edit.

    Carries no score on purpose — turning this into a number is the Scorer's
    job, kept separate so the scoring formula can change (or move to C++ in
    Phase B) without touching match-finding logic at all.
    """
    kind: MatchKind
    error_position: int   # 1-based position within the normalized query; 0 if EXACT
    matched_chars: int


@runtime_checkable
class Normalizer(Protocol):
    """Collapses irrelevant differences between two pieces of text.

    MUST be applied identically to corpus lines and to user queries.
    """
    def normalize(self, text: str) -> str: ...


@runtime_checkable
class Scorer(Protocol):
    """Converts a MatchResult into the integer score defined by the
    assignment's scoring rule. Pure function of its input — no search,
    no I/O. This is Shadi's file: domain/scorer.py.
    """
    def score(self, match: MatchResult) -> int: ...


@runtime_checkable
class Matcher(Protocol):
    """Finds the best (highest-scoring) single-edit alignment of a query
    inside a sentence, using an injected Scorer to compare candidates.
    Returns None if no alignment exists within one edit.

    This is Ofek's file: domain/matcher.py (OneEditMatcher).
    """
    def match(self, normalized_query: str, normalized_sentence: str) -> MatchResult | None: ...


@runtime_checkable
class SentenceIndex(Protocol):
    """Narrows the corpus to a short list of plausible sentences.

    candidates() returns a SUPERSET: every sentence that could match is in
    it, but not everything in it matches — verification is the Matcher's
    job. This filter-and-verify seam is what the C++ backend replaces in
    Phase B.

    This is Ofek's file: infrastructure/index/ngram_index.py (NGramIndex).
    """
    def add(self, sentence_id: int, normalized: str) -> None: ...
    def candidates(self, normalized_query: str) -> Iterable[int]: ...


@runtime_checkable
class IndexStore(Protocol):
    """Persists the built corpus so the offline phase runs only once.

    This is Orel's file: infrastructure/storage/pickle_store.py
    (protobuf_store.py in Phase B).
    """
    def save(self, corpus, path: Path) -> None: ...
    def load(self, path: Path): ...


@runtime_checkable
class CorpusReader(Protocol):
    """Walks the corpus tree at any depth, yielding raw lines.

    This is Orel's file: infrastructure/corpus_reader.py.
    """
    def read(self, root: Path) -> Iterator[RawLine]: ...