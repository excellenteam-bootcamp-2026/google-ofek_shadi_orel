"""Frozen interface contract for the autocomplete engine.

Written together at the Tuesday kickoff and frozen. Changes require unanimous
agreement in the group chat.

Everything in ``infrastructure/`` and ``application/`` depends on this module.
This module depends on nothing but the standard library, which is what lets
the domain be tested without a corpus, a file system or a network.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable


class MatchKind(Enum):
    """Which single edit, if any, was needed to align the query to the sentence.

    The direction of INSERTION and DELETION is defined from the *query's*
    point of view. The Hebrew spec names the same two cases after the
    *correction* applied to the query, which is the mirror image of this — so
    "הוספת תו" in the spec is DELETION here, and "מחיקת תו" is INSERTION.
    The penalty table is identical for both, but ``matched_chars`` is not,
    so the direction has to be right.
    """

    EXACT = "exact"
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The single best alignment of a query against one sentence.

    This is the contract between the matcher (which finds *where* and *how*
    the query aligns) and the scorer (which turns that into a number). The
    matcher never computes a score; the scorer never looks at a string.

    Attributes:
        kind: Which edit was needed, if any.
        error_position: 1-based position of the edit **within the normalized
            query**; 0 when ``kind`` is EXACT. For INSERTION this is the index
            of the extra character the user typed. For DELETION it is the
            position where the missing character would have gone.
        matched_chars: How many query characters were consumed against
            sentence characters. The scorer's base score is exactly
            ``2 * matched_chars``.

            Given a normalized query of length L:

            ==============  ==============  =================================
            kind            matched_chars   why
            ==============  ==============  =================================
            EXACT           L               every character lands
            SUBSTITUTION    L               the wrong character still lands
            INSERTION       L - 1           the extra character lands on
                                            nothing
            DELETION        L               every typed character lands
            ==============  ==============  =================================

            SUBSTITUTION is the one row where the Hebrew body and the English
            appendix disagree. We follow the appendix. If the staff rules for
            the Hebrew body instead, the fix is to emit ``L - 1`` here for
            SUBSTITUTION and change nothing else.
    """

    kind: MatchKind
    error_position: int
    matched_chars: int


class RawLine(NamedTuple):
    """One line as it was found on disk, before any normalization."""

    text: str
    path: str
    line_no: int


@runtime_checkable
class Normalizer(Protocol):
    """Collapses the irrelevant differences between two pieces of text.

    Must be applied identically to corpus lines and to user queries, or every
    score in the system is subtly wrong.
    """

    def normalize(self, text: str) -> str: ...


@runtime_checkable
class Matcher(Protocol):
    """Decides whether a query aligns to a sentence with at most one edit."""

    def match(
        self, normalized_query: str, normalized_sentence: str
    ) -> MatchResult | None:
        """Return the best alignment, or None if more than one edit is needed.

        "Best" means highest-scoring. The alignment is searched at every
        position in the sentence, not just the start.
        """
        ...


@runtime_checkable
class Scorer(Protocol):
    """Turns an alignment into a number, per the appendix scoring rules."""

    def score(self, result: MatchResult) -> int: ...


@runtime_checkable
class SentenceIndex(Protocol):
    """Narrows the corpus down to a short list of plausible sentences.

    ``candidates`` returns a **superset**: every sentence that could match is
    in it, but not everything in it matches. Verification is the matcher's
    job. This filter-and-verify split is what keeps the engine fast, and it is
    also the seam the C++ backend replaces in Phase B.
    """

    def add(self, sentence_id: int, normalized: str) -> None: ...

    def candidates(self, normalized_query: str) -> Iterable[int]: ...


@runtime_checkable
class IndexStore(Protocol):
    """Persists the built corpus so the offline phase runs only once."""

    def save(self, corpus, path: Path) -> None: ...

    def load(self, path: Path): ...


@runtime_checkable
class CorpusReader(Protocol):
    """Walks the corpus directory tree at any depth, yielding raw lines."""

    def read(self, root: Path) -> Iterator[RawLine]: ...
