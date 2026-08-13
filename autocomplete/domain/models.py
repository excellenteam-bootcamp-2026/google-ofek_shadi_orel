"""Value object for a single autocomplete suggestion.

`domain/ports.py` is empty in this repository as of this writing, so there
was no frozen field list to conform to here. The four fields below come
from the illustrative shape given in the team's Phase A plan (section 5);
see the agent report for this flagged discrepancy.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=False)
class AutoCompleteData:
    """A single ranked autocomplete result.

    Ordering, equality and hashing are all defined explicitly on this class
    (rather than left to the matcher/ranker) so that every caller in the
    codebase -- `sorted()`, `heapq.nlargest()`, `set()` dedup -- gets the
    same, single, correct behaviour for free. Splitting "how results
    compare" across multiple files would let the sort rule and the data it
    sorts drift out of sync.
    """

    completed_sentence: str
    source_text: str
    offset: int
    score: int

    def __lt__(self, other: object) -> bool:
        """Rank "better" as greater, matching Python's normal ordering:
        higher score wins; equal scores break alphabetically, with the
        alphabetically-earlier sentence counted as better.

        `heapq.nlargest(n, results)` picks the n *greatest* elements under
        `__lt__` -- it does not know "better" means "higher score", it only
        knows "greater". So this must define greater-is-better, not
        lesser-is-better: a `<` that instead sorted best-first would make
        `sorted()` look right by eye while silently making
        `heapq.nlargest()` return the worst results first. Encoding the
        real ordering once here, rather than a "looks right" inverted one,
        is what lets every caller -- `sorted()`, `sorted(reverse=True)`,
        `heapq.nlargest()` -- get the correct top-5 ordering for free.

        True duplicates -- same score AND same sentence text, e.g. the same
        line repeated across many corpus files -- fall through to a
        deterministic tie-break on source_text then offset. Without this,
        which results "win" among identical duplicates depends on Python's
        set/dict iteration order, which varies per process run
        (PYTHONHASHSEED) -- the same query against the same corpus could
        return a different top-5 every time it's run.
        """
        if not isinstance(other, AutoCompleteData):
            return NotImplemented
        if self.score != other.score:
            return self.score < other.score
        if self.completed_sentence != other.completed_sentence:
            return self.completed_sentence > other.completed_sentence
        if self.source_text != other.source_text:
            return self.source_text < other.source_text
        return self.offset < other.offset

    def __eq__(self, other: object) -> bool:
        """Two results are equal only if all four fields match.

        A weaker equality (e.g. by `completed_sentence` alone) would
        collapse genuinely different matches -- the same sentence text
        found at two different offsets, or the same line matched via two
        different edits with two different scores -- into "duplicates"
        during dedup, silently dropping real results.
        """
        if not isinstance(other, AutoCompleteData):
            return NotImplemented
        return (
            self.completed_sentence == other.completed_sentence
            and self.source_text == other.source_text
            and self.offset == other.offset
            and self.score == other.score
        )

    def __hash__(self) -> int:
        """Hash matches `__eq__` field-for-field, so results can live in a
        `set`/dict key for deduplication across multiple match paths.
        """
        return hash(
            (self.completed_sentence, self.source_text, self.offset, self.score)
        )

    def __str__(self) -> str:
        """One-line, human-readable form for CLI display."""
        return f"{self.completed_sentence}  ({self.source_text}:{self.offset})"
