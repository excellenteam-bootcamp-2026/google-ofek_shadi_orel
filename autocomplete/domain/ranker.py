"""Ranking adapter: turns a pool of scored results into the top-k output.

Pure post-processing over `AutoCompleteData` -- no I/O, no matching, no
scoring. Depends only on `AutoCompleteData.__lt__` for its notion of
"best", so if the tie-break rule in `domain/models.py` ever changes, this
file does not need to.
"""
from __future__ import annotations

import heapq
from typing import Iterable

from autocomplete.domain.models import AutoCompleteData


class TopKRanker:
    """Deduplicates, then returns the k best `AutoCompleteData`.

    Dedup runs before ranking, not after: two candidates that resolve to
    the exact same result (same sentence, source, offset and score) can
    legitimately arrive via two different match paths -- e.g. two
    overlapping n-gram candidates pointing at the same line. Without
    dedup first, a single true result could occupy more than one of the
    k output slots and silently push out a distinct one.
    """

    def rank(self, results: Iterable[AutoCompleteData], k: int = 5) -> list[AutoCompleteData]:
        deduped = set(results)
        return heapq.nlargest(k, deduped)
