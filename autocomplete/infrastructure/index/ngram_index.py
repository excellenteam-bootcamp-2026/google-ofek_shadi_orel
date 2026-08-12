"""Character n-gram inverted index — the *filter* half of filter-and-verify.

Turns "which sentences could possibly match this query?" from a scan of the
whole corpus into a couple of dictionary lookups.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_GRAM_SIZE = 3

# Shortest half the pigeonhole split will bother producing. A single
# character appears in almost every sentence, so a half that short filters
# nothing while still costing a full union to compute.
MIN_USEFUL_HALF = 2


class NGramIndex:
    """Maps every n-character window in the corpus to the sentences holding it.

    Two ideas do all the work here.

    **The index answers exact substring questions.** A sentence containing
    some substring must contain every n-gram of that substring, so
    intersecting those posting lists gives a superset of the sentences that
    contain it. A superset, not the answer: ``"cor"`` and ``"ore"`` can both
    appear in a sentence that never contains ``"core"``. The matcher does the
    real check.

    **The pigeonhole principle extends that to typos.** Only one edit is
    allowed, so if the query is split in half the edit can land in at most one
    half — the other half must appear in the sentence exactly as typed. Look
    up both halves, union the results, and every possible match is in there.

    Splitting in the middle is not required for correctness; any split would
    do. It is chosen because we cannot know in advance where the user will
    slip, and an even split is the one that keeps the surviving half as long
    as possible in the worst case.
    """

    def __init__(self, gram_size: int = DEFAULT_GRAM_SIZE) -> None:
        self._gram_size = gram_size
        self._postings: dict[str, set[int]] = {}
        self._all_ids: set[int] = set()
        self._gramless_ids: set[int] = set()

    @property
    def gram_size(self) -> int:
        return self._gram_size

    def __len__(self) -> int:
        return len(self._all_ids)

    def add(self, sentence_id: int, normalized: str) -> None:
        """Register one normalized sentence.

        Each distinct n-gram of the sentence is stored once, no matter how
        often it repeats — the postings answer "which sentences", never "how
        many times".
        """
        self._all_ids.add(sentence_id)
        if len(normalized) < self._gram_size:
            # Too short to produce a single gram, so no posting list will ever
            # mention it. Kept aside so short-query lookups can add it back.
            self._gramless_ids.add(sentence_id)
            return
        for gram in self._grams(normalized):
            self._postings.setdefault(gram, set()).add(sentence_id)

    def exact_candidates(self, normalized_query: str) -> Iterable[int]:
        """Sentence ids that might contain the query **exactly**, no edit.

        The fast path. An exact match scores 2 x the query length, which is
        the highest score any alignment can reach, so once five of these are
        confirmed the answer is settled and the edit-tolerant search below is
        wasted work.

        Far more selective than :meth:`candidates` because it constrains on
        every gram of the query at once instead of on half of them, and
        intersects instead of unioning. It is also usable on much shorter
        queries — see :meth:`_containing`.
        """
        if not normalized_query:
            return set()
        return self._containing(normalized_query)

    def candidates(self, normalized_query: str) -> Iterable[int]:
        """Sentence ids that might match, including with one edit.

        The result is a superset. Anything that really matches is in it;
        plenty that does not will be too.
        """
        query_length = len(normalized_query)
        if query_length == 0:
            return set()

        # A one-character half constrains almost nothing — nearly every
        # sentence contains any given letter — so splitting this far costs a
        # large union and returns roughly the whole corpus anyway. Hand the
        # corpus over directly instead; it is the same answer, computed for
        # free. In practice these queries never get here: the exact fast path
        # settles them, because a string this short has matches everywhere.
        if query_length < 2 * MIN_USEFUL_HALF:
            return set(self._all_ids)

        split = query_length // 2
        first_half = normalized_query[:split]
        second_half = normalized_query[split:]

        return self._containing(first_half) | self._containing(second_half)

    def _containing(self, substring: str) -> set[int]:
        """Superset of the sentences containing ``substring`` exactly."""
        if len(substring) < self._gram_size:
            return self._containing_short(substring)

        grams = set(self._grams(substring))
        if not grams:
            return set()

        postings: list[set[int]] = []
        for gram in grams:
            posting = self._postings.get(gram)
            if not posting:
                # This n-gram is nowhere in the corpus, so neither is the
                # substring that contains it.
                return set()
            postings.append(posting)

        # Start from the rarest n-gram: the first intersection is then as
        # small as it can be, and every later one only shrinks it further.
        postings.sort(key=len)
        survivors = set(postings[0])
        for posting in postings[1:]:
            survivors &= posting
            if not survivors:
                break
        return survivors

    def _containing_short(self, substring: str) -> set[int]:
        """Same question, for a substring too short to have a gram of its own.

        A sentence containing a one- or two-character string must contain
        some gram that has that string inside it, so the answer is the union
        of the postings of every gram in the index that does. Scanning the
        gram keys is cheap — there are tens of thousands of them, not
        millions — and this needs no second index and no extra memory.

        Sentences shorter than one gram have no postings at all, so they are
        added back explicitly.
        """
        if not substring:
            return set(self._all_ids)

        survivors = set(self._gramless_ids)
        for gram, posting in self._postings.items():
            if substring in gram:
                survivors |= posting
        return survivors

    def _grams(self, text: str) -> Iterable[str]:
        size = self._gram_size
        for i in range(len(text) - size + 1):
            yield text[i : i + size]
