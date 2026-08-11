"""Character n-gram inverted index — the *filter* half of filter-and-verify.

Turns "which sentences could possibly match this query?" from a scan of the
whole corpus into a couple of dictionary lookups.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_GRAM_SIZE = 3


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
        for gram in self._grams(normalized):
            self._postings.setdefault(gram, set()).add(sentence_id)

    def candidates(self, normalized_query: str) -> Iterable[int]:
        """Sentence ids that might match, including with one edit.

        The result is a superset. Anything that really matches is in it;
        plenty that does not will be too.
        """
        query_length = len(normalized_query)
        if query_length == 0:
            return set()

        # Both halves need to be at least one full n-gram long, or there is
        # nothing to look up. Below that the index cannot help at all and we
        # fall back to offering the whole corpus for verification. Correct,
        # slow, and rare: it only happens for queries the user has barely
        # started typing, and the matcher is fast per sentence.
        if query_length < 2 * self._gram_size:
            return set(self._all_ids)

        split = query_length // 2
        first_half = normalized_query[:split]
        second_half = normalized_query[split:]

        return self._containing(first_half) | self._containing(second_half)

    def _containing(self, substring: str) -> set[int]:
        """Superset of the sentences containing ``substring`` exactly."""
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

    def _grams(self, text: str) -> Iterable[str]:
        size = self._gram_size
        for i in range(len(text) - size + 1):
            yield text[i : i + size]
