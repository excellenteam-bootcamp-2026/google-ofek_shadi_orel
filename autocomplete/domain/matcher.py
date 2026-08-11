"""Alignment of a query against a single sentence, allowing at most one edit.

This is the *verify* half of filter-and-verify. The index has already thrown
away almost the whole corpus; what arrives here is a short list of plausible
sentences that still have to be checked properly and turned into a score.
"""

from __future__ import annotations

from autocomplete.domain.ports import MatchKind, MatchResult, Scorer


class OneEditMatcher:
    """Finds the best-scoring alignment of a query inside a sentence.

    The query has to appear as a substring of the sentence, either exactly or
    after at most one edit. Any position in the sentence counts — the match
    does not have to start at the beginning.

    The matcher takes a :class:`~autocomplete.domain.ports.Scorer` rather than
    knowing the penalty tables itself. It has to compare alignments found at
    different positions to pick the best one, and duplicating the scoring
    rules here would give the project two sources of truth for the same
    formula.
    """

    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer

    def match(
        self, normalized_query: str, normalized_sentence: str
    ) -> MatchResult | None:
        """Return the highest-scoring alignment, or None if there isn't one.

        None means the query needs more than one edit to fit anywhere in this
        sentence, which per the spec is not a match at all — not a low score.
        """
        query_length = len(normalized_query)
        if query_length == 0:
            return None

        perfect_score = 2 * query_length
        best: MatchResult | None = None
        best_score = 0

        for start in range(len(normalized_sentence)):
            candidate = self._align_at(
                normalized_query, normalized_sentence, start
            )
            if candidate is None:
                continue

            candidate_score = self._scorer.score(candidate)
            if best is None or candidate_score > best_score:
                best, best_score = candidate, candidate_score

            if best_score == perfect_score:
                break

        return best

    def _align_at(
        self, query: str, sentence: str, start: int
    ) -> MatchResult | None:
        """Best alignment that begins at ``sentence[start]``, or None.

        Walks forward while the characters agree. If it runs out of query,
        the match is exact. Otherwise the first disagreement is the *only*
        place the single edit can be: everything before it already matched
        character for character, so spending the edit earlier would be
        possible but never cheaper — the penalty tables only get smaller as
        the position grows.
        """
        query_length = len(query)
        sentence_length = len(sentence)

        query_i, sentence_i = 0, start
        while (
            query_i < query_length
            and sentence_i < sentence_length
            and query[query_i] == sentence[sentence_i]
        ):
            query_i += 1
            sentence_i += 1

        if query_i == query_length:
            return MatchResult(MatchKind.EXACT, 0, query_length)

        error_position = query_i + 1

        # The three repairs are tried in a fixed order because at one and the
        # same position their ranking is fixed, whatever the exact numbers in
        # the tables are:
        #   substitution beats deletion  — same base, and a substitution
        #                                  penalty is always below the
        #                                  matching insert/delete penalty
        #   deletion beats insertion     — same penalty, but insertion loses
        #                                  a character off the base
        # So the first one that fits is the best one here, and there is no
        # need to score all three.

        # Substitution: the typed character is wrong. Both sides move on.
        # Only possible if there is actually a sentence character sitting
        # under the typo — the walk above can also stop because the sentence
        # ran out, and you cannot substitute for a character that isn't there.
        if sentence_i < sentence_length and self._tail_matches(
            query, sentence, query_i + 1, sentence_i + 1
        ):
            return MatchResult(
                MatchKind.SUBSTITUTION, error_position, query_length
            )

        # Deletion: a character the sentence has is missing from the query,
        # so only the sentence moves on.
        if self._tail_matches(query, sentence, query_i, sentence_i + 1):
            return MatchResult(
                MatchKind.DELETION, error_position, query_length
            )

        # Insertion: the query has a character the sentence does not, so only
        # the query moves on — and that character lands on nothing, which is
        # why the base score drops by one character's worth.
        if self._tail_matches(query, sentence, query_i + 1, sentence_i):
            return MatchResult(
                MatchKind.INSERTION, error_position, query_length - 1
            )

        return None

    @staticmethod
    def _tail_matches(query: str, sentence: str, query_i: int, sentence_i: int) -> bool:
        """True if the rest of the query matches from here with no further edits."""
        query_length = len(query)
        sentence_length = len(sentence)
        while (
            query_i < query_length
            and sentence_i < sentence_length
            and query[query_i] == sentence[sentence_i]
        ):
            query_i += 1
            sentence_i += 1
        return query_i == query_length
