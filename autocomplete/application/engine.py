"""Facade over the autocomplete pipeline: normalize -> filter -> match ->
score -> rank.

Every collaborator here is a `ports.py` Protocol type, injected through
the constructor -- never a concrete teammate class imported directly.
That is what lets this file be built and fully tested today, against
fakes, before the real `NGramIndex`/`OneEditMatcher`/etc. exist on this
branch, and lets any of them be swapped later (including for a C++
adapter in Phase B) without a single line changing here.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from autocomplete.domain.models import AutoCompleteData
from autocomplete.domain.ports import Matcher, Normalizer, RawLine, Scorer, SentenceIndex
from autocomplete.domain.ranker import TopKRanker


@runtime_checkable
class Corpus(Protocol):
    """Maps a sentence_id back to its normalized form and original line.

    Stand-in for the `Corpus` protocol proposed to the team for addition
    to `domain/ports.py` (see the agent report) -- `ports.py` is frozen
    and off-limits on this branch, so this lives here until the team
    accepts the proposal and it can move to its real home.
    """

    def get(self, sentence_id: int) -> tuple[str, RawLine]: ...


class AutoCompleteEngine:
    """The one method the whole assignment is graded on."""

    def __init__(
        self,
        normalizer: Normalizer,
        index: SentenceIndex,
        matcher: Matcher,
        scorer: Scorer,
        corpus: Corpus,
        ranker: TopKRanker,
    ) -> None:
        self._normalizer = normalizer
        self._index = index
        self._matcher = matcher
        self._scorer = scorer
        self._corpus = corpus
        self._ranker = ranker

    def get_best_k_completions(self, prefix: str) -> list[AutoCompleteData]:
        normalized_query = self._normalizer.normalize(prefix)
        results: list[AutoCompleteData] = []
        for sentence_id in self._index.candidates(normalized_query):
            normalized_sentence, raw = self._corpus.get(sentence_id)
            match = self._matcher.match(normalized_query, normalized_sentence)
            if match is None:
                continue
            score = self._scorer.score(match)
            results.append(AutoCompleteData(raw.text, raw.path, raw.line_no, score))
        return self._ranker.rank(results, k=5)
