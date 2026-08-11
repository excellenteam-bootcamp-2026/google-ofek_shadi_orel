"""End-to-end test for application/engine.py.

Ofek's and Orel's real collaborators (`OneEditMatcher`, `NGramIndex`, a
real `Normalizer`, a real corpus builder) live on their own branches, not
this one, so this test wires `AutoCompleteEngine` against small fakes for
those four roles instead. The scorer and ranker are this track's own,
already-tested code (`RuleBasedScorer`, `TopKRanker`), so they're used for
real rather than faked.

This is a plumbing test, not a substitute for the real integration test
against Ofek's and Orel's actual implementations once their branches
merge -- see the agent report.
"""
from __future__ import annotations

from autocomplete.application.engine import AutoCompleteEngine
from autocomplete.domain.models import AutoCompleteData
from autocomplete.domain.ports import MatchKind, MatchResult, RawLine
from autocomplete.domain.ranker import TopKRanker
from autocomplete.domain.scorer import RuleBasedScorer

# sentence_id -> (normalized_sentence, RawLine as it appears on disk)
_CORPUS = {
    0: (
        "to be or not to be that is the question",
        RawLine("To be or not to be, that is the question.", "hamlet.txt", 1),
    ),
    1: ("or not to be wise", RawLine("Or not to be wise.", "hamlet.txt", 2)),
    2: ("to be free", RawLine("To be free.", "hamlet.txt", 3)),
    3: ("completely unrelated line", RawLine("Completely unrelated line.", "hamlet.txt", 4)),
}

# sentence_id -> canned MatchResult (None means "matcher found no match")
_MATCHES = {
    0: MatchResult(MatchKind.EXACT, 0, 5),        # -> score 10
    1: MatchResult(MatchKind.INSERTION, 4, 6),    # -> score 8
    2: MatchResult(MatchKind.SUBSTITUTION, 1, 5), # -> score 5
    3: None,
}


class FakeIndex:
    def candidates(self, normalized_query):
        return list(_CORPUS.keys())


class FakeMatcher:
    def match(self, normalized_query, normalized_sentence):
        for sentence_id, (text, _raw) in _CORPUS.items():
            if text == normalized_sentence:
                return _MATCHES[sentence_id]
        return None


class FakeCorpus:
    def get(self, sentence_id):
        return _CORPUS[sentence_id]


class FakeNormalizer:
    def normalize(self, text):
        return text.lower().strip().rstrip(".,")


def _build_engine() -> AutoCompleteEngine:
    return AutoCompleteEngine(
        normalizer=FakeNormalizer(),
        index=FakeIndex(),
        matcher=FakeMatcher(),
        scorer=RuleBasedScorer(),
        corpus=FakeCorpus(),
        ranker=TopKRanker(),
    )


def test_results_are_ordered_by_score_and_non_matches_are_excluded():
    engine = _build_engine()
    results = engine.get_best_k_completions("to be")

    assert [r.score for r in results] == [10, 8, 5]
    assert [r.completed_sentence for r in results] == [
        "To be or not to be, that is the question.",
        "Or not to be wise.",
        "To be free.",
    ]
    # sentence_id 3 ("completely unrelated line") has no match and must
    # not appear in the output at all.
    assert all(r.offset != 4 for r in results)


def test_source_metadata_is_carried_through_from_the_raw_line():
    engine = _build_engine()
    top = engine.get_best_k_completions("to be")[0]

    assert isinstance(top, AutoCompleteData)
    assert top.source_text == "hamlet.txt"
    assert top.offset == 1
