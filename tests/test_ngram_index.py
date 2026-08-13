"""Tests for NGramIndex.

The index is a filter, so there is only one thing it must never do: lose a
sentence that really matches. Returning extra sentences is allowed and
expected — the matcher throws those out. Most of this file is about that
asymmetry.
"""

from __future__ import annotations

import pytest

from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from tests.doubles import AppendixScorer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "a cat and a dog are friends",
    "python decorators wrap functions",
]


@pytest.fixture
def index() -> NGramIndex:
    built = NGramIndex()
    for sentence_id, sentence in enumerate(CORPUS):
        built.add(sentence_id, sentence)
    return built


@pytest.fixture
def matcher() -> OneEditMatcher:
    return OneEditMatcher(AppendixScorer())


# --- the index actually narrows things down -------------------------------


def test_exact_query_keeps_the_right_sentence_and_drops_the_rest(index):
    candidates = set(index.candidates("sat on the mat"))
    assert 0 in candidates
    assert 2 not in candidates
    assert 3 not in candidates


def test_a_query_from_one_sentence_does_not_drag_in_unrelated_ones(index):
    candidates = set(index.candidates("decorators wrap"))
    assert candidates == {3}


# --- the pigeonhole split, from both sides --------------------------------


def test_typo_in_the_second_half_is_rescued_by_the_first(index):
    """'the mut' is nowhere in the corpus, so only 'sat on ' can find this."""
    assert 0 in set(index.candidates("sat on the mut"))


def test_typo_in_the_first_half_is_rescued_by_the_second(index):
    """'sut on ' is nowhere in the corpus, so only 'the mat' can find this."""
    assert 0 in set(index.candidates("sut on the mat"))


def test_a_half_that_is_nowhere_in_the_corpus_contributes_nothing(index):
    """Both halves unknown means nothing to offer — not a crash."""
    assert set(index.candidates("zzzzzzz qqqqqqq")) == set()


# --- short queries, which have no n-gram of their own ---------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ca", {0, 2}),   # 'the cat', 'a cat and a dog'
        ("rug", {1}),
        ("z", set()),     # nowhere in the corpus
    ],
)
def test_short_substrings_are_found_through_the_grams_that_contain_them(
    index, query, expected
):
    """One and two character strings still resolve exactly.

    They are too short to be gram keys themselves, so the lookup unions the
    postings of every gram that contains them. No second index needed.
    """
    assert set(index._containing(query)) == expected


def test_queries_too_short_to_split_still_narrow_the_corpus(index):
    """These used to fall back to the entire corpus. They no longer do."""
    everything = set(range(len(CORPUS)))
    assert set(index.candidates("cats")) < everything
    assert set(index.candidates("cat s")) < everything


@pytest.mark.parametrize("query", ["c", "ca", "cat"])
def test_very_short_queries_still_terminate_and_stay_sound(index, query):
    """A one-character query has a substitution variant for nearly every
    other character, so on a small corpus like this one it will often
    still touch every sentence — but through ~100-300 direct lookups now,
    not by handing over `_all_ids` unconditionally. See the test below for
    a case where that distinction actually shows up as fewer candidates.
    """
    assert set(index.candidates(query)) <= set(range(len(CORPUS)))


def test_a_typo_at_the_unsplittable_length_still_excludes_real_non_matches(
    index, matcher
):
    """'cot' (length 3, too short to split) is a typo for 'cat'. Sentence 1
    ("the dog sat on the rug") shares no such alignment with it — proof this
    isn't silently falling back to the whole corpus, the way it used to.
    """
    candidates = set(index.candidates("cot"))
    assert 1 not in candidates
    assert matcher.match("cot", CORPUS[1]) is None      # confirms the exclusion is correct
    assert matcher.match("cot", CORPUS[0]) is not None  # and that real matches are still found
    assert 0 in candidates


def test_a_length_that_cannot_split_still_returns_nothing_for_the_unrelated(index):
    assert set(index.candidates("zzz")) == set()


def test_sentences_shorter_than_one_gram_are_not_lost(index):
    """A sentence with no grams has no postings, so it has to be added back."""
    index.add(99, "ok")
    assert 99 in set(index._containing("ok"))
    assert 99 in set(index._containing("o"))


def test_empty_query_returns_nothing(index):
    assert set(index.candidates("")) == set()
    assert set(index.exact_candidates("")) == set()


# --- the exact fast path --------------------------------------------------


def test_exact_candidates_finds_a_query_typed_correctly(index):
    assert 0 in set(index.exact_candidates("sat on the mat"))


def test_exact_candidates_is_blind_to_typos(index):
    """One wrong character poisons up to gram_size grams, and the
    intersection needs all of them — so this finds nothing. That emptiness
    is the signal to fall through to `candidates`, not a failure.
    """
    assert set(index.exact_candidates("sat on the mut")) == set()
    assert 0 in set(index.candidates("sat on the mut"))


def test_exact_candidates_is_never_wider_than_candidates(index):
    """The fast path may only ever be a subset of the tolerant path.

    If it could offer something `candidates` does not, taking the shortcut
    would change which results come back, not just how fast.
    """
    for query in ["sat on the mat", "the dog", "cats", "decorators wrap"]:
        assert set(index.exact_candidates(query)) <= set(index.candidates(query))


def test_exact_candidates_works_below_the_split_threshold(index):
    """The fast path only needs one gram's worth of query, not two."""
    assert set(index.exact_candidates("cat")) == {0, 2}
    assert set(index.exact_candidates("ca")) == {0, 2}


# --- the property that actually matters -----------------------------------


ALPHABET = "aeioustx "


def one_edit_variants(text: str):
    """Every string one edit away from ``text``, over a small alphabet."""
    for i in range(len(text)):
        for character in ALPHABET:
            if character != text[i]:
                yield text[:i] + character + text[i + 1 :]
    for i in range(len(text)):
        yield text[:i] + text[i + 1 :]
    for i in range(len(text) + 1):
        for character in ALPHABET:
            yield text[:i] + character + text[i:]


def test_the_filter_never_loses_a_sentence_that_really_matches(index, matcher):
    """Cross-check the filter against the verifier, exhaustively.

    For every one-edit corruption of every substring, if the matcher says it
    matches the sentence it came from, the index must have offered that
    sentence. A failure here means real completions are silently missing.
    """
    checked = 0
    for sentence_id, sentence in enumerate(CORPUS):
        for start in range(0, len(sentence) - 8, 3):
            for query in one_edit_variants(sentence[start : start + 8]):
                if matcher.match(query, sentence) is None:
                    continue
                checked += 1
                assert sentence_id in set(index.candidates(query)), (
                    f"index lost sentence {sentence_id} for query {query!r}"
                )
    # Guard against the loop silently doing nothing.
    assert checked > 500
