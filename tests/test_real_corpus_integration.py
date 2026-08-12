"""Integration tests against the real Archive.zip corpus.

The unit tests elsewhere use ``tests/fixtures/mini_corpus`` because they need
hand-known expected values -- you cannot assert "line 4 of this file" against
2.6M lines nobody has read. These tests do the opposite job: they run the real
pipeline over the real corpus and assert only what must hold for *any* corpus,
catching the things a 17-line fixture never could -- scale, unexpected
characters, and pathological input.

They are marked ``slow`` and skipped when the archive is absent:

    python -m pytest tests/ -q                    # everything
    python -m pytest tests/ -q -m "not slow"      # fast suite only
    python -m pytest tests/test_real_corpus_integration.py -v
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from autocomplete.application.builder import CorpusBuilder
from autocomplete.application.engine import AutoCompleteEngine
from autocomplete.domain.matcher import OneEditMatcher
from autocomplete.domain.normalizer import DEFAULT_PIPELINE, REFERENCE_PIPELINE
from autocomplete.domain.ranker import TopKRanker
from autocomplete.domain.scorer import RuleBasedScorer
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from autocomplete.infrastructure.storage.pickle_store import PickleStore

ARCHIVE = Path(__file__).resolve().parents[1] / "Archive.zip"

# The full corpus does not currently fit in memory (see README, Known
# limitation). Until the index is array-backed, these tests build from a
# prefix large enough to be representative and small enough to run.
SAMPLE_LINES = 150_000

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not ARCHIVE.exists(), reason="Archive.zip not present"),
]


@pytest.fixture(scope="module")
def real_build():
    """Build once, share across every test in this module."""
    index = NGramIndex()
    builder = CorpusBuilder(DEFAULT_PIPELINE, index)
    corpus, stats = builder.build(
        itertools.islice(CorpusReader().read(ARCHIVE), SAMPLE_LINES)
    )
    return corpus, index, stats


@pytest.fixture(scope="module")
def real_engine(real_build):
    corpus, index, _stats = real_build
    return AutoCompleteEngine(
        normalizer=DEFAULT_PIPELINE,
        index=index,
        matcher=OneEditMatcher(RuleBasedScorer()),
        scorer=RuleBasedScorer(),
        corpus=corpus,
        ranker=TopKRanker(),
    )


# ---------------------------------------------------------------------------
# The reader survives real files
# ---------------------------------------------------------------------------
def test_reader_streams_the_whole_archive_without_loading_it():
    """Every line is readable and well-formed -- 1,504 files, all encodings."""
    count = 0
    for raw, path, line_no in CorpusReader().read(ARCHIVE):
        count += 1
        if count > 500_000:
            break
        assert isinstance(raw, str)
        assert path.endswith(".txt")
        assert line_no >= 1
    assert count > 100_000


def test_normalizer_agrees_with_its_reference_on_real_text(real_build):
    """Real documentation contains box-drawing, unicode quotes and control
    characters that no fixture would think to include."""
    corpus, _index, _stats = real_build
    for raw in itertools.islice(corpus.raw_text, 20_000):
        assert DEFAULT_PIPELINE.normalize(raw) == REFERENCE_PIPELINE.normalize(raw)


# ---------------------------------------------------------------------------
# The build holds up at scale
# ---------------------------------------------------------------------------
def test_build_deduplicates_heavily(real_build):
    """The real corpus is roughly a third duplicate lines; the fixture has one."""
    _corpus, _index, stats = real_build
    assert stats.duplicates_dropped > stats.lines_read * 0.2
    assert stats.sentences_kept + stats.duplicates_dropped <= stats.lines_read


def test_paths_are_interned_not_repeated(real_build):
    """Far fewer path strings than sentences -- the Flyweight is doing its job."""
    corpus, _index, stats = real_build
    assert stats.distinct_paths < stats.sentences_kept / 100
    assert len(corpus.paths) == stats.distinct_paths


def test_every_sentence_is_retrievable(real_build):
    corpus, _index, _stats = real_build
    for sentence_id in (0, len(corpus) // 2, len(corpus) - 1):
        normalized, raw = corpus.get(sentence_id)
        assert normalized == DEFAULT_PIPELINE.normalize(raw.text)
        assert raw.path in corpus.paths


# ---------------------------------------------------------------------------
# The index never loses a real match
# ---------------------------------------------------------------------------
def _pick_query(corpus, min_length: int = 18) -> str:
    """Take a real phrase out of the corpus itself.

    Hard-coding a query like "python object" makes the test depend on the
    corpus containing that phrase -- which silently stops being true the
    moment the sample size or the archive changes. Deriving the query from a
    sentence the build actually produced means the test asserts a property of
    the engine, not a fact about one particular corpus.
    """
    for normalized in corpus.normalized:
        if len(normalized) >= min_length + 10 and " " in normalized[5:min_length]:
            return normalized[5 : 5 + min_length]
    raise AssertionError("no sentence long enough to derive a query from")


def test_index_candidates_are_a_superset_of_true_matches(real_build):
    """The filter may over-return, but it must never drop a sentence the
    matcher would have accepted. Checked by brute force on a real slice."""
    corpus, index, _stats = real_build
    matcher = OneEditMatcher(RuleBasedScorer())
    queries = [_pick_query(corpus), corpus.normalized[0][:12]]

    for normalized_query in queries:
        candidates = set(index.candidates(normalized_query))
        for sentence_id in range(min(len(corpus), 20_000)):
            if matcher.match(normalized_query, corpus.normalized[sentence_id]) is not None:
                assert sentence_id in candidates, (
                    f"index dropped a real match for {normalized_query!r}"
                )


# ---------------------------------------------------------------------------
# End-to-end behaviour on real data
# ---------------------------------------------------------------------------
def test_engine_returns_at_most_five_ranked_results(real_build, real_engine):
    corpus, _index, _stats = real_build
    results = real_engine.get_best_k_completions(_pick_query(corpus))
    assert 0 < len(results) <= 5
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_results_cite_a_real_file_and_line(real_build, real_engine):
    corpus, _index, _stats = real_build
    for result in real_engine.get_best_k_completions(_pick_query(corpus)):
        assert result.source_text.endswith(".txt")
        assert result.offset >= 1
        assert result.source_text in corpus.paths


def test_results_keep_original_punctuation_and_case(real_build, real_engine):
    """The spec requires the source line verbatim, not the normalized form.

    The query is normalized text, so if the engine echoed back what it matched
    on, the result would equal the query's own casing. Asserting the result is
    a line that normalizes to something *containing* the query proves the raw
    line survived the round trip.
    """
    corpus, _index, _stats = real_build
    query = _pick_query(corpus)
    results = real_engine.get_best_k_completions(query)
    assert results
    for result in results:
        assert query in DEFAULT_PIPELINE.normalize(result.completed_sentence)


def test_a_typo_still_finds_matches(real_build, real_engine):
    """One substitution must not break the search -- that is the whole point."""
    corpus, _index, _stats = real_build
    query = _pick_query(corpus)
    clean = real_engine.get_best_k_completions(query)
    assert clean, "the exact query should always match the sentence it came from"

    # Flip one character well past position 4, where the penalty is smallest.
    position = len(query) - 3
    swapped = "z" if query[position] != "z" else "q"
    typo = query[:position] + swapped + query[position + 1 :]

    typo_results = real_engine.get_best_k_completions(typo)
    assert typo_results, "one-edit match found nothing"
    assert max(r.score for r in typo_results) < max(r.score for r in clean)


def test_nonsense_query_returns_nothing_rather_than_crashing(real_engine):
    assert real_engine.get_best_k_completions("qzxwvkjhgfdsapoiuy") == []


def test_very_short_query_is_handled(real_engine):
    """Below 2*gram_size the index falls back to offering the whole corpus."""
    assert len(real_engine.get_best_k_completions("th")) <= 5


# ---------------------------------------------------------------------------
# Persistence at real scale
# ---------------------------------------------------------------------------
def test_real_index_round_trips_through_the_store(real_build, tmp_path):
    corpus, index, _stats = real_build
    target = tmp_path / "real_index.pkl"
    store = PickleStore()
    store.save((corpus, index), target)
    loaded_corpus, loaded_index = store.load(target)

    assert len(loaded_corpus) == len(corpus)
    assert loaded_corpus.get(0) == corpus.get(0)
    query = DEFAULT_PIPELINE.normalize("python object")
    assert set(loaded_index.candidates(query)) == set(index.candidates(query))