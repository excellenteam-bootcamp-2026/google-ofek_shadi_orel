"""Tests for the offline build: dedup, path interning, and the corpus contract."""
from __future__ import annotations

from pathlib import Path

import pytest

from autocomplete.application.builder import BuiltCorpus, CorpusBuilder
from autocomplete.domain.normalizer import DEFAULT_PIPELINE
from autocomplete.domain.ports import RawLine
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus"


@pytest.fixture
def built():
    index = NGramIndex()
    builder = CorpusBuilder(DEFAULT_PIPELINE, index)
    corpus, stats = builder.build(CorpusReader().read(FIXTURE))
    return corpus, index, stats


def test_duplicate_sentences_are_dropped(built):
    """hamlet.txt and reference/quotes.txt share a line verbatim."""
    corpus, _index, stats = built
    assert stats.duplicates_dropped == 1
    assert stats.sentences_kept == stats.lines_read - stats.duplicates_dropped


def test_first_occurrence_keeps_the_citation(built):
    """The surviving copy must cite where it was FIRST seen, not last."""
    corpus, _index, _stats = built
    target = DEFAULT_PIPELINE.normalize("To be or not to be, that is the question.")
    sentence_id = corpus.normalized.index(target)
    _norm, raw = corpus.get(sentence_id)
    assert raw.path.endswith("hamlet.txt")
    assert raw.line_no == 1


def test_paths_are_interned(built):
    """One string per file, not one per sentence."""
    corpus, _index, stats = built
    assert stats.distinct_paths == 6
    assert len(corpus.paths) == 6
    assert len(corpus.path_ids) == stats.sentences_kept


def test_get_returns_the_original_line_not_the_normalized_one(built):
    """The spec requires output in its original form, punctuation included."""
    corpus, _index, _stats = built
    _norm, raw = corpus.get(0)
    assert isinstance(raw, RawLine)
    assert raw.text == "To be or not to be, that is the question."


def test_every_sentence_reached_the_index(built):
    corpus, index, _stats = built
    assert len(index) == len(corpus)


def test_blank_and_punctuation_only_lines_never_enter_the_corpus():
    """'!!!' normalizes to empty and can never match a query."""
    index = NGramIndex()
    corpus, stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(
        [("!!!", "junk.txt", 1), ("   ", "junk.txt", 2), ("real line", "junk.txt", 3)]
    )
    assert stats.sentences_kept == 1
    assert corpus.normalized == ["real line"]


def test_build_consumes_lazily():
    """The builder must not materialise the reader's output."""
    consumed = []

    def spy():
        for record in CorpusReader().read(FIXTURE):
            consumed.append(record)
            yield record

    CorpusBuilder(DEFAULT_PIPELINE, NGramIndex()).build(spy())
    assert consumed  # generator was driven, not list()-ed up front
