"""Tests for index persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

from autocomplete.application.builder import CorpusBuilder
from autocomplete.domain.normalizer import DEFAULT_PIPELINE
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from autocomplete.infrastructure.storage.pickle_store import PickleStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus"


def test_round_trip_preserves_the_corpus(tmp_path):
    index = NGramIndex()
    corpus, _stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(
        CorpusReader().read(FIXTURE)
    )
    target = tmp_path / "nested" / "index.pkl"

    store = PickleStore()
    store.save((corpus, index), target)
    loaded_corpus, loaded_index = store.load(target)

    assert loaded_corpus.normalized == corpus.normalized
    assert loaded_corpus.paths == corpus.paths
    assert loaded_corpus.get(0) == corpus.get(0)
    assert len(loaded_index) == len(index)


def test_loaded_index_still_answers_queries(tmp_path):
    index = NGramIndex()
    corpus, _stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(
        CorpusReader().read(FIXTURE)
    )
    target = tmp_path / "index.pkl"
    PickleStore().save((corpus, index), target)
    _loaded_corpus, loaded_index = PickleStore().load(target)

    query = DEFAULT_PIPELINE.normalize("to be or not")
    assert set(loaded_index.candidates(query)) == set(index.candidates(query))


def test_save_creates_missing_directories(tmp_path):
    target = tmp_path / "a" / "b" / "c" / "index.pkl"
    PickleStore().save({"x": 1}, target)
    assert target.exists()


def test_no_temporary_file_is_left_behind(tmp_path):
    target = tmp_path / "index.pkl"
    PickleStore().save({"x": 1}, target)
    assert list(tmp_path.glob("*.tmp")) == []


def test_loading_a_missing_index_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline build"):
        PickleStore().load(tmp_path / "nope.pkl")
