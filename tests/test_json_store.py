"""Round-trip and correctness tests for JsonStore."""
from __future__ import annotations

from pathlib import Path

import pytest

from autocomplete.application.builder import CorpusBuilder
from autocomplete.domain.normalizer import DEFAULT_PIPELINE
from autocomplete.infrastructure.corpus_reader import CorpusReader
from autocomplete.infrastructure.index.ngram_index import NGramIndex
from autocomplete.infrastructure.storage.json_store import JsonStore

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus"


def _build():
    index = NGramIndex()
    corpus, _stats = CorpusBuilder(DEFAULT_PIPELINE, index).build(
        CorpusReader().read(FIXTURE)
    )
    return corpus, index


def test_round_trip_preserves_the_corpus(tmp_path):
    corpus, index = _build()
    target = tmp_path / "index.json"

    JsonStore().save((corpus, index), target)
    loaded_corpus, loaded_index = JsonStore().load(target)

    assert loaded_corpus.normalized == corpus.normalized
    assert loaded_corpus.raw_text == corpus.raw_text
    assert loaded_corpus.path_ids == corpus.path_ids
    assert loaded_corpus.line_numbers == corpus.line_numbers
    assert loaded_corpus.paths == corpus.paths


def test_loaded_index_answers_queries_identically(tmp_path):
    corpus, index = _build()
    target = tmp_path / "index.json"
    JsonStore().save((corpus, index), target)
    _loaded_corpus, loaded_index = JsonStore().load(target)

    for query in ("to be", "beg", "recommended", "gu"):
        normalized = DEFAULT_PIPELINE.normalize(query)
        assert set(loaded_index.candidates(normalized)) == set(index.candidates(normalized))
        assert set(loaded_index.exact_candidates(normalized)) == set(
            index.exact_candidates(normalized)
        )


def test_the_file_is_actually_json(tmp_path):
    corpus, index = _build()
    target = tmp_path / "index.json"
    JsonStore().save((corpus, index), target)

    import json
    with open(target) as f:
        document = json.load(f)  # must not raise
    assert "corpus" in document and "index" in document


def test_no_temporary_file_is_left_behind(tmp_path):
    corpus, index = _build()
    target = tmp_path / "index.json"
    JsonStore().save((corpus, index), target)
    assert list(tmp_path.glob("*.tmp")) == []


def test_loading_a_missing_index_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline build"):
        JsonStore().load(tmp_path / "nope.json")
