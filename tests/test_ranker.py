"""Tests for domain/ranker.py."""
from autocomplete.domain.models import AutoCompleteData
from autocomplete.domain.ranker import TopKRanker

ranker = TopKRanker()


def _result(sentence, score, path="corpus.txt", offset=0):
    return AutoCompleteData(sentence, path, offset, score)


def test_returns_top_k_by_score_descending():
    results = [_result("d", 4), _result("a", 10), _result("c", 6), _result("b", 8), _result("e", 2), _result("f", 1)]
    ranked = ranker.rank(results, k=5)
    assert [r.completed_sentence for r in ranked] == ["a", "b", "c", "d", "e"]


def test_exact_ties_break_alphabetically():
    results = [_result("banana", 10), _result("apple", 10), _result("cherry", 10)]
    ranked = ranker.rank(results, k=5)
    assert [r.completed_sentence for r in ranked] == ["apple", "banana", "cherry"]


def test_fewer_than_k_available_returns_all_of_them():
    results = [_result("only one", 7)]
    ranked = ranker.rank(results, k=5)
    assert len(ranked) == 1
    assert ranked[0].completed_sentence == "only one"


def test_duplicate_results_are_counted_once():
    dup_a = _result("same sentence", 9, path="a.txt", offset=3)
    dup_b = AutoCompleteData("same sentence", "a.txt", 3, 9)
    results = [dup_a, dup_b, _result("other", 5)]
    ranked = ranker.rank(results, k=5)
    assert [r.completed_sentence for r in ranked] == ["same sentence", "other"]


def test_empty_input_returns_empty_list():
    assert ranker.rank([], k=5) == []
